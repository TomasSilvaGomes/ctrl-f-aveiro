from __future__ import annotations

import base64
import csv
import html
import io
import json
import logging
import os
import re
import sqlite3
import tempfile
from collections import Counter
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from threading import Lock
from urllib.parse import quote
from zoneinfo import ZoneInfo

import streamlit as st
import yaml
from PIL import Image
from streamlit_folium import st_folium

from catalog_map import (
    CATEGORY_ACCENTS, CATEGORY_LABELS, build_map, build_marker_layer, group_points,
    marker_data, marker_icon, point_bounds,
)
from contributions import (
    MAX_UPLOAD_MB, PHOTO_CATEGORY, SOURCE, UPLOAD_TYPES, ContributionStore,
    PrivacyModelsMissing, decode_photo, photo_camera, photo_coordinates,
)
from mapillary_photos import image_key, mapillary_token, photo_link
from photo_preview import (
    SAM2_MODEL, PhotoPreview, PreviewQueue, SegmentationError, catalog_photo_views, has_catalog_preview,
    load_catalog_preview, sam_available,
)
from photo_scene import build_catalog_scene, build_contribution_scene, classified_objects
from place_search import find_place, place_suggestions


APP_DIR = Path(__file__).resolve().parent          # ficheiros da app (ui.css, marker_icons.css)
MENU_MOTION_JS = (APP_DIR / "menu_motion.js").read_text(encoding="utf-8")
LOGO_PATH = APP_DIR / "logo.png"
ROOT = APP_DIR.parent                              # raiz do projeto (data/, config/)
MUNICIPALITY = os.environ.get("CTRL_F_MUNICIPALITY", "Aveiro").strip() or "Aveiro"
# Por omissão o catálogo para a app (scripts/imagens.py app): o mesmo catálogo + campo "imagem" com
# o recorte já desenhado de cada elemento. Sem ele, usa o catálogo da pipeline.
_APP_CATALOG = ROOT / "data" / "app" / "catalogo.geojson"
CATALOG_PATH = Path(
    os.environ.get("CTRL_F_CATALOG_PATH")
    or str(_APP_CATALOG if _APP_CATALOG.is_file() else ROOT / "data" / "catalogo" / "catalogo.geojson")
).expanduser()
if not CATALOG_PATH.is_absolute():
    CATALOG_PATH = ROOT / CATALOG_PATH
CSV_PATH = CATALOG_PATH.with_name("catalogo.csv")
CONTRIBUTIONS = ContributionStore(CATALOG_PATH.with_name("contribuicoes.sqlite3"))
LOGGER = logging.getLogger(__name__)
CATEGORY_GLYPHS = {
    "sinalizacao": "signpost",
    "semaforos": "traffic",
    "iluminacao": "lightbulb",
    "residuos": "recycling",
    PHOTO_CATEGORY: "photo_camera",
}
SUBTYPE_FILTER_PREFIX = "catalog-subtype-"
REVIEW_STATUS = {"pendente": "Por rever", "aprovado": "Aprovado", "rejeitado": "Rejeitado"}
REVIEW_ACTIONS = {"submetido": "Recebido", **REVIEW_STATUS}
REVIEW_SOURCES = {"ficheiro": "Ficheiro", "mapillary": "Mapillary"}
CSV_COLUMNS = [
    "id", "categoria", "categoria_label", "subtipo", "label", "codigo", "valor",
    "lon", "lat", "confianca", "concordancia", "n_vistas", "n_imagens", "membros",
    "imagem_id", "imagem_url", "autor", "licenca", "fonte", "estado",
]

st.set_page_config(
    page_title="IAMU",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_data(show_spinner=False)
def _read_catalog(path, modified_ns, size) -> dict:
    with path.open(encoding="utf-8") as file:
        collection = json.load(file)
    if collection.get("type") != "FeatureCollection":
        raise ValueError(f"Formato GeoJSON inválido: {path}")
    features = collection.get("features", [])
    for feature in features:
        props = feature["properties"]
        if props.get("fonte") == "Mapillary":
            props["fonte"] = "Deteções do catálogo"
    return collection


def load_catalog_document() -> dict:
    stat = CATALOG_PATH.stat()
    return _read_catalog(CATALOG_PATH, stat.st_mtime_ns, stat.st_size)


def load_catalog() -> list[dict]:
    return load_catalog_document().get("features", [])


def csv_bytes(features: list[dict], *, include_status=True) -> bytes:
    fields = [name for name in CSV_COLUMNS if include_status or name != "estado"]
    extra_fields = sorted(
        {
            key
            for feature in features
            for key in feature.get("properties", {})
            if key not in fields and (include_status or key != "estado")
        }
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields + extra_fields, extrasaction="ignore")
    writer.writeheader()
    for feature in features:
        row = dict(feature.get("properties", {}))
        coordinates = feature.get("geometry", {}).get("coordinates", [])
        if len(coordinates) >= 2:
            row["lon"], row["lat"] = coordinates[:2]
        for key, value in row.items():
            if isinstance(value, (list, dict)):
                row[key] = json.dumps(value, ensure_ascii=False)
        writer.writerow(row)
    return output.getvalue().encode("utf-8-sig")


def feature_label(feature: dict) -> str:
    props = feature.get("properties", {})
    return props.get("label") or props.get("subtipo") or "Elemento"


@st.cache_resource
def catalog_write_lock():
    return Lock()


def save_correction(feature: dict, *, changes=None, coordinates=None, removed_photo=None, remove_object=False) -> bool:
    props = feature["properties"]

    def edit(current):
        return {
            **current,
            "properties": {**current["properties"], **changes},
            "geometry": {**current["geometry"], "coordinates": coordinates},
        }

    if props.get("fonte") == SOURCE:
        return CONTRIBUTIONS.update(
            props["id"], edit, remove_photo=removed_photo is not None, remove_object=remove_object,
        )

    with catalog_write_lock():
        collection = load_catalog_document()
        features = collection.get("features", [])
        index = next((i for i, item in enumerate(features) if item["properties"]["id"] == props["id"]), None)
        if index is None:
            return False
        removed = []
        if remove_object:
            removed = [features.pop(index)]
        elif removed_photo is not None:
            kept = []
            for item in features:
                if item["properties"]["id"] == props["id"] or uses_catalog_photo(item["properties"], removed_photo):
                    removed.append(item)
                else:
                    kept.append(item)
            features = kept
        else:
            features[index] = edit(features[index])
        if removed:
            removed_segments = collection.setdefault("segmentacoes_removidas", {})
            for item in removed:
                for image_id, box in catalog_detection_boxes(item["properties"]):
                    boxes = removed_segments.setdefault(image_id, [])
                    if box not in boxes:
                        boxes.append(box)
        collection["features"] = features
        outputs = (
            (CATALOG_PATH, json.dumps(collection, ensure_ascii=False, separators=(",", ":")).encode("utf-8")),
            (CSV_PATH, csv_bytes(features)),
        )
        temporary_paths = []
        try:
            for target, content in outputs:
                with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", delete=False) as file:
                    temporary_paths.append((Path(file.name), target))
                    file.write(content)
            for temporary, target in temporary_paths:
                os.replace(temporary, target)
        finally:
            for temporary, _ in temporary_paths:
                temporary.unlink(missing_ok=True)
    return True


@st.cache_resource
def get_analyzer():
    """Reutiliza a pipeline de análise e os modelos de anonimização entre envios."""
    from photo_analysis import PhotoAnalyzer

    return PhotoAnalyzer()


def close_upload():
    st.session_state["upload_open"] = False
    st.session_state.pop("upload_errors", None)


def finish_submission(results, revision):
    errors = [message for level, message in results if level == "error"]
    st.session_state["upload_errors"] = errors
    st.session_state["upload_messages"] = [(level, message) for level, message in results if level != "error"]
    st.session_state["upload_open"] = bool(errors)
    if not errors:
        st.session_state["upload_revision"] = revision + 1
    st.rerun()


def submit_photo(digest, data, coordinates, location_source, *, name, on_step=None):
    analyzer = get_analyzer()
    # rumo e focal do EXIF: com eles cada objeto fica à sua distância da câmara (senão no local da foto)
    camera = {**photo_camera(data), "lat": coordinates[0], "lon": coordinates[1]}
    with decode_photo(data) as image:
        panoramic = abs(image.width - 2 * image.height) <= 0.02 * image.height   # 360° equirretangular
        detections = analyzer.analyze(image, panoramic=panoramic, on_step=on_step, camera=camera)
        preview = None if detections else analyzer.preview_without_detections(image)
    return CONTRIBUTIONS.submit(
        digest, *coordinates, location_source, detections, name=name, source="ficheiro", preview=preview,
    )


@st.dialog("Adicionar fotografias", width="large", on_dismiss=close_upload)
def render_upload():
    st.caption("O GPS da foto situa os elementos no mapa. Os recortes são anonimizados antes de seguir para revisão.")
    for message in st.session_state.get("upload_errors", []):
        st.error(message)
    revision = st.session_state.get("upload_revision", 0)
    files = st.file_uploader(
        "Fotografias de mobiliário urbano",
        type=UPLOAD_TYPES,
        accept_multiple_files=True,
        key=f"photos-{revision}",
        help=f"JPEG, PNG, WebP ou HEIC, até {MAX_UPLOAD_MB} MB por fotografia.",
    )
    ready = []
    for index, file in enumerate(files):
        data = file.getvalue()
        try:
            coords = photo_coordinates(data)
        except ValueError as error:
            st.error(f"{file.name} · {error}")
            continue
        filename = html.escape(file.name)
        if coords is not None:
            st.html(f'<div class="upload-status"><span>GPS encontrado</span> · {filename}<br>{coords[0]:.6f}, {coords[1]:.6f}</div>')
            ready.append((file, coords, "gps_exif"))
            continue
        st.html(f'<div class="upload-status">{filename} · Indica a localização</div>')
        photo_key = f"{revision}-{index}-{sha256(data).hexdigest()[:12]}"
        lat_col, lon_col = st.columns(2)
        with lat_col:
            lat = st.number_input("Latitude", -90.0, 90.0, value=None, format="%.7f", key=f"photo-lat-{photo_key}")
        with lon_col:
            lon = st.number_input("Longitude", -180.0, 180.0, value=None, format="%.7f", key=f"photo-lon-{photo_key}")
        if lat is not None and lon is not None:
            ready.append((file, (lat, lon), "manual"))
    if not st.button(
        "Analisar e enviar", icon=":material/upload:", type="primary",
        disabled=not ready or len(ready) != len(files), width="stretch",
    ):
        return

    results = []
    total = len(ready)
    progress = st.progress(0.0, text="A carregar os modelos de análise (na 1.ª vez, sem GPU, pode demorar 1–2 min)…")
    try:
        get_analyzer().warm_up()
    except PrivacyModelsMissing as error:
        finish_submission([("error", str(error))], revision)
    except Exception:
        LOGGER.exception("Erro ao carregar os modelos de análise")
        finish_submission([("error", "Não foi possível carregar os modelos de análise (causa no terminal).")], revision)
    for index, (file, coordinates, location_source) in enumerate(ready):
        def on_step(fraction, text, index=index):
            progress.progress((index + fraction) / total, text=f"Fotografia {index + 1} de {total} · {text}…")

        on_step(0.0, "a ler")
        data = file.getvalue()
        digest = sha256(data).hexdigest()
        try:
            if CONTRIBUTIONS.has_photo(digest):
                results.append(("info", f"{file.name} · Esta fotografia já foi enviada."))
                continue
            request_id = submit_photo(digest, data, coordinates, location_source, name=file.name, on_step=on_step)
            if request_id:
                results.append(("success", f"{file.name} · Pedido {request_id} recebido e por rever."))
            else:
                results.append(("info", f"{file.name} · Esta fotografia já foi enviada."))
        except PrivacyModelsMissing as error:
            LOGGER.exception("Erro ao preparar a anonimização de uma contribuição fotográfica")
            results.append(("error", f"{file.name} · {error}"))
        except SegmentationError as error:
            LOGGER.exception("Erro na segmentação de uma contribuição fotográfica")
            results.append(("error", f"{file.name} · {error}"))
        except Exception:
            LOGGER.exception("Erro ao processar uma contribuição fotográfica")
            results.append(("error", f"{file.name} · Não foi possível analisar e enviar a fotografia."))
    progress.progress(1.0, text="Concluído")
    finish_submission(results, revision)


def photo_credit(props):
    key = image_key(props.get("imagem_id"))
    if not key:
        return ""
    credit = ""
    author = props.get("autor")
    if author:
        profile = quote(str(author), safe="")
        credit = f' · <a href="https://www.mapillary.com/app/user/{profile}">{html.escape(str(author))}</a>'
    return (
        '<details class="photo-credit"><summary aria-label="Créditos da fotografia" title="Créditos da fotografia">©</summary>'
        f'<p><a href="{photo_link(key)}">Origem da fotografia</a>{credit} · '
        '<a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a> · Imagem adaptada</p></details>'
    )


def render_photo(preview, props):
    photo = base64.b64encode(preview.image).decode("ascii")
    with Image.open(io.BytesIO(preview.image)) as image:
        ratio = image.width / image.height
    overlay = ""
    if preview.mask:
        mask = base64.b64encode(preview.mask).decode("ascii")
        overlay = f'<img class="photo-overlay" src="data:image/png;base64,{mask}" alt="">'
    elif preview.box:
        left, top, right, bottom = preview.box
        overlay = (
            f'<span class="photo-overlay photo-box" style="left:{left:.2%};top:{top:.2%};'
            f'width:{right - left:.2%};height:{bottom - top:.2%}" aria-hidden="true"></span>'
        )
    st.html(
        f'<figure class="photo-preview"><div class="photo-frame" style="--photo-ratio:{ratio:.5f}">'
        f'<img src="data:image/jpeg;base64,{photo}" alt="Fotografia do elemento selecionado">{overlay}'
        f'{photo_credit(props)}</div></figure>'
    )


@st.cache_data(show_spinner=False, max_entries=16, ttl=3600)
def catalog_scene(image_id, detections, _token):
    from photo_analysis import anonymize  # a foto completa também passa pelo EgoBlur antes de se mostrar

    return build_catalog_scene(CATALOG_PATH, image_id, detections, _token, anonymize=anonymize)


def catalog_detection_boxes(props):
    for view in (*catalog_photo_views(props), *(props.get("itens") or ())):
        if (key := image_key(view.get("imagem_id"))) and (box := view.get("imagem_bbox")):
            yield key, [round(value) for value in box]


def catalog_scene_detections(photo):
    image_id = image_key(photo["imagem_id"])
    by_box = {
        tuple(round(value) for value in item["bbox"]): item["label"]
        for item in classified_objects(CATALOG_PATH, image_id) if item.get("label")
    }
    collection = load_catalog_document()
    for feature in collection.get("features", []):
        props = feature["properties"]
        for view in catalog_photo_views(props):
            if image_key(view.get("imagem_id")) != image_id or not view.get("imagem_bbox"):
                continue
            key = tuple(round(value) for value in view["imagem_bbox"])
            if props.get("estado") == "rejeitado":
                by_box.pop(key, None)
            else:
                by_box[key] = feature_label(feature)
    for box in collection.get("segmentacoes_removidas", {}).get(image_id, ()):
        by_box.pop(tuple(box), None)
    return tuple((label, box) for box, label in by_box.items())


def contribution_scene(props):
    contribution_id = props.get("contribuicao_id")
    if contribution_id:
        items = CONTRIBUTIONS.request_features(contribution_id) or [
            feature for feature in CONTRIBUTIONS.features()
            if feature["properties"].get("contribuicao_id") == contribution_id
        ]
    else:
        items = [{"properties": props}]
    previews = []
    for feature in items:
        item = feature["properties"]
        if item.get("estado") == "rejeitado":
            continue
        preview = CONTRIBUTIONS.preview(item["id"], item.get("imagem_bbox_relativa"))
        if preview:
            if item.get("segmentacao_modelo") != SAM2_MODEL:
                preview = PhotoPreview(preview.image, box=preview.box)
            previews.append((feature_label(feature), preview))
    return build_contribution_scene(previews)


def render_scene(scene, props):
    image = base64.b64encode(scene.image).decode("ascii")
    ratio = scene.size[0] / scene.size[1]
    overlay = ""
    if scene.overlay:
        mask = base64.b64encode(scene.overlay).decode("ascii")
        overlay = f'<img class="scene-overlay" src="data:image/png;base64,{mask}" alt="">'
    objects = []
    for label, (left, top, right, bottom), masked in scene.objects:
        label = html.escape(label, quote=True)
        kind = "scene-object" if masked else "scene-object scene-object--caixa"   # sem máscara: caixa visível
        objects.append(
            f'<span class="{kind}" tabindex="0" aria-label="{label}" style="left:{left:.2%};top:{top:.2%};'
            f'width:{right - left:.2%};height:{bottom - top:.2%}"><span>{label}</span></span>'
        )
    st.html(
        '<div class="scene-viewer"><div class="scene-viewport">'
        f'<figure class="scene-frame" style="--scene-ratio:{ratio:.5f}">'
        f'<img src="data:image/jpeg;base64,{image}" alt="Fotografia completa">{overlay}{"".join(objects)}'
        f'{photo_credit(props)}</figure></div></div>'
    )


@st.dialog("Fotografia", width="large")
def render_scene_dialog(props, *, crop=None):
    with st.container(key="full_photo"):
        if crop is not None:
            render_photo(PhotoPreview(crop), props)
            return
        try:
            with st.spinner("A preparar a imagem completa…"):
                if props.get("fonte") == SOURCE:
                    scene = contribution_scene(props)
                else:
                    image_id = image_key(props.get("imagem_id"))
                    if not image_id:
                        raise ValueError("A fotografia completa ainda não está disponível.")
                    scene = catalog_scene(image_id, catalog_scene_detections(props), mapillary_token())
            render_scene(scene, props)
        except PrivacyModelsMissing as error:
            st.error(str(error))
        except (ValueError, SegmentationError) as error:
            st.caption(str(error))
        except (OSError, sqlite3.Error, Image.DecompressionBombError):
            LOGGER.exception("Erro ao abrir a fotografia completa")
            st.error("Não foi possível abrir a imagem completa.")


def full_image_button(props, *, loading=False, label="Imagem completa", crop=None):
    if st.button(
        label, icon=":material/open_in_full:", key=f"full-image-{props['id']}",
        width="stretch", disabled=loading,
    ):
        render_scene_dialog(props, crop=crop)


@st.cache_resource
def photo_jobs():
    return PreviewQueue()


@st.fragment(run_every=0.5)
def render_photo_progress(job):
    if job.done():
        st.rerun()
    st.caption("A preparar a fotografia…")


def render_selected_photo(props, preview, *, token="") -> bool:
    if preview is None and not token:
        if image_key(props.get("imagem_id")):
            st.caption("A fotografia falta em data/images. Configura MAPILLARY_TOKEN para a carregar pela API.")
        else:
            st.caption("A fotografia usada na deteção ainda não está disponível.")
        return False
    if preview and (preview.mask or not preview.box):
        render_photo(preview, props)
        if not preview.mask and props.get("imagem_bbox"):
            st.caption("A caixa da deteção não pôde ser alinhada com esta versão da fotografia.")
        return False
    if preview and not sam_available():
        render_photo(preview, props)               # sem SAM 2 neste PC: foto com a caixa da deteção
        return False

    jobs = photo_jobs()
    job = jobs.submit(CATALOG_PATH, props, preview, token=token)
    if job is None:
        if preview:
            render_photo(preview, props)
        st.info("Há várias fotografias em análise. Tenta novamente daqui a pouco.")
    elif not job.done():
        if preview:
            render_photo(preview, props)
        render_photo_progress(job)
        return True
    else:
        result = job.result()
        if result.preview:
            render_photo(result.preview, props)
        if not result.notice:
            return False
        st.caption(result.notice)

    if st.button("Tentar novamente", icon=":material/refresh:", key=f"retry-photo-{props['id']}"):
        jobs.submit(CATALOG_PATH, props, preview, token=token, retry=True)
        st.rerun()
    return False


def photo_reference(props):
    if props.get("fonte") == SOURCE:
        return "contribuicao"
    if image_id := props.get("imagem_id"):
        return f"imagem:{image_key(image_id) or image_id}"
    return f"recorte:{props.get('imagem') or props['id']}"


def uses_catalog_photo(props, reference):
    views = [props]
    views.extend({**props, "sem_fotografia": False, **item} for item in props.get("itens") or ())
    views.extend({"id": props["id"], **view} for view in props.get("vistas_foto") or ())
    views.extend({"id": props["id"], "imagem_id": value} for value in props.get("imagens") or ())
    return any(not view.get("sem_fotografia") and photo_reference(view) == reference for view in views)


def render_catalog_removal(feature, button_area, photo=None):
    reference = photo_reference(photo) if photo is not None else None
    element_id = feature["properties"]["id"]
    key = f"remove-image-{element_id}-{reference}" if photo is not None else f"remove-element-{element_id}"
    label = "Remover imagem" if photo is not None else "Remover elemento"
    if button_area.button(label, icon=":material/delete:", key=key, width="stretch"):
        st.session_state["catalog_removal_pending"] = key
    if st.session_state.get("catalog_removal_pending") != key:
        return
    remove_object = photo is None
    with st.container(border=True):
        if remove_object:
            st.write(f"O elemento «{feature_label(feature)}» e a sua segmentação serão removidos.")
        else:
            st.write("A fotografia e todos os objetos associados serão retirados do mapa e do catálogo.")
        cancel, confirm = st.columns(2)
        if cancel.button("Cancelar", key=f"{key}-cancel"):
            st.session_state.pop("catalog_removal_pending", None)
            st.rerun()
        if not confirm.button("Confirmar remoção", key=f"{key}-confirm", type="primary"):
            return
        try:
            saved = save_correction(
                feature, removed_photo=reference, remove_object=remove_object,
            )
        except (OSError, sqlite3.Error, ValueError):
            LOGGER.exception("Erro ao remover um objeto ou uma fotografia do catálogo")
            saved = False
        if not saved:
            st.error("Não foi possível concluir a remoção.")
            return
    st.session_state.pop("catalog_removal_pending", None)
    st.session_state["catalog-panel"] = None
    if remove_object:
        st.session_state["catalog_removal_notice"] = "Objeto removido do mapa e do catálogo."
    else:
        st.session_state["catalog_removal_notice"] = "Imagem e objetos associados removidos do mapa e do catálogo."
    st.rerun()


def rendered_views(props):
    """Recortes preparados pela pipeline (scripts/imagens.py app): [(rótulo, props, PhotoPreview)] com a
    foto, a camada da segmentação (se houver) e a caixa do objeto. Nas ilhas: a vista da ilha e a de
    cada contentor. Só ficheiros dentro da pasta do catálogo."""
    base = CATALOG_PATH.parent.resolve()

    def inside(relative):
        if not relative:
            return None
        path = (base / relative).resolve()
        return path if path.is_relative_to(base) and path.is_file() else None

    views = [("Vista do elemento" if props.get("itens") else props.get("label") or "Elemento", props)]
    for number, item in enumerate(props.get("itens") or (), 1):
        view = {**props, "sem_fotografia": False, **item, "id": props["id"]}
        views.append((f"Contentor {number} · {item.get('label') or ''}".strip(" ·"), view))
    found = []
    removed = props.get("imagens_removidas", ())
    for label, view in views:
        if view.get("sem_fotografia") or photo_reference(view) in removed:
            continue
        photo = inside(view.get("imagem"))
        if photo is None:
            continue
        mask = inside(view.get("imagem_mascara"))
        box = view.get("imagem_caixa")
        found.append((label, view, PhotoPreview(
            photo.read_bytes(), mask=mask.read_bytes() if mask else None,
            box=tuple(box) if box and len(box) == 4 else None,
        )))
    return found


def render_image(feature: dict, *, review=False) -> dict | None:
    props = feature.get("properties", {})
    removed = props.get("imagens_removidas", ())
    if props.get("fonte") == SOURCE:
        if props.get("sem_fotografia") or photo_reference(props) in removed:
            st.caption("Sem fotografia associada.")
            return None
        preview = load_catalog_preview(CATALOG_PATH, props, prepared_only=True)
        if preview is None:
            preview = CONTRIBUTIONS.preview(props["id"], props.get("imagem_bbox_relativa"))
            if preview and props.get("segmentacao_modelo") != SAM2_MODEL:
                preview = PhotoPreview(preview.image, box=preview.box)
        if preview:
            loading = render_selected_photo(props, preview)
            full_image_button(props, loading=loading, label="Ir para a foto" if review else "Imagem completa")
            return props
        crop = CONTRIBUTIONS.crop(props["id"])
        if crop:
            render_photo(PhotoPreview(crop), props)
            if review:
                full_image_button(props, label="Ir para a foto", crop=crop)
            return props
        st.caption("A análise desta fotografia não identificou elementos para recortar.")
        return None
    rendered = rendered_views(props)
    if rendered:
        index = 0
        if len(rendered) > 1:
            index = st.selectbox(
                "Fotografias", range(len(rendered)), format_func=lambda value: rendered[value][0],
                key=f"rendered-{props['id']}",
            )
        _, view, preview = rendered[index]
        render_photo(preview, view)               # foto + segmentação (ou caixa) já preparadas
        if image_key(view.get("imagem_id")):
            full_image_button(view)
        return view
    token = mapillary_token()
    photos = [
        view for view in catalog_photo_views(props)
        if not view.get("sem_fotografia") and photo_reference(view) not in removed
        and (has_catalog_preview(CATALOG_PATH, view) or (token and image_key(view.get("imagem_id"))))
    ]
    if not photos and (removed or props.get("sem_fotografia")):
        st.caption("Sem fotografia disponível.")
        return None
    photo = photos[0] if photos else props
    if len(photos) > 1:
        index = st.selectbox(
            "Fotografias", range(len(photos)),
            format_func=lambda value: f"Fotografia {value + 1} de {len(photos)}",
            key=f"gallery-{props['id']}",
        )
        photo = photos[index]
    try:
        preview = load_catalog_preview(CATALOG_PATH, photo) if photos else None
    except (OSError, ValueError, Image.DecompressionBombError):
        LOGGER.exception("Não foi possível abrir a fotografia local do catálogo")
        preview = None
    loading = render_selected_photo(photo, preview, token=token if image_key(photo.get("imagem_id")) else "")
    if preview or (token and image_key(photo.get("imagem_id"))):
        full_image_button(photo, loading=loading)
    return photo if preview or image_key(photo.get("imagem_id")) else None


@st.cache_data
def catalog_subtypes():
    with (ROOT / "config" / "taxonomia.yaml").open(encoding="utf-8") as file:
        taxonomy = yaml.safe_load(file)["categorias"]
    return {
        category: {item["id"]: item for item in spec["subtipos"]}
        for category, spec in taxonomy.items()
    }


def render_details(feature: dict, admin=False) -> None:
    props = feature.get("properties", {})
    longitude, latitude = feature["geometry"]["coordinates"][:2]
    category = props.get("categoria")
    status = props.get("estado", "automatico")
    color = CATEGORY_ACCENTS.get(category, "#c0ced8")
    # O HTML da ficha aceita o SVG como fonte da imagem.
    _, pictogram = marker_icon(props, inline_svg=False)
    label = html.escape(str(props.get("label") or props.get("subtipo") or "Elemento urbano"))
    category_name = html.escape(CATEGORY_LABELS.get(category, str(props.get("categoria_label") or category or "Elemento")))
    st.html(
        f'<div class="item-heading" style="--category:{color}">'
        f'<div class="item-heading__icon" aria-hidden="true">{pictogram}</div>'
        f'<div><div class="item-heading__category">{category_name}</div><h2>{label}</h2></div></div>'
    )
    actions = st.container() if admin else None
    photo = render_image(feature)
    if actions is not None:
        with actions:
            if photo is None:
                render_catalog_removal(feature, actions)
            else:
                element_action, image_action = st.columns(2)
                render_catalog_removal(feature, element_action)
                render_catalog_removal(feature, image_action, photo)
    if props.get("confianca") is not None:
        confidence = f"{min(1.0, max(0.0, float(props['confianca']))):.0%}"
        st.html(
            '<div class="confidence"><div class="confidence__label">'
            f'<span>Confiança da deteção</span><strong>{confidence}</strong></div>'
            f'<div class="confidence__track" aria-hidden="true"><div class="confidence__value" '
            f'style="--confidence:{confidence}"></div></div></div>'
        )
    fields = []
    if admin:
        fields.append(("Estado", {"automatico": "Por rever", "confirmado": "Confirmado", "rejeitado": "Rejeitado"}.get(status, status)))
    if category != PHOTO_CATEGORY:
        fields.append(("Subtipo", subtype_label(category, props.get("subtipo") or "")))
    if props.get("metodo_posicao") == "local da foto":
        fields.append(("Posição", "local da foto (sem direção da câmara no EXIF)"))
    elif props.get("distancia_m") is not None:
        fields.append(("Posição", f"a {props['distancia_m']:.0f} m da câmara"))
    metadata = "".join(f"<div><dt>{name}</dt><dd>{html.escape(str(value))}</dd></div>" for name, value in fields)
    st.html(
        f'<dl class="item-meta">{metadata}<div class="item-meta__full"><dt>Coordenadas</dt>'
        f'<dd>{latitude:.6f}, {longitude:.6f}</dd></div></dl>'
    )


def render_correction(feature: dict, features: list[dict]) -> None:
    props = feature["properties"]
    item_id = props["id"]
    category, subtype = props.get("categoria", ""), props.get("subtipo", "")
    longitude, latitude = feature["geometry"]["coordinates"][:2]
    taxonomy = catalog_subtypes()
    categories = [PHOTO_CATEGORY] if category == PHOTO_CATEGORY else sorted(set(taxonomy) | {category})
    with st.expander("Editar elemento", expanded=True):
        category_choice = st.selectbox(
            "Categoria", categories, index=categories.index(category),
            format_func=category_label, key=f"edit-category-{item_id}",
        )
        subtypes = set(taxonomy.get(category_choice, {})) | {
            item["properties"].get("subtipo", "")
            for item in features if item["properties"].get("categoria") == category_choice
        }
        subtype_options = sorted(subtypes)
        subtype_choice = st.selectbox(
            "Subtipo", subtype_options, index=subtype_options.index(subtype) if subtype in subtype_options else 0,
            format_func=lambda value: subtype_label(category_choice, value),
            key=f"edit-subtype-{item_id}-{category_choice}",
        )
        label = st.text_input("Designação", value=props.get("label", ""), key=f"edit-label-{item_id}")
        lat_col, lon_col = st.columns(2)
        with lat_col:
            new_latitude = st.number_input("Latitude", -90.0, 90.0, float(latitude), format="%.7f", key=f"edit-lat-{item_id}")
        with lon_col:
            new_longitude = st.number_input("Longitude", -180.0, 180.0, float(longitude), format="%.7f", key=f"edit-lon-{item_id}")
        statuses = {"automatico": "Por rever", "confirmado": "Confirmado", "rejeitado": "Rejeitado"}
        current_status = props.get("estado")
        status_choice = st.selectbox(
            "Estado", list(statuses), index=list(statuses).index(current_status) if current_status in statuses else 0,
            format_func=statuses.__getitem__, key=f"edit-status-{item_id}",
        )
        if not st.button("Guardar correção", icon=":material/save:", type="primary", key=f"edit-save-{item_id}"):
            return

    updated = {
        "categoria": category_choice,
        "categoria_label": category_label(category_choice),
        "subtipo": subtype_choice,
        "label": label.strip(),
        "estado": status_choice,
    }
    if (category_choice, subtype_choice) != (category, subtype):
        updated["codigo"] = taxonomy.get(category_choice, {}).get(subtype_choice, {}).get("codigo")
        updated["valor"] = None
    if props.get("fonte") == SOURCE and (new_latitude, new_longitude) != (latitude, longitude):
        updated["origem_coordenadas"] = "manual"
    try:
        saved = save_correction(feature, changes=updated, coordinates=[new_longitude, new_latitude])
    except (OSError, sqlite3.Error, ValueError):
        LOGGER.exception("Erro ao guardar uma correção no catálogo")
        st.error("Não foi possível guardar a correção.")
        return
    if not saved:
        st.error("O elemento já não está disponível no catálogo.")
        return
    st.session_state["catalog_saved"] = True
    st.rerun()


def subtype_label(category, subtype):
    spec = catalog_subtypes().get(category, {}).get(subtype, {})
    if subtype == "sem_deteccoes":
        return "Sem elementos detetados"
    return spec.get("label", subtype.replace("_", " ").capitalize()).replace(" {n} km/h", "")


def category_label(category):
    return CATEGORY_LABELS.get(category, category.replace("_", " ").capitalize())


def glyph(category):
    return f":material/{CATEGORY_GLYPHS.get(category, 'location_on')}:"


def render_brand(admin):
    with st.container(key="hud_brand"):
        with st.container(key="site_brand", horizontal=True, vertical_alignment="center", width="content", gap="small"):
            st.image(str(LOGO_PATH), width="content")
            st.html(
                '<span class="site-brand__name" role="img" aria-label="IAMU">'
                '<span class="site-brand__letter site-brand__letter--i" aria-hidden="true"></span>'
                '<span class="site-brand__letter site-brand__letter--a" aria-hidden="true"></span>'
                '<span class="site-brand__letter site-brand__letter--m" aria-hidden="true"></span>'
                '<span class="site-brand__letter site-brand__letter--u" aria-hidden="true"></span></span>',

                width="content",
            )
            if admin:
                st.html('<span class="site-brand__mode">Edição</span>', width="content")


def render_intro():
    if st.session_state.get("catalog-intro-shown"):
        return
    st.session_state["catalog-intro-shown"] = True
    with st.container(key="catalog_intro"):
        st.html(
            '<div class="catalog-intro__veil" aria-hidden="true"></div>'
            '<div class="catalog-intro__pulse" aria-hidden="true"></div>'
            '<div class="catalog-intro__pulse" aria-hidden="true"></div>',
        )
        st.image(str(LOGO_PATH), width="content")


def render_navigation(admin):
    if st.button("Adicionar fotos", key="map-upload", icon=":material/add_a_photo:", help="Adicionar fotografias"):
        st.session_state["upload_open"] = True
    if admin:
        st.button(
            "Pedidos", key="map-requests", icon=":material/inbox:", help="Rever pedidos de publicação",
            type="primary" if st.session_state.get("catalog-panel") == "Administração" else "secondary",
            on_click=toggle_panel, args=("Administração",),
        )


def clear_filters(clear_search=False):
    st.session_state["catalog-categories"] = []
    for key in list(st.session_state):
        if key == "catalog-pending" or key.startswith(SUBTYPE_FILTER_PREFIX):
            st.session_state[key] = False
    if clear_search:
        for key in ("map-search", "search-pin"):
            st.session_state.pop(key, None)


def submit_search():
    query = st.session_state.get("map-search")
    st.session_state["map-search"] = None
    if query and (place := find_place(query)):
        point, _ = place
        if st.session_state.get("catalog-panel") == "Ficha":
            st.session_state["catalog-panel"] = None
        request_view(point=point, zoom=14)
        st.session_state["search-pin"] = {
            "point": point,
            "revision": st.session_state["map-view-request"]["revision"],
        }


def render_search():
    with st.container(key="hud_search", width="content"):
        st.selectbox(
            "Pesquisar localidades em Portugal", place_suggestions(), index=None,
            label_visibility="collapsed", key="map-search", placeholder="Localidade",
            icon=":material/search:", filter_mode="contains", on_change=submit_search,
        )


def signal_type(props):
    code = str(props.get("codigo") or "").removeprefix("PT:")
    value = str(props.get("valor") or "") if props.get("subtipo") == "limite_velocidade" else ""
    return code, value


def filter_catalog(available, categories, subtypes, pending_only, signal_types=()):
    selected_signs = {}
    for family, sign in signal_types:
        selected_signs.setdefault(family, set()).add(sign)
    matches = []
    for feature in available:
        props = feature["properties"]
        category = props.get("categoria")
        if categories and category not in categories:
            continue
        family = props.get("subtipo")
        if subtypes and (category, family) not in subtypes:
            continue
        if category == "sinalizacao" and family in selected_signs and signal_type(props) not in selected_signs[family]:
            continue
        if pending_only and props.get("estado", "automatico") != "automatico":
            continue
        matches.append(feature)
    return matches


def render_filter_tile(option, label, props):
    _, icon = marker_icon(props, inline_svg=False)
    with st.container(key=f"subtype_tile_{option}"):
        st.html(f'<span class="subtype-tile-icon" aria-hidden="true">{icon}</span>')
        return st.checkbox(label, key=f"{SUBTYPE_FILTER_PREFIX}{option}", width="stretch", wrap=True)


def signal_family_options(available):
    families = {}
    for feature in available:
        props = feature["properties"]
        family = props.get("subtipo")
        if props.get("categoria") == "sinalizacao" and family:
            key = signal_type(props)
            if key[0]:
                label = feature_label(feature).removeprefix("Sinal ")
                families.setdefault(family, {}).setdefault(key, (label, props))
    return families


def render_subtype_tiles(subtypes):
    st.markdown("Subtipos")
    selected = []
    with st.container(key="subtype_grid"):
        for category, subtype in subtypes:
            if render_filter_tile(
                f"{category}-{subtype}", subtype_label(category, subtype),
                {"categoria": category, "subtipo": subtype},
            ):
                selected.append((category, subtype))
    return selected


def render_signal_menu(available, categories, subtypes):
    families = [family for category, family in subtypes if category == "sinalizacao"]
    if not families or (categories and "sinalizacao" not in categories):
        return []
    options = signal_family_options(available)
    families = [family for family in families if len(options.get(family, ())) > 1]
    if not families:
        return []
    selected = []
    with st.container(key="hud_signals"), st.container(key="hud_signal_contents"):
        for family in families:
            with st.container(key=f"signal_family_{family}"):
                st.markdown(f"**{subtype_label('sinalizacao', family)}**")
                with st.container(key=f"subtype_grid_{family}"):
                    for (code, value), (label, props) in sorted(options[family].items()):
                        if render_filter_tile(f"sinal-{family}-{code}-{value}", f"{code} · {label}", props):
                            selected.append((family, (code, value)))
    return selected


def render_filters(available, admin, signal_menu_slot):
    categories = list(dict.fromkeys(
        feature["properties"]["categoria"] for feature in available if feature["properties"].get("categoria")
    ))
    categories.sort(key=category_label)
    subtypes = sorted({
        (props["categoria"], props["subtipo"])
        for feature in available
        if (props := feature["properties"]).get("categoria") and props.get("subtipo")
    })
    with st.container(key="hud_filters"):
        selected_categories = st.pills(
            "Categorias no mapa", categories, selection_mode="multi",
            format_func=lambda category: f"{glyph(category)} {category_label(category)}",
            key="catalog-categories",
        )
        selected_subtypes = render_subtype_tiles(subtypes)
        pending_only = admin and st.checkbox("Mostrar apenas por rever", key="catalog-pending")
        st.button("Limpar filtros", icon=":material/filter_alt_off:", on_click=clear_filters)
    with signal_menu_slot:
        selected_signal_types = render_signal_menu(available, selected_categories, selected_subtypes)
    return filter_catalog(available, selected_categories, selected_subtypes, pending_only, selected_signal_types)


def render_downloads(features, admin):
    filename = f"catalogo_{re.sub(r'[^a-zA-Z0-9_-]+', '_', MUNICIPALITY.lower())}"
    export = features if admin else [
        {**feature, "properties": {key: value for key, value in feature["properties"].items() if key != "estado"}}
        for feature in features
    ]
    st.download_button(
        "Descarregar GeoJSON",
        data=json.dumps({"type": "FeatureCollection", "features": export}, ensure_ascii=False).encode("utf-8"),
        file_name=f"{filename}.geojson", mime="application/geo+json", width="stretch", on_click="ignore",
    )
    st.download_button(
        "Descarregar CSV", data=csv_bytes(features, include_status=admin), file_name=f"{filename}.csv", mime="text/csv",
        width="stretch", on_click="ignore",
    )


def request_view(*, point=None, bounds=None, zoom=18):
    revision = st.session_state.get("map-view-request", {}).get("revision", 0)
    st.session_state["map-view-request"] = {"revision": revision + 1, "point": point, "bounds": bounds, "zoom": zoom}


def select_element(feature_id, point):
    st.session_state.update({"map-point": point, "selected-id": feature_id, "catalog-panel": "Ficha"})


def toggle_panel(name):
    st.session_state["catalog-panel"] = None if st.session_state.get("catalog-panel") == name else name


def render_view_controls(points, filtered, admin):
    with st.container(key="hud_map_actions"):
        st.button(
            "Enquadrar", key="map-fit", icon=":material/my_location:", help="Enquadrar elementos", disabled=not points,
            on_click=request_view, kwargs={"bounds": point_bounds(points)},
        )
        export_menu = st.popover(
            "Exportar", key="map-export", icon=":material/download:", help="Exportar elementos", disabled=not points,
            width=41, wrap=False, on_change="rerun",
        )
        if export_menu.open:
            with export_menu, st.container(key="hud_export_menu"):
                render_downloads(filtered, admin)
        label = "Voltar ao mapa" if admin else "Administração"
        if st.button(label, key="map-admin", icon=":material/settings:", help=label):
            st.session_state["catalog-admin-mode"] = not admin
            st.session_state["catalog-panel"] = "Administração" if not admin else None
            st.rerun()


def render_map_legend(features):
    with st.container(key="hud_legend"):
        legend = st.expander("Legenda", icon=":material/legend_toggle:", key="map-legend", on_change="rerun")
        if not legend.open:
            return
        if not features:
            legend.caption("Sem elementos para os filtros atuais.")
            return

        groups = {}
        for feature in features:
            props = feature["properties"]
            category = props.get("categoria") or "outros"
            groups.setdefault(category, set()).add(props.get("subtipo") or "")

        sections = []
        for category in sorted(groups, key=category_label):
            heading = category_label(category)
            subtypes = sorted(groups[category], key=lambda subtype: subtype_label(category, subtype))
            entries = []
            for subtype in subtypes:
                label = subtype_label(category, subtype) if subtype else heading
                _, icon = marker_icon({"categoria": category, "subtipo": subtype}, inline_svg=False)
                entries.append(
                    f'<li><span aria-hidden="true">{icon}</span><span>{html.escape(label)}</span></li>'
                )
            sections.append(
                f'<section class="map-legend__group"><h3>{html.escape(heading)}</h3>'
                f'<ul>{"".join(entries)}</ul></section>'
            )
        legend.html(f'<div class="map-legend__content">{"".join(sections)}</div>')


def review_time(value):
    local = datetime.fromisoformat(value).astimezone(ZoneInfo("Europe/Lisbon"))
    return local.strftime("%d/%m/%Y · %H:%M:%S")


def render_review_log(request_id=None):
    events = CONTRIBUTIONS.review_log(request_id)
    if not events:
        st.caption("Ainda não há registos.")
        return
    rows = []
    for event in events:
        row = {"Data": review_time(event["at"]), "Ação": REVIEW_ACTIONS[event["action"]]}
        if request_id is None:
            row.update({"Pedido": event["name"], "Referência": event["request_id"], "Origem": REVIEW_SOURCES[event["source"]]})
        row["Nota"] = event["note"]
        rows.append(row)
    st.dataframe(rows, hide_index=True, width="stretch", height=min(300, 38 + len(rows) * 35))


def render_review_request(request, points):
    request_id = request["id"]
    count = request["detection_count"]
    detections = f"{count} {'deteção' if count == 1 else 'deteções'}"
    st.caption(f"{REVIEW_SOURCES[request['source']]} · {detections} · {review_time(request['submitted_at'])}")
    st.caption(f"{request['latitude']:.6f}, {request['longitude']:.6f}")
    items = CONTRIBUTIONS.request_features(request_id)
    if items:
        index = 0
        if len(items) > 1:
            index = st.selectbox(
                "Deteção a rever", range(len(items)), key=f"review-detection-{request_id}",
                format_func=lambda index: feature_label(items[index]),
            )
        item = items[index]
        published = request["status"] == "aprovado"
        st.text(feature_label(item))
        render_image(item, review=not published)
        if published:
            longitude, latitude = item["geometry"]["coordinates"][:2]
            point = (latitude, longitude)
            item_id = item["properties"]["id"]
            visible = any(candidate["properties"]["id"] == item_id for candidate in points.get(point, ()))
            if st.button(
                "Ir para a foto", icon=":material/location_on:", width="stretch",
                key=f"request-photo-{request_id}", on_click=None if visible else clear_filters,
            ):
                select_element(item_id, point)
                request_view(point=point)
                st.rerun()
    else:
        st.caption("Este pedido já não tem fotografias ou elementos disponíveis.")
    if request["status"] == "pendente":
        note = st.text_input("Nota da decisão (opcional)", max_chars=300, key=f"review-note-{request_id}")
        approve, reject = st.columns(2)
        with approve:
            approved = st.button("Aprovar e publicar", icon=":material/check:", type="primary", width="stretch")
        with reject:
            rejected = st.button("Rejeitar", icon=":material/close:", width="stretch")
        if approved or rejected:
            updated = CONTRIBUTIONS.review(request_id, "aprovado" if approved else "rejeitado", note)
            if updated:
                if approved:
                    request_view(point=(request["latitude"], request["longitude"]))
                message = "Pedido aprovado e publicado no mapa." if approved else "Pedido rejeitado."
                st.session_state["review_notice"] = ("success", message)
            else:
                st.session_state["review_notice"] = ("info", "Este pedido já foi avaliado.")
            st.rerun()
    with st.expander("Histórico do pedido"):
        render_review_log(request_id)


def render_admin(points):
    with st.container(key="admin_workspace"):
        if notice := st.session_state.pop("review_notice", None):
            getattr(st, notice[0])(notice[1])
        queue, log = st.tabs(["Pedidos", "Registo"])
        with queue:
            st.caption(f"{CONTRIBUTIONS.pending_count()} por rever")
            statuses = {"todos": "Todos", **REVIEW_STATUS}
            status = st.radio(
                "Estado dos pedidos", list(statuses), format_func=statuses.__getitem__,
                horizontal=True, label_visibility="collapsed", key="review-request-filter",
            )
            requests = CONTRIBUTIONS.requests(None if status == "todos" else status)
            if requests:
                request = st.selectbox(
                    "Pedido", requests, key=f"review-request-{status}",
                    format_func=lambda request: f"{request['name']} · {REVIEW_STATUS[request['status']]} · {request['id']}",
                )
                render_review_request(request, points)
            else:
                st.caption("Sem pedidos neste estado.")
        with log:
            render_review_log()


def render_panel(panel, filtered, points, admin, features):
    with st.container(key="inspector"):
        title, close = st.columns([6, 1], vertical_alignment="center")
        with title:
            st.subheader(panel)
        with close, st.container(key="hud_close"):
            st.button("Fechar painel", icon=":material/close:", help="Fechar painel", on_click=toggle_panel, args=(panel,))

        if panel == "Administração" and admin:
            try:
                render_admin(points)
            except (OSError, sqlite3.Error, ValueError):
                LOGGER.exception("Erro ao consultar ou rever pedidos de publicação")
                st.error("Não foi possível abrir ou atualizar os pedidos.")
            return
        if not filtered:
            st.write("Nenhum elemento corresponde aos filtros.")
            st.button(
                "Limpar filtros e pesquisa", icon=":material/filter_alt_off:",
                on_click=clear_filters, kwargs={"clear_search": True},
            )
            return

        at_point = points.get(st.session_state.get("map-point"))
        selected_id = st.session_state.get("selected-id")
        alternatives = at_point or filtered
        selected = next((item for item in alternatives if item["properties"]["id"] == selected_id), None)
        if selected is None:
            selected = alternatives[0] if at_point or admin else None
        if selected is None:
            st.write("Seleciona um marcador no mapa.")
            return
        if at_point and len(at_point) > 1:
            labels = [feature_label(item) for item in at_point]
            repetitions = Counter(labels)
            index = st.selectbox(
                "Elementos neste ponto", list(range(len(at_point))),
                index=at_point.index(selected),
                format_func=lambda index: f"{labels[index]} ({index + 1})" if repetitions[labels[index]] > 1 else labels[index],
                key=f"selected-feature-{st.session_state.get('map-point')}-{selected_id}",
            )
            selected = at_point[index]
        longitude, latitude = selected["geometry"]["coordinates"][:2]
        render_details(selected, admin)
        st.button(
            "Localizar no mapa", icon=":material/location_searching:", width="stretch",
            on_click=request_view, kwargs={"point": (latitude, longitude)},
        )
        if admin:
            render_correction(selected, features)


MAP_MENU_KEYS = ("map-legend", "map-export", "upload_open")


def close_map_menus():
    if any(st.session_state.get(key) for key in MAP_MENU_KEYS):
        st.session_state["map-menus-close-requested"] = True
        st.rerun()


@st.fragment
def render_explorer(filtered, points, layer, map_bounds, admin, features):
    search_pin = st.session_state.get("search-pin")
    with st.container(key="map_surface"):
        event = st_folium(
            build_map(map_bounds), key="catalog-map", height=560, use_container_width=True,
            feature_group_to_add=build_marker_layer(*layer, st.session_state.get("map-view-request"), search_pin),
            returned_objects=[],
        )
    if search_pin and event.get("catalog_search_pin_dismissed") == search_pin["revision"]:
        st.session_state.pop("search-pin", None)
    dismissed = event.get("catalog_dismiss")
    if dismissed is not None and dismissed != st.session_state.get("map-last-dismiss"):
        st.session_state["map-last-dismiss"] = dismissed
        for key in ("catalog-panel", "selected-id", "map-point", "catalog_removal_pending"):
            st.session_state.pop(key, None)
        close_map_menus()

    selected = event.get("catalog_selection")
    if isinstance(selected, dict) and selected != st.session_state.get("map-last-click"):
        point = (selected.get("lat"), selected.get("lng"))
        element_id = selected.get("id")
        st.session_state["map-last-click"] = selected
        if all(isinstance(value, (int, float)) for value in point) and any(
            item["properties"]["id"] == element_id for item in points.get(point, ())
        ):
            select_element(element_id, point)
            close_map_menus()

    panel = st.session_state.get("catalog-panel")
    if panel in ("Ficha", "Administração"):
        render_panel(panel, filtered, points, admin, features)


def main() -> None:
    if st.session_state.pop("map-menus-close-requested", False):
        for key in MAP_MENU_KEYS:
            st.session_state[key] = False
    st.html(APP_DIR / "ui.css")
    st.html(APP_DIR / "marker_icons.css")
    st.html(f"<script>{MENU_MOTION_JS}</script>", unsafe_allow_javascript=True)
    admin = st.session_state.get("catalog-admin-mode", False)
    render_brand(admin)
    render_intro()
    try:
        features = load_catalog()
    except (OSError, json.JSONDecodeError, ValueError) as error:
        with st.container(key="hud_notices"):
            st.error(f"Não foi possível carregar o catálogo: {error}")
        return
    map_bounds = point_bounds(group_points(features))
    messages = st.session_state.pop("upload_messages", [])
    try:
        features.extend(CONTRIBUTIONS.features())
    except (OSError, sqlite3.Error, ValueError):
        LOGGER.exception("Erro ao abrir a base de dados de contribuições")
        messages.append(("error", "Não foi possível carregar as contribuições fotográficas."))
    if st.session_state.pop("catalog_saved", False):
        messages.append(("success", "Correção guardada no catálogo."))
    if notice := st.session_state.pop("catalog_removal_notice", None):
        messages.append(("success", notice))
    if messages:
        with st.container(key="hud_notices"):
            for level, message in messages:
                getattr(st, level)(message)
    available = [feature for feature in features if admin or feature["properties"].get("estado") != "rejeitado"]
    signal_menu_slot = st.container()
    with st.container(key="hud_top", horizontal_alignment="center", gap="small"):
        with st.container(key="hud_controls", horizontal=True, horizontal_alignment="center", gap="small"):
            render_search()
            st.html(
                '<details class="catalog-filter-toggle"><summary>'
                '<span class="catalog-filter-icon" aria-hidden="true"></span>Filtros</summary></details>',
                width="content",
            )
            render_navigation(admin)
        filtered = render_filters(available, admin, signal_menu_slot)
    points = group_points(filtered)
    render_map_legend(filtered)
    render_view_controls(points, filtered, admin)
    if st.session_state.get("upload_open"):
        render_upload()
    render_explorer(
        filtered, points, marker_data(points), map_bounds, admin, features,
    )


if __name__ == "__main__":
    main()
