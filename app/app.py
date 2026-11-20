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
    PrivacyModelsMissing, decode_photo, photo_coordinates,
)
from mapillary_photos import PHOTO_LICENSE, MapillaryPhotoError, fetch_photo, image_key, mapillary_token, photo_link
from photo_preview import (
    SAM2_MODEL, PhotoPreview, PreviewQueue, SegmentationError, catalog_photo_views, has_catalog_preview,
    load_catalog_preview,
)
from photo_scene import build_catalog_scene, build_contribution_scene, classified_objects
from place_search import find_place, place_suggestions


APP_DIR = Path(__file__).resolve().parent          # ficheiros da app (ui.css, marker_icons.css)
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
    page_title=f"Ctrl+F {MUNICIPALITY} | Catálogo urbano",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_data(show_spinner=False)
def _read_catalog(path, modified_ns, size) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        collection = json.load(file)
    if collection.get("type") != "FeatureCollection":
        raise ValueError(f"Formato GeoJSON inválido: {path}")
    features = collection.get("features", [])
    for feature in features:
        props = feature["properties"]
        if props.get("fonte") == "Mapillary":
            props["fonte"] = "Deteções do catálogo"
    return features


def load_catalog() -> list[dict]:
    stat = CATALOG_PATH.stat()
    return _read_catalog(CATALOG_PATH, stat.st_mtime_ns, stat.st_size)


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


def save_correction(feature: dict) -> bool:
    props = feature["properties"]
    if props.get("fonte") == SOURCE:
        return CONTRIBUTIONS.update(feature)

    with catalog_write_lock():
        features = load_catalog()
        index = next((i for i, item in enumerate(features) if item["properties"]["id"] == props["id"]), None)
        if index is None:
            return False
        features[index] = feature
        collection = {"type": "FeatureCollection", "features": features}
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


def submit_photo(digest, data, coordinates, location_source, *, name, source, credit=None, panoramic=False):
    # Todas as fotos (também as do Mapillary) passam pelo EgoBlur antes de se guardar seja o que for.
    analyzer = get_analyzer()
    with decode_photo(data) as image:
        detections = analyzer.analyze(image, panoramic=panoramic)
        preview = None if detections else analyzer.preview_without_detections(image)
    return CONTRIBUTIONS.submit(
        digest, *coordinates, location_source, detections, name=name, source=source, preview=preview, credit=credit,
    )


def import_mapillary_photo(key, token):
    digest = f"mapillary:{key}"
    if CONTRIBUTIONS.has_photo(digest):
        return "info", "Esta fotografia do Mapillary já foi recebida."
    photo = fetch_photo(key, token)
    if photo.coordinates is None:
        raise MapillaryPhotoError("A API não devolveu coordenadas para esta fotografia.")
    credit = {
        "imagem_id": key, "imagem_url": photo_link(key), "autor": photo.author, "licenca": PHOTO_LICENSE,
    }
    request_id = submit_photo(
        digest, photo.data, photo.coordinates, "gps_mapillary",
        name=f"Mapillary · {key}", source="mapillary", credit=credit, panoramic=photo.panoramic,
    )
    if request_id is None:
        return "info", "Esta fotografia do Mapillary já foi recebida."
    return "success", "Fotografia recebida."


def render_mapillary_import(revision):
    token = mapillary_token()
    raw = st.text_input("Link ou ID da fotografia do Mapillary", key=f"mapillary-image-{revision}")
    key = image_key(raw)
    if not token:
        st.caption("Configura MAPILLARY_TOKEN no ficheiro .env do servidor para importar fotografias.")
    if raw and not key:
        st.caption("Usa um ID numérico ou um link de fotografia com pKey.")
    if not st.button(
        "Enviar fotografia do Mapillary", icon=":material/add_a_photo:", type="primary",
        width="stretch", disabled=not (token and key),
    ):
        return
    with st.spinner("A obter a fotografia e a executar OWLv2, SigLIP e SAM 2…"):
        try:
            message = import_mapillary_photo(key, token)
        except (PrivacyModelsMissing, SegmentationError, ValueError) as error:
            message = ("error", str(error))
        except Exception:
            LOGGER.exception("Erro ao analisar uma fotografia do Mapillary")
            message = ("error", "Não foi possível enviar a fotografia.")
    finish_submission([message], revision)


@st.dialog("Adicionar fotografias", width="large", on_dismiss=close_upload)
def render_upload():
    st.caption("O GPS da foto situa os elementos no mapa. Os recortes são anonimizados antes de seguir para revisão.")
    for message in st.session_state.get("upload_errors", []):
        st.error(message)
    revision = st.session_state.get("upload_revision", 0)
    source = st.radio("Origem da fotografia", ["Ficheiros", "Mapillary"], horizontal=True, key="photo-source")
    if source == "Mapillary":
        render_mapillary_import(revision)
        return
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
    progress = st.progress(0, text="A preparar a análise…")
    for index, (file, coordinates, location_source) in enumerate(ready):
        progress.progress(index / len(ready), text=f"A analisar fotografia {index + 1} de {len(ready)}")
        data = file.getvalue()
        digest = sha256(data).hexdigest()
        try:
            if CONTRIBUTIONS.has_photo(digest):
                results.append(("info", f"{file.name} · Esta fotografia já foi enviada."))
                continue
            request_id = submit_photo(digest, data, coordinates, location_source, name=file.name, source="ficheiro")
            if request_id:
                results.append(("success", f"{file.name} · Fotografia recebida."))
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


def catalog_scene_detections(photo):
    image_id = str(photo["imagem_id"])
    by_box = {
        tuple(round(value) for value in item["bbox"]): item["label"]
        for item in classified_objects(CATALOG_PATH, image_id) if item.get("label")
    }
    for feature in load_catalog():
        props = feature["properties"]
        for view in catalog_photo_views(props):
            if str(view.get("imagem_id")) != image_id or not view.get("imagem_bbox"):
                continue
            key = tuple(round(value) for value in view["imagem_bbox"])
            if props.get("estado") == "rejeitado":
                by_box.pop(key, None)
            else:
                by_box[key] = feature_label(feature)
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
    for label, (left, top, right, bottom) in scene.objects:
        label = html.escape(label, quote=True)
        objects.append(
            f'<span class="scene-object" tabindex="0" aria-label="{label}" style="left:{left:.2%};top:{top:.2%};'
            f'width:{right - left:.2%};height:{bottom - top:.2%}"><span>{label}</span></span>'
        )
    st.html(
        '<div class="scene-viewer"><div class="scene-viewport">'
        f'<figure class="scene-frame" style="--scene-ratio:{ratio:.5f}">'
        f'<img src="data:image/jpeg;base64,{image}" alt="Fotografia completa">{overlay}{"".join(objects)}'
        f'{photo_credit(props)}</figure></div></div>'
    )


@st.dialog("Imagem completa", width="large")
def render_scene_dialog(props):
    with st.container(key="full_photo"):
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


def full_image_button(props, *, loading=False):
    if st.button(
        "Imagem completa", icon=":material/open_in_full:", key=f"full-image-{props['id']}",
        width="stretch", disabled=loading,
    ):
        render_scene_dialog(props)


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
        views.append((f"Contentor {number} · {item.get('label') or ''}".strip(" ·"), {**props, **item, "id": props["id"]}))
    found = []
    for label, view in views:
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


def render_image(feature: dict) -> None:
    props = feature.get("properties", {})
    if props.get("fonte") == SOURCE:
        preview = load_catalog_preview(CATALOG_PATH, props, prepared_only=True)
        if preview is None:
            preview = CONTRIBUTIONS.preview(props["id"], props.get("imagem_bbox_relativa"))
            if preview and props.get("segmentacao_modelo") != SAM2_MODEL:
                preview = PhotoPreview(preview.image, box=preview.box)
        if preview:
            loading = render_selected_photo(props, preview)
            full_image_button(props, loading=loading)
            return
        crop = CONTRIBUTIONS.crop(props["id"])
        if crop:
            render_photo(PhotoPreview(crop), props)
        else:
            st.caption("A análise desta fotografia não identificou elementos para recortar.")
        return
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
        return
    token = mapillary_token()
    photos = [
        view for view in catalog_photo_views(props)
        if has_catalog_preview(CATALOG_PATH, view) or (token and image_key(view.get("imagem_id")))
    ]
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
    render_image(feature)
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
        **props,
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
    correction = {
        **feature,
        "properties": updated,
        "geometry": {**feature["geometry"], "coordinates": [new_longitude, new_latitude]},
    }
    try:
        saved = save_correction(correction)
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
        mode = '<span class="site-brand__mode">Edição</span>' if admin else ""
        st.html(
            '<div class="site-brand"><span class="site-brand__symbol" aria-hidden="true">⌖</span>'
            '<div><span class="site-brand__name">Ctrl+F</span>'
            f'<span class="site-brand__location">{html.escape(MUNICIPALITY)}</span></div>{mode}</div>'
        )


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
        for key in ("map-search", "submitted-map-search", "search-marker"):
            st.session_state.pop(key, None)


def toggle_filters():
    st.session_state["catalog-filters-open"] = not st.session_state.get("catalog-filters-open", False)


def render_search():
    with st.container(key="hud_search", width="content"):
        query = st.selectbox(
            "Pesquisar localidades em Portugal", place_suggestions(), index=None,
            label_visibility="collapsed", key="map-search", placeholder="Localidade",
            icon=":material/search:", filter_mode="contains",
        )
        if query != st.session_state.get("submitted-map-search"):
            st.session_state["submitted-map-search"] = query
            st.session_state.pop("search-marker", None)
            if query and (place := find_place(query)):
                point, _ = place
                st.session_state["search-marker"] = point
                if st.session_state.get("catalog-panel") == "Ficha":
                    st.session_state["catalog-panel"] = None
                request_view(point=point, zoom=14)


def filter_catalog(available, categories, subtypes, pending_only):
    matches = []
    for feature in available:
        props = feature["properties"]
        category = props.get("categoria")
        if categories and category not in categories:
            continue
        if subtypes and (category, props.get("subtipo")) not in subtypes:
            continue
        if pending_only and props.get("estado", "automatico") != "automatico":
            continue
        matches.append(feature)
    return matches


def render_subtype_tiles(subtypes):
    st.markdown("Subtipos")
    selected = []
    with st.container(key="subtype_grid"):
        for category, subtype in subtypes:
            option = f"{category}-{subtype}"
            _, icon = marker_icon({"categoria": category, "subtipo": subtype}, inline_svg=False)
            with st.container(key=f"subtype_tile_{option}"):
                st.html(f'<span class="subtype-tile-icon" aria-hidden="true">{icon}</span>')
                if st.checkbox(
                    subtype_label(category, subtype), key=f"{SUBTYPE_FILTER_PREFIX}{option}", width="stretch", wrap=True,
                ):
                    selected.append((category, subtype))
    return selected


def render_filters(available, admin, filters_open):
    categories = list(dict.fromkeys(
        feature["properties"]["categoria"] for feature in available if feature["properties"].get("categoria")
    ))
    categories.sort(key=category_label)
    subtypes = sorted({
        (props["categoria"], props["subtipo"])
        for feature in available
        if (props := feature["properties"]).get("categoria") and props.get("subtipo")
    })
    key = "hud_filters" if filters_open else "hud_filters_hidden"
    with st.container(key=key):
        selected_categories = st.pills(
            "Categorias no mapa", categories, selection_mode="multi",
            format_func=lambda category: f"{glyph(category)} {category_label(category)}",
            key="catalog-categories",
        )
        selected_subtypes = render_subtype_tiles(subtypes)
        pending_only = admin and st.checkbox("Mostrar apenas por rever", key="catalog-pending")
        st.button("Limpar filtros", icon=":material/filter_alt_off:", on_click=clear_filters)
    return filter_catalog(available, selected_categories, selected_subtypes, pending_only)


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
                st.html(
                    """<script>
                    if (!window.catalogExportDismissInstalled) {
                        window.catalogExportDismissInstalled = true;
                        window.addEventListener("blur", () => requestAnimationFrame(() => {
                            if (document.activeElement?.matches(".st-key-map_surface iframe")) {
                                document.querySelector(".st-key-map-export button[aria-expanded='true']")?.click();
                            }
                        }));
                    }
                    </script>""",
                    unsafe_allow_javascript=True,
                )
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


def render_review_request(request):
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
        st.text(feature_label(items[index]))
        render_image(items[index])
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
                message = "Pedido aprovado e publicado no mapa." if approved else "Pedido rejeitado."
                st.session_state["review_notice"] = ("success", message)
            else:
                st.session_state["review_notice"] = ("info", "Este pedido já foi avaliado.")
            st.rerun()
    with st.expander("Histórico do pedido"):
        render_review_log(request_id)


def render_admin():
    with st.container(key="admin_workspace"):
        if notice := st.session_state.pop("review_notice", None):
            getattr(st, notice[0])(notice[1])
        queue, log = st.tabs(["Pedidos", "Registo"])
        with queue:
            st.caption(f"{CONTRIBUTIONS.pending_count()} por rever")
            status = st.radio(
                "Estado dos pedidos", list(REVIEW_STATUS), format_func=REVIEW_STATUS.__getitem__,
                horizontal=True, label_visibility="collapsed", key="review-status",
            )
            requests = CONTRIBUTIONS.requests(status)
            if requests:
                request = st.selectbox(
                    "Pedido", requests, key=f"review-request-{status}",
                    format_func=lambda request: f"{request['name']} · {request['id']}",
                )
                render_review_request(request)
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
                render_admin()
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
                key=f"selected-feature-{st.session_state.get('map-point')}",
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


@st.fragment
def render_explorer(filtered, points, layer, map_bounds, admin, features, coordinates):
    with st.container(key="map_surface"):
        event = st_folium(
            build_map(map_bounds), key="catalog-map", height=560, use_container_width=True,
            feature_group_to_add=build_marker_layer(*layer, st.session_state.get("map-view-request"), coordinates),
            returned_objects=[],
        )
    selected = event.get("catalog_selection")
    if isinstance(selected, dict) and selected != st.session_state.get("map-last-click"):
        point = (selected.get("lat"), selected.get("lng"))
        element_id = selected.get("id")
        if all(isinstance(value, (int, float)) for value in point) and any(
            item["properties"]["id"] == element_id for item in points.get(point, ())
        ):
            select_element(element_id, point)
        st.session_state["map-last-click"] = selected

    panel = st.session_state.get("catalog-panel")
    if panel in ("Ficha", "Administração"):
        render_panel(panel, filtered, points, admin, features)


def main() -> None:
    st.html(APP_DIR / "ui.css")
    st.html(APP_DIR / "marker_icons.css")
    admin = st.session_state.get("catalog-admin-mode", False)
    render_brand(admin)
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
    if messages:
        with st.container(key="hud_notices"):
            for level, message in messages:
                getattr(st, level)(message)
    available = [feature for feature in features if admin or feature["properties"].get("estado") != "rejeitado"]
    filters_open = st.session_state.get("catalog-filters-open", False)
    with st.container(key="hud_top", horizontal_alignment="center", gap="small"):
        with st.container(key="hud_controls", horizontal=True, horizontal_alignment="center", gap="small"):
            render_search()
            st.button(
                "Filtros", icon=":material/tune:", key="map-filters", on_click=toggle_filters,
                type="primary" if filters_open else "secondary",
            )
            render_navigation(admin)
        filtered = render_filters(available, admin, filters_open)
    points = group_points(filtered)
    render_map_legend(filtered)
    render_view_controls(points, filtered, admin)
    if st.session_state.get("upload_open"):
        render_upload()
    render_explorer(
        filtered, points, marker_data(points), map_bounds, admin, features, st.session_state.get("search-marker"),
    )


if __name__ == "__main__":
    main()
