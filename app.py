from __future__ import annotations

import base64
import csv
import html
import io
import json
import logging
import math
import os
import re
import sqlite3
import tempfile
import unicodedata
from collections import Counter
from hashlib import sha256
from pathlib import Path
from threading import Lock

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


ROOT = Path(__file__).resolve().parent
MUNICIPALITY = os.environ.get("CTRL_F_MUNICIPALITY", "Aveiro").strip() or "Aveiro"
CATALOG_PATH = Path(
    os.environ.get("CTRL_F_CATALOG_PATH", str(ROOT / "data" / "catalogo" / "catalogo.geojson"))
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
FILTER_KEYS = (
    "catalog-categories", "catalog-subtypes", "catalog-source", "catalog-confidence", "catalog-search",
)
COORDINATE_FORMATS = (
    re.compile(r"([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)"),
    re.compile(r"([+-]?\d+(?:[.,]\d+)?)(?:\s*;\s*|\s+)([+-]?\d+(?:[.,]\d+)?)"),
)
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


def csv_bytes(features: list[dict]) -> bytes:
    fields = list(CSV_COLUMNS)
    extra_fields = sorted(
        {
            key
            for feature in features
            for key in feature.get("properties", {})
            if key not in fields
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


def finish_publication(published, results, revision):
    errors = [message for level, message in results if level == "error"]
    st.session_state["upload_errors"] = errors
    st.session_state["upload_messages"] = [(level, message) for level, message in results if level != "error"]
    st.session_state["upload_open"] = bool(errors)
    if not errors:
        st.session_state["upload_revision"] = revision + 1
    if published:
        st.session_state["new_contribution"] = published[0]
    st.rerun()


def import_mapillary_photo(key, token):
    digest = f"mapillary:{key}"
    if CONTRIBUTIONS.has_photo(digest):
        return None, ("info", "Esta fotografia do Mapillary já foi analisada.")
    photo = fetch_photo(key, token)
    if photo.coordinates is None:
        raise MapillaryPhotoError("A API não devolveu coordenadas para esta fotografia.")
    with decode_photo(photo.data) as image:
        detections = get_analyzer().analyze(image, already_anonymized=True, panoramic=photo.panoramic)
    if not detections:
        return None, ("info", "A análise não encontrou mobiliário urbano nesta fotografia.")
    credit = {
        "imagem_id": key, "imagem_url": photo_link(key), "autor": photo.author, "licenca": PHOTO_LICENSE,
    }
    items = [({**props, **credit}, crop, preview) for props, crop, preview in detections]
    upload_id = CONTRIBUTIONS.publish(digest, *photo.coordinates, "gps_mapillary", items)
    if upload_id is None:
        return None, ("info", "Esta fotografia do Mapillary já foi analisada.")
    return (upload_id, photo.coordinates), ("success", f"Publicada no mapa ({len(items)} deteções).")


def render_mapillary_import(revision):
    token = mapillary_token()
    raw = st.text_input("Link ou ID da fotografia do Mapillary", key=f"mapillary-image-{revision}")
    key = image_key(raw)
    if not token:
        st.caption("Configura MAPILLARY_TOKEN no ficheiro .env do servidor para importar fotografias.")
    if raw and not key:
        st.caption("Usa um ID numérico ou um link de fotografia com pKey.")
    if not st.button(
        "Analisar e publicar do Mapillary", icon=":material/add_a_photo:", type="primary",
        width="stretch", disabled=not (token and key),
    ):
        return
    with st.spinner("A obter a fotografia e a executar OWLv2, SigLIP e SAM 2…"):
        try:
            published, message = import_mapillary_photo(key, token)
        except (SegmentationError, ValueError) as error:
            published, message = None, ("error", str(error))
        except Exception:
            LOGGER.exception("Erro ao analisar uma fotografia do Mapillary")
            published, message = None, ("error", "Não foi possível analisar e publicar a fotografia.")
    finish_publication([published] if published else [], [message], revision)


@st.dialog("Adicionar fotografias", width="large", on_dismiss=close_upload)
def render_upload():
    st.caption("O GPS da foto situa os elementos no mapa. A localização e os recortes anonimizados ficam públicos.")
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
        "Analisar e publicar", icon=":material/upload:", type="primary",
        disabled=not ready or len(ready) != len(files), width="stretch",
    ):
        return

    published = []
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
            with decode_photo(data) as image:
                detections = get_analyzer().analyze(image)
            upload_id = CONTRIBUTIONS.publish(digest, *coordinates, location_source, detections)
            if upload_id:
                published.append((upload_id, coordinates))
                summary = f"{len(detections)} deteções" if detections else "sem elementos detetados"
                results.append(("success", f"{file.name} · Publicada no mapa ({summary})."))
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
            results.append(("error", f"{file.name} · Não foi possível analisar e publicar a fotografia."))
    finish_publication(published, results, revision)


def render_photo(preview):
    photo = base64.b64encode(preview.image).decode("ascii")
    with Image.open(io.BytesIO(preview.image)) as image:
        ratio = image.width / image.height
    overlay = control = ""
    if preview.mask:
        mask = base64.b64encode(preview.mask).decode("ascii")
        overlay = f'<img class="photo-overlay" src="data:image/png;base64,{mask}" alt="">'
        label = "Segmentação SAM 2"
    elif preview.box:
        left, top, right, bottom = preview.box
        overlay = (
            f'<span class="photo-overlay photo-box" style="left:{left:.2%};top:{top:.2%};'
            f'width:{right - left:.2%};height:{bottom - top:.2%}" aria-hidden="true"></span>'
        )
        label = "Área detetada"
    if overlay:
        control = (
            '<figcaption><label class="photo-switch"><input type="checkbox" checked>'
            f'<span>{label}</span></label></figcaption>'
        )
    st.html(
        f'<figure class="photo-preview"><div class="photo-frame" style="--photo-ratio:{ratio:.5f}">'
        f'<img src="data:image/jpeg;base64,{photo}" alt="Fotografia do elemento selecionado">{overlay}'
        f'</div>{control}</figure>'
    )


@st.cache_resource
def photo_jobs():
    return PreviewQueue()


@st.fragment(run_every=0.5)
def render_photo_progress(job, downloading):
    if job.done():
        st.rerun()
    if downloading:
        st.caption("A obter a fotografia do Mapillary e a preparar a máscara…")
    else:
        st.caption("A gerar a máscara com o SAM 2…")


def render_selected_photo(props, preview, *, token=""):
    if preview is None and not token:
        if image_key(props.get("imagem_id")):
            st.caption("A fotografia falta em data/images. Configura MAPILLARY_TOKEN para a carregar pela API.")
        else:
            st.caption("A fotografia usada na deteção ainda não está disponível.")
        return
    if preview and (preview.mask or not preview.box):
        render_photo(preview)
        if not preview.mask and props.get("imagem_bbox"):
            st.caption("A caixa da deteção não pôde ser alinhada com esta versão da fotografia.")
        return

    jobs = photo_jobs()
    job = jobs.submit(CATALOG_PATH, props, preview, token=token)
    if job is None:
        if preview:
            render_photo(preview)
        st.info("Há várias fotografias em análise. Tenta novamente daqui a pouco.")
    elif not job.done():
        if preview:
            render_photo(preview)
        render_photo_progress(job, preview is None)
        return
    else:
        result = job.result()
        if result.preview:
            render_photo(result.preview)
        if not result.notice:
            return
        st.caption(result.notice)

    if st.button("Tentar novamente", icon=":material/refresh:", key=f"retry-photo-{props['id']}"):
        jobs.submit(CATALOG_PATH, props, preview, token=token, retry=True)
        st.rerun()


def render_photo_source(photo):
    if photo.get("autor"):
        st.caption(f"Fotografia por {photo['autor']} · {photo.get('licenca') or PHOTO_LICENSE}")
    key = image_key(photo.get("imagem_id"))
    if key:
        st.link_button("Ver no Mapillary", photo_link(key), icon=":material/photo_library:", width="stretch")
    elif photo.get("imagem_url"):
        st.link_button("Abrir fotografia", photo["imagem_url"], icon=":material/photo_library:", width="stretch")


def render_image(feature: dict) -> None:
    props = feature.get("properties", {})
    if props.get("fonte") == SOURCE:
        preview = load_catalog_preview(CATALOG_PATH, props, prepared_only=True)
        if preview is None:
            preview = CONTRIBUTIONS.preview(props["id"], props.get("imagem_bbox_relativa"))
            if preview and props.get("segmentacao_modelo") != SAM2_MODEL:
                preview = PhotoPreview(preview.image, box=preview.box)
        if preview:
            render_selected_photo(props, preview)
            render_photo_source(props)
            return
        crop = CONTRIBUTIONS.crop(props["id"])
        if crop:
            render_photo(PhotoPreview(crop))
        else:
            st.caption("A análise desta fotografia não identificou elementos para recortar.")
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
    render_selected_photo(photo, preview, token=token if image_key(photo.get("imagem_id")) else "")
    render_photo_source(photo)


@st.cache_data
def catalog_subtypes():
    with (ROOT / "config" / "taxonomia.yaml").open(encoding="utf-8") as file:
        taxonomy = yaml.safe_load(file)["categorias"]
    return {
        category: {item["id"]: item for item in spec["subtipos"]}
        for category, spec in taxonomy.items()
    }


def render_details(feature: dict) -> None:
    props = feature.get("properties", {})
    longitude, latitude = feature["geometry"]["coordinates"][:2]
    category = props.get("categoria")
    status = props.get("estado", "automatico")
    color = CATEGORY_ACCENTS.get(category, "#c0ced8")
    _, pictogram = marker_icon(props)
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
    fields = [
        ("Estado", {"automatico": "Por rever", "confirmado": "Confirmado", "rejeitado": "Rejeitado"}.get(status, status)),
    ]
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


def format_total(value):
    return f"{value:,}".replace(",", "\u202f")


def category_label(category):
    return CATEGORY_LABELS.get(category, category.replace("_", " ").capitalize())


def glyph(category):
    return f":material/{CATEGORY_GLYPHS.get(category, 'location_on')}:"


def render_navigation(admin):
    with st.container(key="hud_brand"):
        mode = '<span class="site-brand__mode">Edição</span>' if admin else ""
        st.html(
            '<div class="site-brand"><span class="site-brand__symbol" aria-hidden="true">⌖</span>'
            '<div><span class="site-brand__name">Ctrl+F</span>'
            f'<span class="site-brand__location">{html.escape(MUNICIPALITY)}</span></div>{mode}</div>'
        )
    with st.container(key="hud_actions"):
        if st.button(
            "Adicionar fotografias", icon=":material/add_a_photo:", type="primary", help="Adicionar fotografias",
        ):
            st.session_state["upload_open"] = True
        label = "Voltar ao mapa" if admin else "Administração"
        if st.button(
            label, icon=":material/settings:", type="primary" if admin else "secondary", help=label,
        ):
            st.session_state["catalog-admin-mode"] = not admin
            st.session_state["catalog-panel"] = "Ficha" if not admin else None
            st.rerun()


def clear_filters():
    for key in FILTER_KEYS:
        st.session_state.pop(key, None)
    st.session_state["catalog-page"] = 0


def normalized_search(value):
    return "".join(
        char for char in unicodedata.normalize("NFKD", value.casefold()) if not unicodedata.combining(char)
    )


def parse_coordinates(text):
    text = text.strip().removeprefix("(").removesuffix(")").strip()
    for pattern in COORDINATE_FORMATS:
        match = pattern.fullmatch(text)
        if match is None:
            continue
        latitude, longitude = (float(value.replace(",", ".")) for value in match.groups())
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise ValueError("Indica uma latitude entre -90 e 90 e uma longitude entre -180 e 180.")
        return latitude, longitude
    return None


def sort_by_proximity(features, coordinates):
    latitude, longitude = coordinates
    longitude_scale = math.cos(math.radians(latitude))

    def distance(feature):
        point_longitude, point_latitude = feature["geometry"]["coordinates"][:2]
        longitude_delta = (point_longitude - longitude + 180) % 360 - 180
        return (point_latitude - latitude) ** 2 + (longitude_delta * longitude_scale) ** 2

    features.sort(key=distance)


def filter_catalog(available, categories, subtypes, source, confidence, search, pending_only):
    matches = []
    for feature in available:
        props = feature["properties"]
        category = props.get("categoria")
        if categories and category not in categories:
            continue
        if subtypes and (category, props.get("subtipo")) not in subtypes:
            continue
        if source and props.get("fonte") != source:
            continue
        if float(props.get("confianca") or 0) < confidence:
            continue
        if pending_only and props.get("estado", "automatico") != "automatico":
            continue
        if search:
            searchable = " ".join(str(props.get(key) or "") for key in ("id", "label", "subtipo", "codigo"))
            if search not in normalized_search(searchable):
                continue
        matches.append(feature)
    return matches


def render_toolbar(available, admin):
    categories = list(dict.fromkeys(
        feature["properties"]["categoria"] for feature in available if feature["properties"].get("categoria")
    ))
    categories.sort(key=category_label)
    subtypes = sorted({
        (props["categoria"], props["subtipo"])
        for feature in available
        if (props := feature["properties"]).get("categoria") and props.get("subtipo")
    })
    sources = sorted({feature["properties"]["fonte"] for feature in available if feature["properties"].get("fonte")})
    with st.container(key="hud_tools"):
        with st.popover("Pesquisa", icon=":material/search:", help="Pesquisar no catálogo"):
            st.subheader("Pesquisar")
            search = st.text_input(
                "Pesquisar no catálogo", placeholder="Nome, tipo ou coordenadas",
                label_visibility="collapsed", key="catalog-search",
            )
            st.caption("Latitude e longitude, por exemplo 40.6412, -8.6549")
            try:
                coordinates = parse_coordinates(search)
            except ValueError as error:
                st.error(str(error))
                coordinates = None
        with st.popover("Filtros", icon=":material/tune:", help="Filtrar elementos"):
            st.subheader("Filtros")
            selected_categories = st.pills(
                "Categorias no mapa", categories, selection_mode="multi",
                format_func=lambda category: f"{glyph(category)} {category_label(category)}",
                key="catalog-categories",
            )
            selected_subtypes = st.multiselect(
                "Subtipos", subtypes, format_func=lambda value: subtype_label(*value), key="catalog-subtypes",
            )
            source = st.selectbox(
                "Fonte", [None, *sources], format_func=lambda value: value or "Todas as fontes", key="catalog-source",
            )
            confidence = st.slider("Confiança mínima (%)", 0, 100, 0, 5, key="catalog-confidence") / 100
            pending_only = admin and st.checkbox("Mostrar apenas por rever", value=True)
            st.button("Limpar filtros", icon=":material/filter_alt_off:", on_click=clear_filters)
    matches = filter_catalog(
        available, selected_categories, selected_subtypes, source, confidence,
        "" if coordinates else normalized_search(search.strip()), pending_only,
    )
    if coordinates:
        sort_by_proximity(matches, coordinates)
    return matches, coordinates


def render_downloads(features):
    filename = f"catalogo_{re.sub(r'[^a-zA-Z0-9_-]+', '_', MUNICIPALITY.lower())}"
    st.download_button(
        "Descarregar GeoJSON",
        data=json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False).encode("utf-8"),
        file_name=f"{filename}.geojson", mime="application/geo+json", width="stretch", on_click="ignore",
    )
    st.download_button(
        "Descarregar CSV", data=csv_bytes(features), file_name=f"{filename}.csv", mime="text/csv",
        width="stretch", on_click="ignore",
    )


def request_view(*, point=None, bounds=None):
    revision = st.session_state.get("map-view-request", {}).get("revision", 0)
    st.session_state["map-view-request"] = {"revision": revision + 1, "point": point, "bounds": bounds}


def select_element(feature_id, point, move_map=False):
    st.session_state.update({"map-point": point, "selected-id": feature_id, "catalog-panel": "Ficha"})
    if move_map:
        request_view(point=point)


def change_page(delta):
    st.session_state["catalog-page"] = st.session_state.get("catalog-page", 0) + delta


def render_result_list(filtered):
    page_size = 6
    last_page = (len(filtered) - 1) // page_size
    page = min(last_page, max(0, st.session_state.get("catalog-page", 0)))
    st.session_state["catalog-page"] = page
    with st.container(key="result_list"):
        for feature in filtered[page * page_size:(page + 1) * page_size]:
            props = feature["properties"]
            element_id = props["id"]
            longitude, latitude = feature["geometry"]["coordinates"][:2]
            st.button(
                feature_label(feature), icon=glyph(props.get("categoria")),
                key=f"result-{element_id}", on_click=select_element, args=(element_id, (latitude, longitude), True),
            )
    if last_page:
        previous, indicator, following = st.columns([1, 0.9, 1])
        with previous:
            st.button("Anterior", disabled=page == 0, on_click=change_page, args=(-1,), width="stretch")
        with indicator:
            st.html(f'<div class="page-number">{page + 1} / {last_page + 1}</div>')
        with following:
            st.button("Seguinte", disabled=page == last_page, on_click=change_page, args=(1,), width="stretch")


def toggle_panel(name):
    st.session_state["catalog-panel"] = None if st.session_state.get("catalog-panel") == name else name


def render_panel(panel, filtered, points, admin, features, coordinates):
    with st.container(key="inspector"):
        title, close = st.columns([6, 1], vertical_alignment="center")
        with title:
            st.subheader(panel)
        with close, st.container(key="hud_close"):
            st.button("Fechar painel", icon=":material/close:", help="Fechar painel", on_click=toggle_panel, args=(panel,))

        if panel == "Exportar":
            render_downloads(filtered)
            return
        if not filtered:
            st.write("Nenhum elemento corresponde aos filtros.")
            if st.button("Limpar filtros e pesquisa", icon=":material/filter_alt_off:"):
                clear_filters()
                st.rerun()
            return
        if panel == "Resultados":
            if coordinates:
                st.caption(f"Por proximidade a {coordinates[0]:.6f}, {coordinates[1]:.6f}")
            render_result_list(filtered)
            return

        at_point = points.get(st.session_state.get("map-point"))
        selected_id = st.session_state.get("selected-id")
        alternatives = at_point or filtered
        selected = next((item for item in alternatives if item["properties"]["id"] == selected_id), None)
        if selected is None:
            selected = alternatives[0] if at_point or admin else None
        if selected is None:
            st.write("Seleciona um marcador ou abre a lista de resultados.")
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
        render_details(selected)
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
    with st.container(key="hud_view"):
        st.button(
            "Resultados", icon=":material/view_list:", help="Ver resultados",
            type="primary" if panel == "Resultados" else "secondary",
            on_click=toggle_panel, args=("Resultados",),
        )
        st.button(
            "Enquadrar", icon=":material/my_location:", help="Enquadrar resultados", disabled=not points,
            on_click=request_view, kwargs={"bounds": point_bounds(points)},
        )
        st.button(
            "Exportar", icon=":material/download:", help="Exportar resultados", disabled=not filtered,
            type="primary" if panel == "Exportar" else "secondary",
            on_click=toggle_panel, args=("Exportar",),
        )
    with st.container(key="hud_status"):
        count = format_total(len(filtered))
        text = f'{count} {"resultado" if len(filtered) == 1 else "resultados"}' if filtered else "Sem resultados"
        st.html(f'<div class="result-count" role="status">{text}</div>')
    if panel:
        render_panel(panel, filtered, points, admin, features, coordinates)


def main() -> None:
    st.html(ROOT / "ui.css")
    admin = st.session_state.get("catalog-admin-mode", False)
    render_navigation(admin)
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
    if st.session_state.get("upload_open"):
        render_upload()
    new_contribution = st.session_state.pop("new_contribution", None)
    if new_contribution:
        upload_id, position = new_contribution
        clear_filters()
        select_element(f"{upload_id}-1", position, move_map=True)
    filtered, coordinates = render_toolbar(available, admin)
    if coordinates != st.session_state.get("searched-coordinates"):
        st.session_state["searched-coordinates"] = coordinates
        if coordinates:
            request_view(point=coordinates)
            st.session_state["catalog-panel"] = "Resultados"
    points = group_points(filtered)
    if st.session_state.get("result-filter") != tuple(st.session_state.get(key) for key in FILTER_KEYS):
        st.session_state["catalog-page"] = 0
        st.session_state["result-filter"] = tuple(st.session_state.get(key) for key in FILTER_KEYS)
    render_explorer(filtered, points, marker_data(points), map_bounds, admin, features, coordinates)


if __name__ == "__main__":
    main()
