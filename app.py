from __future__ import annotations

import csv
import html
import hmac
import io
import json
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import quote

import folium
import streamlit as st
from folium.plugins import FastMarkerCluster
from streamlit_folium import st_folium


ROOT = Path(__file__).resolve().parent
MUNICIPALITY = os.environ.get("CTRL_F_MUNICIPALITY", "Aveiro").strip() or "Aveiro"
CATALOG_PATH = Path(
    os.environ.get("CTRL_F_CATALOG_PATH", str(ROOT / "data" / "catalogo" / "catalogo.geojson"))
).expanduser()
if not CATALOG_PATH.is_absolute():
    CATALOG_PATH = ROOT / CATALOG_PATH
CSV_PATH = CATALOG_PATH.with_name("catalogo.csv")
CATEGORY_LABELS = {
    "sinalizacao": "Sinalização",
    "semaforos": "Semáforos",
    "iluminacao": "Iluminação pública",
    "residuos": "Resíduos",
}
CATEGORY_COLORS = {
    "sinalizacao": "#e25858",
    "semaforos": "#f59e0b",
    "iluminacao": "#16877a",
    "residuos": "#3978c5",
}
CSV_COLUMNS = [
    "id", "categoria", "categoria_label", "subtipo", "label", "codigo", "valor",
    "lon", "lat", "confianca", "concordancia", "n_vistas", "n_imagens", "membros",
    "imagem_id", "imagem_url", "autor", "licenca", "fonte", "estado",
]

st.set_page_config(
    page_title=f"Ctrl+F {MUNICIPALITY} | Catálogo urbano",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_catalog() -> list[dict]:
    with CATALOG_PATH.open(encoding="utf-8") as file:
        collection = json.load(file)
    if collection.get("type") != "FeatureCollection":
        raise ValueError(f"Formato GeoJSON inválido: {CATALOG_PATH}")
    return collection.get("features", [])


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


def save_catalog(features: list[dict]) -> None:
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    collection = json.dumps(
        {"type": "FeatureCollection", "features": features},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    outputs = ((CATALOG_PATH, collection), (CSV_PATH, csv_bytes(features)))
    temporary_paths = []
    try:
        for target, content in outputs:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=target.parent, prefix=f".{target.name}.", delete=False
            ) as temporary:
                temporary.write(content)
                temporary_paths.append((Path(temporary.name), target))
        for temporary, target in temporary_paths:
            os.replace(temporary, target)
    finally:
        for temporary, _ in temporary_paths:
            if temporary.exists():
                temporary.unlink()


def feature_label(feature: dict) -> str:
    props = feature.get("properties", {})
    return f"{props.get('id', 'Sem ID')} · {props.get('label', props.get('subtipo', 'Elemento'))}"


def build_map(features: list[dict]) -> folium.Map | None:
    points = []
    for feature in features:
        coordinates = feature.get("geometry", {}).get("coordinates", [])
        if len(coordinates) < 2:
            continue
        longitude, latitude = coordinates[:2]
        props = feature.get("properties", {})
        label = html.escape(str(props.get("label") or props.get("subtipo") or "Elemento"))
        item_id = html.escape(str(props.get("id") or ""))
        category = html.escape(
            str(props.get("categoria_label") or props.get("categoria") or "Tipo não indicado")
        )
        confidence = float(props.get("confianca") or 0)
        image_id = props.get("imagem_id")
        image_link = (
            f'<br><a href="https://www.mapillary.com/app/?pKey={quote(str(image_id))}" '
            'target="_blank" rel="noopener noreferrer">Ver imagem no Mapillary</a>'
            if image_id else ""
        )
        popup = (
            f'<div style="min-width:170px"><strong>{label}</strong>'
            f'<br>{category} · {confidence:.0%}'
            f'<br><span style="color:#64748b">{latitude:.6f}, {longitude:.6f}</span>'
            f'<br><span style="color:#64748b">{item_id}</span>{image_link}</div>'
        )
        tooltip = f"<strong>{label}</strong><br>{category} · {confidence:.0%}"
        color = CATEGORY_COLORS.get(props.get("categoria"), "#64748b")
        points.append([latitude, longitude, popup, color, tooltip])

    if not points:
        return None

    latitudes = [point[0] for point in points]
    longitudes = [point[1] for point in points]
    map_view = folium.Map(
        location=[sum(latitudes) / len(points), sum(longitudes) / len(points)],
        zoom_start=13,
        tiles="OpenStreetMap",
        control_scale=True,
    )
    marker_callback = """
    function (row) {
        var icon = L.divIcon({
            className: 'catalog-map-marker',
            html: '<span style="display:block;width:12px;height:12px;box-sizing:border-box;'
                + 'border:2px solid #fff;border-radius:50%;'
                + 'box-shadow:0 1px 5px rgba(15,23,42,.55);background:' + row[3] + '"></span>',
            iconSize: [16, 16],
            iconAnchor: [8, 8]
        });
        var marker = L.marker([row[0], row[1]], {icon: icon});
        marker.bindPopup(row[2], {maxWidth: 280});
        marker.bindTooltip(row[4], {direction: 'top', opacity: 0.95});
        return marker;
    }
    """
    FastMarkerCluster(
        points,
        callback=marker_callback,
        options={"maxClusterRadius": 48, "disableClusteringAtZoom": 17},
    ).add_to(map_view)
    padding_lat = max((max(latitudes) - min(latitudes)) * 0.06, 0.003)
    padding_lon = max((max(longitudes) - min(longitudes)) * 0.06, 0.003)
    map_view.fit_bounds(
        [
            [min(latitudes) - padding_lat, min(longitudes) - padding_lon],
            [max(latitudes) + padding_lat, max(longitudes) + padding_lon],
        ],
        max_zoom=15,
        padding=(18, 18),
    )
    return map_view


def render_image(feature: dict) -> None:
    props = feature.get("properties", {})
    image_id = props.get("imagem_id")
    image_url = props.get("imagem_url")
    if image_id:
        embed_url = (
            "https://www.mapillary.com/embed?map_style=Mapillary%20streets"
            f"&image_key={quote(str(image_id))}&style=photo"
        )
        st.iframe(embed_url, height=310)
    elif image_url:
        st.link_button("Abrir imagem de origem", image_url, use_container_width=True)
    else:
        st.info("Este elemento ainda não tem uma imagem associada.")
    if props.get("autor") or image_url:
        author = props.get("autor") or "autor não identificado"
        license_name = props.get("licenca") or "CC BY-SA 4.0"
        credit = f"Imagem Mapillary · {author} · {license_name}"
        if image_url:
            st.markdown(f"[{credit}]({image_url})")
        else:
            st.caption(credit)


def render_details(feature: dict, admin: bool, all_features: list[dict]) -> None:
    props = feature.get("properties", {})
    coordinates = feature.get("geometry", {}).get("coordinates", [None, None])
    latitude, longitude = coordinates[1], coordinates[0]
    st.subheader(props.get("label") or props.get("subtipo") or "Elemento urbano")
    image_col, info_col = st.columns([1.2, 1])
    with image_col:
        render_image(feature)
    with info_col:
        st.metric("Confiança da deteção", f"{float(props.get('confianca') or 0):.0%}")
        st.write(f"**Tipo:** {props.get('categoria_label', props.get('categoria', '—'))}")
        st.write(f"**Subtipo:** {props.get('subtipo', '—')}")
        st.write(f"**Coordenadas:** {latitude:.6f}, {longitude:.6f}")
        st.write(f"**Vistas / imagens:** {props.get('n_vistas', 1)} / {props.get('n_imagens', 1)}")
        status = props.get("estado", "automatico")
        st.write(f"**Estado:** {status.replace('_', ' ').capitalize()}")
        st.link_button(
            "Ver no OpenStreetMap",
            f"https://www.openstreetmap.org/?mlat={latitude}&mlon={longitude}#map=19/{latitude}/{longitude}",
            use_container_width=True,
        )

    if not admin:
        return

    st.divider()
    st.markdown("#### Rever deteção")
    category_options = sorted(
        {item.get("properties", {}).get("categoria", "") for item in all_features}
        | {props.get("categoria", "")}
    )
    category = props.get("categoria", "")
    category_index = category_options.index(category) if category in category_options else 0
    category_choice = st.selectbox(
        "Tipo", category_options, index=category_index,
        format_func=lambda value: CATEGORY_LABELS.get(value, value.replace("_", " ").capitalize()),
        key=f"category-{props.get('id')}",
    )
    subtype_options = sorted(
        {
            item.get("properties", {}).get("subtipo", "")
            for item in all_features
            if item.get("properties", {}).get("categoria") == category_choice
        }
        | ({props.get("subtipo", "")} if category_choice == category else set())
    )
    subtype = props.get("subtipo", "")
    subtype_index = subtype_options.index(subtype) if subtype in subtype_options else 0
    label = st.text_input("Designação", value=props.get("label", ""), key=f"label-{props.get('id')}")
    subtype_choice = st.selectbox(
        "Subtipo", subtype_options, index=subtype_index, key=f"subtype-{props.get('id')}"
    )
    new_latitude = st.number_input(
        "Latitude", value=float(latitude), format="%.7f", key=f"lat-{props.get('id')}"
    )
    new_longitude = st.number_input(
        "Longitude", value=float(longitude), format="%.7f", key=f"lon-{props.get('id')}"
    )
    new_status = st.selectbox(
        "Decisão",
        ["confirmado", "automatico", "rejeitado"],
        index=["confirmado", "automatico", "rejeitado"].index(
            status if status in ("confirmado", "automatico", "rejeitado") else "automatico"
        ),
        format_func=lambda value: {
            "confirmado": "Confirmar elemento",
            "automatico": "Manter por rever",
            "rejeitado": "Rejeitar deteção",
        }[value],
        key=f"status-{props.get('id')}",
    )
    if st.button("Guardar correção", type="primary", key=f"save-{props.get('id')}"):
        saved_feature = next(
            (
                item for item in all_features
                if item.get("properties", {}).get("id") == props.get("id")
            ),
            None,
        )
        if saved_feature is None:
            st.error("O elemento selecionado já não existe no catálogo. Atualize a página.")
            return
        saved_props = saved_feature["properties"]
        saved_props["categoria"] = category_choice
        saved_props["categoria_label"] = CATEGORY_LABELS.get(category_choice, category_choice)
        saved_props["subtipo"] = subtype_choice
        saved_props["label"] = label
        saved_props["estado"] = new_status
        saved_feature["geometry"]["coordinates"] = [new_longitude, new_latitude]
        save_catalog(all_features)
        st.session_state["catalog_saved"] = True
        st.rerun()


def main() -> None:
    try:
        features = load_catalog()
    except (OSError, json.JSONDecodeError, ValueError) as error:
        st.error(f"Não foi possível carregar o catálogo: {error}")
        st.stop()
        return
    if st.session_state.pop("catalog_saved", False):
        st.success("Correção guardada no catálogo GeoJSON e CSV.")

    st.markdown(
        """
        <style>
        .stApp { background: #f5f8f7; }
        [data-testid="stSidebar"] { background: #102b2b; }
        [data-testid="stSidebar"] * { color: #edf5f2; }
        [data-testid="stMetric"] {
            background: white; border: 1px solid #e3ebe8; border-radius: 14px;
            padding: 14px 18px; box-shadow: 0 3px 12px #173a3510;
        }
        h1, h2, h3 { letter-spacing: -0.035em; }
        div.stButton > button[kind="primary"] {
            background: #167d72; border-color: #167d72; border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown(f"# Ctrl+F {MUNICIPALITY}")
        st.caption("Catálogo inteligente de mobiliário urbano")
        st.divider()
        mode = st.radio("Acesso", ["Explorar catálogo", "Administração"])
        admin = False
        if mode == "Administração":
            admin_password = os.environ.get("CTRL_F_ADMIN_PASSWORD")
            if not admin_password:
                st.warning("Configure CTRL_F_ADMIN_PASSWORD para ativar a área de administração.")
            else:
                with st.form("admin-login"):
                    entered_password = st.text_input("Palavra-passe de administrador", type="password")
                    submitted = st.form_submit_button("Entrar como administrador")
                if submitted and hmac.compare_digest(entered_password, admin_password):
                    st.session_state["ctrl_f_admin"] = True
                elif submitted:
                    st.session_state["ctrl_f_admin"] = False
                    st.error("Palavra-passe incorreta.")
                admin = st.session_state.get("ctrl_f_admin", False)
            st.caption("A consulta pública continua disponível sem autenticação.")
        st.divider()

    if mode == "Administração" and not admin:
        st.title("Área de administração")
        st.info("Entre com a palavra-passe configurada para rever e corrigir as deteções.")
        return

    st.title("Catálogo de mobiliário urbano")
    st.caption(f"Deteções georreferenciadas em {MUNICIPALITY} · consulta, revisão e exportação")

    categories = sorted(
        {item.get("properties", {}).get("categoria", "") for item in features if item.get("properties", {}).get("categoria")}
    )
    with st.sidebar:
        selected_categories = st.multiselect(
            "Categorias", categories, default=categories,
            format_func=lambda value: CATEGORY_LABELS.get(value, value.replace("_", " ").capitalize()),
        )
        minimum_confidence = st.slider("Confiança mínima", 0, 100, 0, 5) / 100
        search = st.text_input("Pesquisar por ID ou designação")
        pending_only = False
        if admin:
            pending_only = st.checkbox("Mostrar apenas por rever", value=True)

    filtered = []
    for feature in features:
        props = feature.get("properties", {})
        if props.get("estado") == "rejeitado" and not admin:
            continue
        if props.get("categoria") not in selected_categories:
            continue
        if float(props.get("confianca") or 0) < minimum_confidence:
            continue
        if pending_only and props.get("estado", "automatico") != "automatico":
            continue
        if search and search.casefold() not in (
            f"{props.get('id', '')} {props.get('label', '')} {props.get('subtipo', '')}".casefold()
        ):
            continue
        filtered.append(feature)

    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Elementos", f"{len(filtered):,}")
    metric_b.metric("Categorias", len({item.get("properties", {}).get("categoria") for item in filtered}))
    mean_confidence = (
        sum(float(item.get("properties", {}).get("confianca") or 0) for item in filtered) / len(filtered)
        if filtered else 0
    )
    metric_c.metric("Confiança média", f"{mean_confidence:.0%}")
    st.divider()

    map_col, detail_col = st.columns([1.5, 1], gap="large")
    with map_col:
        st.markdown("### Elementos no mapa")
        if filtered:
            map_view = build_map(filtered)
            if map_view:
                st_folium(
                    map_view,
                    height=610,
                    use_container_width=True,
                    returned_objects=[],
                )
                st.caption(
                    f"{len(filtered):,} elementos agrupados por proximidade. "
                    "Aproxime o mapa para explorar cada ponto."
                )
                legend = " &nbsp; ".join(
                    f'<span style="color:{CATEGORY_COLORS.get(category, "#64748b")}">●</span> '
                    f'{html.escape(CATEGORY_LABELS.get(category, category.capitalize()))}'
                    for category in sorted(
                        {item.get("properties", {}).get("categoria", "") for item in filtered}
                    )
                )
                st.markdown(legend, unsafe_allow_html=True)
        else:
            st.info("Nenhum elemento corresponde aos filtros selecionados.")
        st.markdown("### Exportar catálogo filtrado")
        export_col_a, export_col_b = st.columns(2)
        with export_col_a:
            st.download_button(
                "Descarregar GeoJSON",
                data=json.dumps(
                    {"type": "FeatureCollection", "features": filtered},
                    ensure_ascii=False,
                ).encode("utf-8"),
                file_name=f"catalogo_{re.sub(r'[^a-zA-Z0-9_-]+', '_', MUNICIPALITY.lower())}.geojson",
                mime="application/geo+json",
                use_container_width=True,
            )
        with export_col_b:
            st.download_button(
                "Descarregar CSV",
                data=csv_bytes(filtered),
                file_name=f"catalogo_{re.sub(r'[^a-zA-Z0-9_-]+', '_', MUNICIPALITY.lower())}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with detail_col:
        st.markdown("### Ficha do elemento")
        if filtered:
            selected = st.selectbox(
                "Selecionar elemento",
                filtered,
                format_func=feature_label,
                key="selected-feature",
            )
            render_details(selected, admin, features)
        else:
            st.info("Ajuste os filtros para consultar um elemento.")


if __name__ == "__main__":
    main()
