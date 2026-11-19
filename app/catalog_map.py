"""Marcadores e agrupamento dos pontos do catálogo."""

from __future__ import annotations

import base64
import html
import io
import json
import re
from functools import lru_cache
from pathlib import Path

import folium
from branca.element import Element, MacroElement
from folium.template import Template
from PIL import Image

from contributions import PHOTO_CATEGORY

SIGNS = Path(__file__).resolve().parents[1] / "data" / "referencias" / "sinais"
MAX_MAP_ZOOM = 22
NATIVE_TILE_ZOOM = 19
CATEGORY_LABELS = {
    "sinalizacao": "Sinalização",
    "semaforos": "Semáforos",
    "iluminacao": "Iluminação pública",
    "residuos": "Resíduos",
    PHOTO_CATEGORY: "Fotografias",
}
CATEGORY_ICONS = {
    "sinalizacao": ("⚠", "#8b332b"),
    "semaforos": ("🚦", "#544117"),
    "iluminacao": ("💡", "#145f58"),
    "residuos": ("🗑", "#275885"),
    PHOTO_CATEGORY: ("📷", "#475569"),
}
WASTE_ICONS = {
    "ilha_reciclagem": ("♻", "#22684f"),
    "ecoponto_papel": ("♻", "#275885"),
    "ecoponto_embalagens": ("♻", "#f0c44d"),
    "ecoponto_vidro": ("♻", "#28754e"),
    "subterraneo": ("🗑", "#475569"),
}
CATEGORY_ACCENTS = {
    "sinalizacao": "#a63f46",
    "semaforos": "#845b0b",
    "iluminacao": "#177750",
    "residuos": "#245ab1",
    PHOTO_CATEGORY: "#744bae",
}

MAP_STYLES = """
<style>
  html, body {margin:0; overflow:hidden}
  #map_div {height:100vh !important; height:100dvh !important}
  .leaflet-container {background:#e6edf0; font-family:system-ui,sans-serif}
  .catalog-dark-tiles {filter:invert(1) hue-rotate(188deg) brightness(.96) contrast(.89)}
  .leaflet-top.leaflet-left {top:auto; bottom:42px; left:auto; right:0}
  .leaflet-top.leaflet-left .leaflet-control {margin:0 18px 0 0}
  .leaflet-top.leaflet-right .leaflet-control {margin:18px 18px 0 0}
  .leaflet-bottom.leaflet-left .leaflet-control-scale {margin:0 0 24px 186px}
  .leaflet-control-layers, .leaflet-bar {
    border:1px solid #fff !important; border-radius:17px !important; background:#fffefa;
    box-shadow:0 5px 21px #223a5330 !important; backdrop-filter:blur(14px)}
  .leaflet-bar a, .leaflet-control-layers-toggle {
    color:#223452 !important; background:#fffefa; width:39px !important; height:39px !important; line-height:39px !important}
  .leaflet-bar a:hover, .leaflet-bar a:focus-visible {background:#e7f1fc !important; color:#0b62d3 !important}
  .leaflet-bar a:first-child {border-radius:16px 16px 0 0 !important}
  .leaflet-bar a:last-child {border-radius:0 0 16px 16px !important}
  .leaflet-bar a + a {border-top:1px solid #e5ecf2}
  .leaflet-control-layers-expanded {color:#253652; padding:9px 13px}
  .leaflet-control-layers:not(.leaflet-control-layers-expanded), .leaflet-control-layers-toggle {border-radius:50% !important}
  .leaflet-control-layers-toggle {background-image:none !important; text-align:center}
  .leaflet-control-layers-toggle::after {content:'▤'; font-size:20px}
  .leaflet-control-attribution, .leaflet-control-scale-line {background:#ffffffdc !important; color:#223452 !important}
  .leaflet-control-attribution a {color:#0c57ac}
  .leaflet-control-scale-line {border-color:#4e657f !important; text-shadow:none !important}
  .catalog-pictogram {
    display:grid; place-items:center; width:26px; height:26px; border-radius:9px;
    background:var(--glyph-bg,#e9eef4); color:var(--glyph-color,#435673)}
  .catalog-symbol {
    font:18px/1 "Segoe UI Emoji","Apple Color Emoji",sans-serif;
  }
  .catalog-marker-glyph {
    position:relative; display:grid; place-items:center; width:34px; height:34px;
    border-radius:13px; border:1px solid #fff; box-sizing:border-box; background:#fff;
    box-shadow:0 3px 11px #253e603e;
    transition:transform .18s ease,box-shadow .18s ease}
  .catalog-map-marker:hover .catalog-marker-glyph, .catalog-selected .catalog-marker-glyph {
    transform:translateY(-2px); box-shadow:0 0 0 3px #4d9afb,0 6px 15px #1e518f50}
  .catalog-marker-stack {box-shadow:3px -3px 0 #afc9e8,0 3px 11px #253e603e}
  .catalog-choice-popup .leaflet-popup-content-wrapper, .catalog-choice-popup .leaflet-popup-tip {
    background:#fffffff5; color:#1d2f49; box-shadow:0 8px 26px #22466c34}
  .catalog-choice-popup .leaflet-popup-content-wrapper {border-radius:18px}
  .catalog-choice-popup .leaflet-popup-content {margin:12px; max-width:min(274px,calc(100vw - 112px))}
  .catalog-choice-popup a.leaflet-popup-close-button {right:4px; top:2px; color:#53647a}
  .catalog-choice {width:min(270px,calc(100vw - 118px))}
  .catalog-choice > strong {display:block; margin:0 19px .55rem 2px; font-size:13px}
  .catalog-choice-grid {display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:5px}
  .catalog-choice-item {
    display:grid; justify-items:center; align-items:start; gap:5px; width:100%; min-width:0;
    padding:6px 2px; border:0; border-radius:10px; background:#f0f5fa; color:#183857; cursor:pointer}
  .catalog-choice-item > span:last-child {
    max-width:100%; overflow-wrap:anywhere; font:600 10px/1.2 system-ui;
    display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden}
  .catalog-choice-icon {display:grid; place-items:center; height:30px}
  .catalog-choice-footer {display:flex; gap:5px; margin-top:7px}
  .catalog-choice-footer:empty {display:none}
  .catalog-choice-footer button {
    flex:1; border:0; border-radius:8px; padding:5px 2px; background:#e4f0ff; color:#0b54a6;
    font:600 10px system-ui; cursor:pointer}
  .catalog-choice button:hover {background:#d8ebff}
  .catalog-choice button:focus-visible {outline:2px solid #1675eb; outline-offset:2px}
  .catalog-choice button:disabled {opacity:.42; cursor:default}
  @media (prefers-reduced-motion:reduce) {.catalog-marker-glyph {transition:none}}
</style>
"""

class CatalogPoints(MacroElement):
    _template = Template("""
        {% macro script(this, kwargs) %}
        var {{ this.get_name() }} = (function () {
            const rows = {{ this.rows | tojson }};
            const icons = {{ this.icons | tojson }};
            const map = {{ this._parent._parent.get_name() }};
            {{ this.interaction }}
            layer.addTo({{ this._parent.get_name() }});
            return layer;
        })();
        {% endmacro %}
    """)
    interaction = Path(__file__).with_name("marker_menu.js").read_text(encoding="utf-8")

    def __init__(self, rows, icons):
        super().__init__()
        self.rows = rows
        self.icons = icons


class ViewportAction(MacroElement):
    _template = Template("""
        {% macro script(this, kwargs) %}
        if (window.catalogViewRequest !== {{ this.revision }}) {
            window.catalogViewRequest = {{ this.revision }};
            const map = {{ this._parent._parent.get_name() }};
            const animate = !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
            {% if this.point %}
            map.flyTo({{ this.point | tojson }}, 18, {animate, duration: 0.55});
            {% elif this.bounds %}
            map.flyToBounds({{ this.bounds | tojson }}, {animate, maxZoom: 16, duration: 0.55, padding: [28, 28]});
            {% endif %}
        }
        {% endmacro %}
    """)

    def __init__(self, revision, point=None, bounds=None):
        super().__init__()
        self.revision = revision
        self.point = point
        self.bounds = bounds


@lru_cache(maxsize=320)
def sign_image(code):
    if not re.fullmatch(r"[A-Z]{1,2}[0-9]{1,2}[a-z]?", code):
        return None
    path = SIGNS / f"{code}.png"
    if not path.is_file():
        return None
    with Image.open(path) as image:
        image.thumbnail((60, 60), Image.Resampling.LANCZOS)
        data = io.BytesIO()
        image.save(data, format="PNG")
    return "data:image/png;base64," + base64.b64encode(data.getvalue()).decode("ascii")


def marker_icon(properties):
    category = properties.get("categoria")
    subtype = properties.get("subtipo")
    code = str(properties.get("codigo") or "").removeprefix("PT:") if category == "sinalizacao" else ""
    if code == "H1":
        code = "H1a"
    url = sign_image(code) if code else None
    if url:
        image = f'<img src="{url}" alt="" style="width:22px;height:22px;object-fit:contain">'
        if code == "C13":
            value = str(properties.get("valor") or "")
            speed = value if re.fullmatch(r"[0-9]{1,3}", value) else "·"
            image += (
                '<span style="position:absolute;inset:5px;background:#fff9fa;color:#111;'
                'border-radius:50%;display:grid;place-items:center;font:bold 8px Arial">'
                f'{speed}</span>'
            )
            code += f"-{speed}"
        return code, f'<span class="catalog-pictogram" style="position:relative;background:white">{image}</span>'
    category = category if category in CATEGORY_ICONS else ""
    symbol, background = CATEGORY_ICONS.get(category, ("•", "#475569"))
    if category == "residuos":
        symbol, background = WASTE_ICONS.get(subtype, (symbol, background))
    color = "#32250b" if background == "#f0c44d" else "white"
    return f"{category}-{symbol}-{background}", (
        f'<span class="catalog-pictogram" style="--glyph-bg:{background};--glyph-color:{color}">'
        f'<span aria-hidden="true" class="catalog-symbol">{symbol}</span></span>'
    )


def group_points(features):
    points = {}
    for feature in features:
        coordinates = feature.get("geometry", {}).get("coordinates", [])
        if len(coordinates) >= 2:
            longitude, latitude = coordinates[:2]
            points.setdefault((latitude, longitude), []).append(feature)
    return points


def point_bounds(points):
    if not points:
        return None
    latitudes, longitudes = zip(*points)
    return [[min(latitudes), min(longitudes)], [max(latitudes), max(longitudes)]]


def build_map(bounds):
    origin = (
        [(bounds[0][axis] + bounds[1][axis]) / 2 for axis in (0, 1)]
        if bounds else [40.6412, -8.6549]
    )
    map_view = folium.Map(
        location=origin,
        zoom_start=13,
        tiles=None,
        control_scale=True,
    )
    map_view.get_root().header.add_child(Element(MAP_STYLES), name="catalog-styles")
    folium.TileLayer(
        "OpenStreetMap", name="Mapa claro", max_zoom=MAX_MAP_ZOOM, max_native_zoom=NATIVE_TILE_ZOOM,
    ).add_to(map_view)
    folium.TileLayer(
        "OpenStreetMap", name="Mapa escuro", show=False, class_name="catalog-dark-tiles",
        max_zoom=MAX_MAP_ZOOM, max_native_zoom=NATIVE_TILE_ZOOM,
    ).add_to(map_view)
    folium.LayerControl(position="topright").add_to(map_view)
    if bounds:
        map_view.fit_bounds(bounds, max_zoom=15, padding=(24, 24))
    return map_view


def marker_data(points):
    rows, icons = [], {}
    for (latitude, longitude), features in points.items():
        entries = []
        for feature in features:
            props = feature["properties"]
            key, pictogram = marker_icon(props)
            icons.setdefault(key, pictogram)
            label = str(props.get("label") or props.get("subtipo") or "Elemento")
            entries.append((props["id"], key, label, CATEGORY_ACCENTS.get(props.get("categoria"), "#4f647d")))
        rows.append((latitude, longitude, entries))
    return rows, icons


def build_marker_layer(rows, icons, view_request=None, coordinates=None):
    group = folium.FeatureGroup(name="Elementos", control=False)
    CatalogPoints(rows, icons).add_to(group)
    if coordinates:
        folium.CircleMarker(
            coordinates, radius=8, color="#0868dc", weight=2,
            fill=True, fill_color="#fff", fill_opacity=0.8,
            interactive=False,
        ).add_to(group)
    if view_request:
        ViewportAction(**view_request).add_to(group)
    return group
