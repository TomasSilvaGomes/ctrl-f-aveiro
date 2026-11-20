"""Marcadores e agrupamento dos pontos do catálogo."""

from __future__ import annotations

import base64
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
from sign_icons import SIGN_CODES, sign_svg

SIGNS = Path(__file__).resolve().parents[1] / "data" / "referencias" / "sinais"
MAX_MAP_ZOOM = 22
ICON_STYLES = Path(__file__).with_name("marker_icons.css").read_text(encoding="utf-8")
CATEGORY_LABELS = {
    "sinalizacao": "Sinalização",
    "semaforos": "Semáforos",
    "iluminacao": "Iluminação pública",
    "residuos": "Resíduos",
    PHOTO_CATEGORY: "Fotografias",
}
CATEGORY_ICONS = {
    "semaforos": ("signal", "#f4eddf", "#59422a"),
    "iluminacao": ("lamp", "#fff0c8", "#73501d"),
    "residuos": ("bin", "#e8f6ed", "#287455"),
    PHOTO_CATEGORY: ("camera", "#eee8fb", "#704dae"),
}
WASTE_COLORS = {
    "ecoponto_papel": ("#e4eeff", "#375da3"),
    "ecoponto_embalagens": ("#fff1c5", "#795615"),
    "ecoponto_vidro": ("#e0f6e7", "#24654a"),
    "subterraneo": ("#e8edf4", "#425875"),
}
CATEGORY_ACCENTS = {
    "sinalizacao": "#a63f46",
    "semaforos": "#845b0b",
    "iluminacao": "#177750",
    "residuos": "#245ab1",
    PHOTO_CATEGORY: "#744bae",
}
LANTERN_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 24 24" focusable="false">'
    '<path fill="#783c25" d="M11 1h2v2.6l6.4 5.2-1.1 1.4-2 7.9-2.6 1.5v1.6H16V23H8v-1.8h2.3v-1.6'
    'l-2.6-1.5-2-7.9-1.1-1.4L11 3.6z"/>'
    '<path fill="#ffd18f" d="M8.2 10.2h7.6l-1.4 6.3H9.6z"/>'
    '<path fill="#783c25" d="M11.4 10h1.2v7h-1.2z"/>'
    '</svg>'
)

MAP_STYLES = """
<style>
  html, body {margin:0; overflow:hidden}
  #map_div {height:100vh !important; height:100dvh !important}
  .leaflet-container {background:#f8f5ed; font-family:system-ui,sans-serif}
  .catalog-raster-tiles {filter:saturate(.65) contrast(.86) brightness(1.1)}
  .leaflet-top.leaflet-left {top:auto; bottom:20px; left:auto; right:0}
  .leaflet-top.leaflet-left .leaflet-control {margin:0 18px 0 0}
  .leaflet-bottom.leaflet-left .leaflet-control {margin-left:15px}
  .leaflet-bottom.leaflet-left .leaflet-control-attribution {font-size:10px; border-radius:10px; margin-bottom:9px; padding:2px 6px}
  .leaflet-bottom.leaflet-left .leaflet-control-scale {
    padding:6px 8px; border-radius:10px; background:#fffffff2; margin-bottom:4px; box-shadow:0 3px 12px #25385020}
  .leaflet-control-scale .leaflet-control-scale-line {
    border:0; border-bottom:2px solid #253954; border-radius:0; background:transparent !important;
    text-align:center; font:650 11px/1.05 system-ui,sans-serif; padding:0 0 4px}
  .leaflet-bar {
    border:1px solid #fff !important; border-radius:17px !important; background:#fffefa;
    box-shadow:0 5px 21px #223a5330 !important; backdrop-filter:blur(14px)}
  .leaflet-bar a {
    color:#223452 !important; background:#fffefa; width:39px !important; height:39px !important; line-height:39px !important}
  .leaflet-bar a:hover, .leaflet-bar a:focus-visible {background:#e7f1fc !important; color:#0b62d3 !important}
  .leaflet-bar a:first-child {border-radius:16px 16px 0 0 !important}
  .leaflet-bar a:last-child {border-radius:0 0 16px 16px !important}
  .leaflet-bar a + a {border-top:1px solid #e5ecf2}
  .leaflet-control-attribution, .leaflet-control-scale-line {background:#ffffffdc !important; color:#223452 !important}
  .leaflet-control-attribution a {color:#0c57ac}
  .leaflet-control-scale-line {text-shadow:none !important}
  .catalog-tooltip {font:650 12px system-ui,sans-serif; padding:5px 8px; border-radius:8px;
    border-color:#fff; background:#fff; color:#23334e; box-shadow:0 5px 16px #25395430}
  .catalog-marker-glyph {
    position:relative; display:grid; place-items:center; width:100%; height:100%;
    border-radius:29%; border:1px solid #fff; box-sizing:border-box; background:#fff;
    box-shadow:0 2px 9px #253e6037;
    transition:transform .18s ease,box-shadow .18s ease}
  .catalog-marker-overview {--pictogram-size:20px}
  .catalog-search-pin span {
    position:absolute; inset:1px 4px 6px; border-radius:50% 50% 50% 0;
    transform:rotate(-45deg); background:#176ed7; border:2px solid #fff; box-shadow:0 3px 9px #1755a75c}
  .catalog-search-pin span::after {
    position:absolute; content:""; inset:29%; border-radius:50%; background:#fff}
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
  @media (max-width:630px) {
    .leaflet-control-attribution {font-size:10px !important; line-height:1.4}
  }
  @media (prefers-reduced-motion:no-preference) {
    .catalog-choice-popup .leaflet-popup-content-wrapper,
    .catalog-choice-popup .leaflet-popup-tip {
      animation: catalog-choice-enter .2s ease-out;
    }
    @keyframes catalog-choice-enter {
      from {opacity:0; translate:0 8px}
      to {opacity:1; translate:0 0}
    }
  }
  @media (prefers-reduced-motion:reduce) {
    .catalog-marker-glyph {transition:none}
    .catalog-choice-popup .leaflet-popup-content-wrapper,
    .catalog-choice-popup .leaflet-popup-tip {animation:catalog-choice-fade .14s ease-out}
    @keyframes catalog-choice-fade {
      from {opacity:0}
      to {opacity:1}
    }
  }
</style>
"""

class SimpleBasemap(MacroElement):
    _template = Template("""
        {% macro script(this, kwargs) %}
        (function (map) {
            {{ this.script }}
        })({{ this._parent.get_name() }});
        {% endmacro %}
    """)
    script = Path(__file__).with_name("basemap.js").read_text(encoding="utf-8")


class CatalogPoints(MacroElement):
    _template = Template("""
        {% macro script(this, kwargs) %}
        var {{ this.get_name() }} = (function () {
            const rows = {{ this.rows | tojson }};
            let icons = {{ this.icons | tojson }};
            const searchPin = {{ this.search_pin | tojson }};
            const map = {{ this._parent._parent.get_name() }};
            if (map.catalogPoints) {
                map.catalogPoints.update(rows, icons, searchPin);
                return map.catalogPoints.layer;
            }
            {{ this.interaction }}
            map.catalogPoints = {layer, update: updateCatalog};
            layer.addTo(map);
            updateCatalog(rows, icons, searchPin);
            return layer;
        })();
        {% endmacro %}
    """)
    interaction = Path(__file__).with_name("marker_menu.js").read_text(encoding="utf-8")

    def __init__(self, rows, icons, search_pin=None):
        super().__init__()
        self.rows = rows
        self.icons = icons
        self.search_pin = search_pin


class ViewportAction(MacroElement):
    _template = Template("""
        {% macro script(this, kwargs) %}
        if (window.catalogViewRequest !== {{ this.revision }}) {
            window.catalogViewRequest = {{ this.revision }};
            const map = {{ this._parent._parent.get_name() }};
            const animate = !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
            {% if this.point %}
            map.flyTo({{ this.point | tojson }}, {{ this.zoom }}, {animate, duration: 0.55});
            {% elif this.bounds %}
            map.flyToBounds({{ this.bounds | tojson }}, {animate, maxZoom: 16, duration: 0.55, padding: [28, 28]});
            {% endif %}
        }
        {% endmacro %}
    """)

    def __init__(self, revision, point=None, bounds=None, zoom=18):
        super().__init__()
        self.revision = revision
        self.point = point
        self.bounds = bounds
        self.zoom = zoom


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


def icon_svg(drawing, inline):
    if inline:
        return drawing
    encoded = base64.b64encode(drawing.encode("utf-8")).decode("ascii")
    return f'<img src="data:image/svg+xml;base64,{encoded}" alt="">'


def marker_icon(properties, *, inline_svg=True):
    category = properties.get("categoria")
    subtype = properties.get("subtipo")
    if category == "sinalizacao":
        code = str(properties.get("codigo") or "").removeprefix("PT:")
        known_subtype = SIGN_CODES.get(code)
        official_drawing = sign_image(code) if code and not known_subtype else None
        if official_drawing:
            key, drawing = f"sign-{code}", f'<img src="{official_drawing}" alt="">'
        else:
            key, drawing = sign_svg(known_subtype or subtype, properties.get("valor"))
            drawing = icon_svg(drawing, inline_svg)
        return key, f'<span class="catalog-pictogram catalog-pictogram--sign">{drawing}</span>'
    if category == "iluminacao" and subtype == "candeeiro":
        return "iluminacao-candeeiro", (
            '<span class="catalog-pictogram" style="--glyph-bg:#fae7d7">'
            '<span class="catalog-symbol catalog-symbol--candeeiro" aria-hidden="true">'
            f'{icon_svg(LANTERN_SVG, inline_svg)}</span></span>'
        )
    category = category if category in CATEGORY_ICONS else ""
    symbol, background, color = CATEGORY_ICONS.get(category, ("generic", "#e8edf4", "#425875"))
    if category == "residuos":
        background, color = WASTE_COLORS.get(subtype, (background, color))
    return f"{category}-{symbol}-{background}", (
        f'<span class="catalog-pictogram" style="--glyph-bg:{background};--glyph-color:{color}">'
        f'<span aria-hidden="true" class="catalog-symbol catalog-symbol--{symbol}"></span></span>'
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
        attributionControl=False,
        minZoom=1,
        maxZoom=MAX_MAP_ZOOM,
    )
    map_view.get_root().header.add_child(Element(f"{MAP_STYLES}<style>{ICON_STYLES}</style>"), name="catalog-styles")
    SimpleBasemap().add_to(map_view)
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
            entries.append((props["id"], key, label))
        rows.append((latitude, longitude, entries))
    return rows, icons


def build_marker_layer(rows, icons, view_request=None, search_pin=None):
    group = folium.FeatureGroup(name="Elementos", control=False)
    CatalogPoints(rows, icons, search_pin).add_to(group)
    if view_request:
        ViewportAction(**view_request).add_to(group)
    return group
