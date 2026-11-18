"""
Acesso à Mapillary (API e mosaicos) e ao limite do município (Nominatim/OSM).
Não se corre sozinho: é usado por amostra.py, captura_sequencia.py e fetch_ground_truth.py.

As marcações e os "points" da Mapillary servem SÓ para medir; nunca entram no catálogo.
Requer MAPILLARY_TOKEN no ficheiro .env (Client Token "MLY|...").
"""

from __future__ import annotations

import base64
import json
import os
import sys
import threading

import mapbox_vector_tile
import mercantile
import requests
from shapely.geometry import shape

import comum  # noqa: F401  (carrega o .env)

UA = {"User-Agent": "street-furniture-hackathon-script/0.1"}
GRAPH = "https://graph.mapillary.com"
TILE_URL = "https://tiles.mapillary.com/maps/vtp/mly1_public/2/{z}/{x}/{y}?access_token={token}"
MLY_URL = "https://tiles.mapillary.com/maps/vtp/{tileset}/2/{z}/{x}/{y}?access_token={token}"
MLY_TILES = {  # tileset -> camada dentro do mosaico
    "mly_map_feature_point": "point",
    "mly_map_feature_traffic_sign": "traffic_sign",
}
IMAGE_FIELDS = ("id,captured_at,compass_angle,computed_compass_angle,geometry,"
                "computed_geometry,camera_type,camera_parameters,width,height,"
                "creator,sequence,is_pano,{thumb}")
_local = threading.local()


def token():
    t = os.getenv("MAPILLARY_TOKEN", "").strip()
    if not t or "COLA_AQUI" in t:
        sys.exit("Falta o MAPILLARY_TOKEN no ficheiro .env")
    return t


def session():
    """Uma sessão HTTP por thread."""
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers.update(UA)
    return _local.s


def get_boundary(name: str):
    """Limite administrativo do concelho via Nominatim (OSM)."""
    r = requests.get("https://nominatim.openstreetmap.org/search", headers=UA, timeout=30,
                     params={"q": f"{name}, Portugal", "format": "jsonv2", "polygon_geojson": 1,
                             "extratags": 1, "limit": 10})
    r.raise_for_status()
    results = r.json()
    for res in results:
        tags = res.get("extratags") or {}
        if (res.get("category") == "boundary" and tags.get("admin_level") == "7"  # concelho
                and res["geojson"]["type"] in ("Polygon", "MultiPolygon")):
            return shape(res["geojson"]), res["display_name"]
    sys.exit(f"Não encontrei o limite do concelho '{name}'. Resultados: "
             + "; ".join(r.get("display_name", "?") for r in results))


def tile_points(tile: mercantile.Tile, content: bytes, layer: str = "image"):
    """Descodifica um mosaico vetorial e devolve (lon, lat, props) dos pontos da camada."""
    decoded = mapbox_vector_tile.decode(content, default_options={"y_coord_down": True})
    layer = decoded.get(layer)
    if not layer:
        return
    extent = layer.get("extent", 4096)
    b = mercantile.xy_bounds(tile)  # metros (Web Mercator)
    for feat in layer["features"]:
        geom = feat["geometry"]
        if geom["type"] != "Point":
            continue
        px, py = geom["coordinates"]
        mx = b.left + px / extent * (b.right - b.left)
        my = b.top - py / extent * (b.top - b.bottom)
        lon, lat = mercantile.lnglat(mx, my)
        yield lon, lat, feat["properties"]


def categorize_mly(value: str):
    """Classe Mapillary -> (categoria, subtipo provisório), ou None se não interessa."""
    if value == "object--street-light":
        return "iluminacao", "candeeiro"
    if value in ("object--support--utility-pole", "object--support--pole"):
        return "iluminacao", "poste"   # depois: poste_luz (com luminária perto) ou poste_eletrico
    if value == "object--trash-can":
        return "residuos", "contentor"
    if value.startswith(("regulatory--", "warning--", "information--", "complementary--")):
        if "--stop--" in value:
            return "sinalizacao", "stop"
        if "--yield--" in value:
            return "sinalizacao", "cedencia"
        if "maximum-speed-limit" in value:
            return "sinalizacao", "limite_velocidade"
        return "sinalizacao", value
    return None


def decode_detection(geom_b64, width, height):
    """Geometria Mapillary (MVT em base64, extent 4096) -> bbox em píxeis."""
    tile = mapbox_vector_tile.decode(base64.b64decode(geom_b64), default_options={"y_coord_down": True})
    xs, ys = [], []
    for layer in tile.values():
        extent = layer.get("extent", 4096)
        for feat in layer["features"]:
            g = feat["geometry"]
            rings = g["coordinates"] if g["type"] == "Polygon" else [r for poly in g["coordinates"] for r in poly]
            for ring in rings:
                for x, y in ring:
                    xs.append(x / extent * width)
                    ys.append(y / extent * height)
    if not xs:
        return None
    return [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)]


def fetch_one(image_id, tok, thumb, img_dir):
    """Descarrega 1 foto para img_dir + metadados (pose, câmara, autor) + marcações Mapillary."""
    from PIL import Image
    s = session()
    r = s.get(f"{GRAPH}/{image_id}", timeout=30,
              params={"access_token": tok, "fields": IMAGE_FIELDS.format(thumb=thumb)})
    r.raise_for_status()
    m = r.json()
    url = m.get(thumb)
    if not url:
        raise RuntimeError("sem URL de miniatura")
    path = img_dir / f"{image_id}.jpg"
    if not path.exists():
        img = s.get(url, timeout=60)
        img.raise_for_status()
        path.write_bytes(img.content)
    with Image.open(path) as im:  # dimensões da miniatura (a Mapillary devolve as do original)
        tw, th = im.size

    d = s.get(f"{GRAPH}/{image_id}/detections", timeout=30,
              params={"access_token": tok, "fields": "id,value,geometry"})
    dets = []
    if d.status_code == 200:
        for det in d.json().get("data", []):
            cat = categorize_mly(det.get("value", ""))
            if not cat:
                continue
            bbox = decode_detection(det["geometry"], tw, th) if det.get("geometry") else None
            if bbox:
                dets.append({"value": det["value"], "category": cat[0], "subtype": cat[1], "bbox": bbox})

    geom = m.get("computed_geometry") or m.get("geometry") or {}
    creator = m.get("creator") or {}
    meta = {
        "id": image_id,
        "file": path.name,
        "thumb_w": tw, "thumb_h": th,
        "width": m.get("width"), "height": m.get("height"),
        "lon": geom.get("coordinates", [None, None])[0],
        "lat": geom.get("coordinates", [None, None])[1],
        "lon_gps": (m.get("geometry") or {}).get("coordinates", [None])[0],
        "lat_gps": (m.get("geometry") or {}).get("coordinates", [None, None])[1],
        "compass_angle": m.get("computed_compass_angle", m.get("compass_angle")),
        "camera_type": m.get("camera_type"),
        "camera_parameters": json.dumps(m.get("camera_parameters")),
        "is_pano": m.get("is_pano"),
        "captured_at": m.get("captured_at"),
        "sequence": m.get("sequence"),
        "creator_username": creator.get("username"),
        "creator_id": creator.get("id"),
        "license": "CC BY-SA 4.0",
        "source_url": f"https://www.mapillary.com/app/?pKey={image_id}",
    }
    return meta, dets
