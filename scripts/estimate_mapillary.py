"""
Conta todas as imagens Mapillary dentro do concelho de Aveiro e estima o
espaço em disco necessário para as descarregar.

Não descarrega o dataset: lista as imagens através das vector tiles de
cobertura (zoom 14, camada "image") e descarrega apenas uma pequena amostra
de miniaturas para medir o tamanho médio.

Uso:
    python scripts/estimate_mapillary.py               # concelho de Aveiro
    python scripts/estimate_mapillary.py --sample 60   # amostra maior

Requer MAPILLARY_TOKEN no ficheiro .env (Client Token "MLY|...").
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import mapbox_vector_tile
import mercantile
import requests
from dotenv import load_dotenv
from shapely.geometry import Point, shape
from shapely.prepared import prep

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
# Tudo no workspace: <projeto>/data (pode ser alterado com DATA_DIR no .env)
DATA_DIR = Path(os.getenv("DATA_DIR") or (ROOT / "data"))
META_DIR = DATA_DIR / "meta"
UA = {"User-Agent": "street-furniture-hackathon-script/0.1"}
TILE_URL = "https://tiles.mapillary.com/maps/vtp/mly1_public/2/{z}/{x}/{y}?access_token={token}"
GRAPH_URL = "https://graph.mapillary.com/{id}"
ZOOM = 14  # camada "image" só existe no zoom 14


def get_boundary(name: str):
    """Limite administrativo do concelho via Nominatim (OSM)."""
    r = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={
            "q": f"{name}, Portugal",
            "format": "jsonv2",
            "polygon_geojson": 1,
            "extratags": 1,
            "limit": 10,
        },
        headers=UA,
        timeout=30,
    )
    r.raise_for_status()
    results = r.json()
    for res in results:
        tags = res.get("extratags") or {}
        if (
            res.get("category") == "boundary"
            and tags.get("admin_level") == "7"  # concelho em Portugal
            and res["geojson"]["type"] in ("Polygon", "MultiPolygon")
        ):
            return shape(res["geojson"]), res["display_name"]
    sys.exit(f"Não encontrei o limite do concelho '{name}'. Resultados: "
             + "; ".join(r.get("display_name", "?") for r in results))


def tile_points(tile: mercantile.Tile, content: bytes, layer: str = "image"):
    """Descodifica uma vector tile e devolve (lon, lat, props) dos pontos da camada."""
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


def fetch_index(boundary, token: str):
    poly = prep(boundary)
    minx, miny, maxx, maxy = boundary.bounds
    tiles = list(mercantile.tiles(minx, miny, maxx, maxy, ZOOM))
    print(f"A ler {len(tiles)} tiles (zoom {ZOOM})...")
    images: dict[int, dict] = {}
    session = requests.Session()
    session.headers.update(UA)
    for i, t in enumerate(tiles, 1):
        url = TILE_URL.format(z=t.z, x=t.x, y=t.y, token=token)
        for attempt in range(3):
            r = session.get(url, timeout=30)
            if r.status_code == 200:
                break
            if r.status_code in (401, 403):
                sys.exit(f"Token rejeitado ({r.status_code}). Confirma o Client Token no .env.")
            time.sleep(2 * (attempt + 1))
        else:
            print(f"  tile {t} falhou ({r.status_code}), a saltar")
            continue
        if r.content:
            for lon, lat, p in tile_points(t, r.content):
                if p.get("id") in images or not poly.contains(Point(lon, lat)):
                    continue
                images[p["id"]] = {
                    "id": p["id"],
                    "lon": round(lon, 7),
                    "lat": round(lat, 7),
                    "captured_at": p.get("captured_at"),
                    "compass_angle": p.get("compass_angle"),
                    "is_pano": bool(p.get("is_pano")),
                    "sequence_id": p.get("sequence_id"),
                    "creator_id": p.get("creator_id"),
                }
        if i % 25 == 0 or i == len(tiles):
            print(f"  {i}/{len(tiles)} tiles, {len(images)} imagens até agora")
    return list(images.values())


def sample_sizes(images: list[dict], token: str, n: int):
    """Mede o tamanho real das miniaturas 1024/2048 numa amostra aleatória."""
    sizes = {"thumb_1024_url": {"pano": [], "persp": []},
             "thumb_2048_url": {"pano": [], "persp": []}}
    sample = random.Random(42).sample(images, min(n, len(images)))
    s = requests.Session()
    s.headers.update(UA)
    for img in sample:
        r = s.get(GRAPH_URL.format(id=img["id"]),
                  params={"access_token": token,
                          "fields": "thumb_1024_url,thumb_2048_url"},
                  timeout=30)
        if r.status_code != 200:
            continue
        data = r.json()
        kind = "pano" if img["is_pano"] else "persp"
        for field in sizes:
            url = data.get(field)
            if not url:
                continue
            t = s.get(url, timeout=60)
            if t.status_code == 200:
                sizes[field][kind].append(len(t.content))
    return sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concelho", default="Aveiro")
    ap.add_argument("--sample", type=int, default=40)
    args = ap.parse_args()

    token = os.getenv("MAPILLARY_TOKEN", "").strip()
    if not token or token.startswith("MLY|COLA"):
        sys.exit("Falta o MAPILLARY_TOKEN no ficheiro .env")

    boundary, label = get_boundary(args.concelho)
    print(f"Limite: {label}")

    images = fetch_index(boundary, token)
    if not images:
        sys.exit("Nenhuma imagem encontrada.")

    META_DIR.mkdir(parents=True, exist_ok=True)
    out = META_DIR / f"{args.concelho.lower()}_index.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(images[0].keys()))
        w.writeheader()
        w.writerows(images)
    print(f"Índice guardado em {out}")

    # --- estatísticas ---
    def year(ms):
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).year if ms else None

    years = Counter(year(i["captured_at"]) for i in images)
    n_pano = sum(i["is_pano"] for i in images)
    n_seq = len({i["sequence_id"] for i in images})
    print("\n=== Cobertura Mapillary ===")
    print(f"Imagens: {len(images):,}  |  panorâmicas 360°: {n_pano:,}  |  sequências: {n_seq:,}")
    print("Por ano: " + ", ".join(f"{y}: {c:,}" for y, c in sorted(years.items(), key=lambda x: (x[0] is None, x[0]))))

    print(f"\nA medir tamanhos numa amostra de {args.sample} imagens...")
    sizes = sample_sizes(images, token, args.sample)

    now_year = datetime.now(timezone.utc).year
    recent = [i for i in images if (year(i["captured_at"]) or 0) >= now_year - 3]
    subsets = {"todas": images, f"desde {now_year - 3}": recent}

    print("\n=== Estimativa de espaço ===")
    for field, label_f in (("thumb_1024_url", "1024 px"), ("thumb_2048_url", "2048 px")):
        avg = {k: (statistics.mean(v) if v else None) for k, v in sizes[field].items()}
        fallback = next((a for a in avg.values() if a), None)
        if not fallback:
            print(f"{label_f}: sem amostra válida")
            continue
        a_pano = avg["pano"] or fallback
        a_persp = avg["persp"] or fallback
        print(f"{label_f}: média perspetiva {a_persp / 1e3:.0f} KB, panorâmica {a_pano / 1e3:.0f} KB "
              f"(n={len(sizes[field]['persp'])}/{len(sizes[field]['pano'])})")
        for name, subset in subsets.items():
            p = sum(i["is_pano"] for i in subset)
            total = p * a_pano + (len(subset) - p) * a_persp
            print(f"   {name:>12}: {len(subset):>9,} imagens  ->  {total / 1e9:6.1f} GB")


if __name__ == "__main__":
    main()
