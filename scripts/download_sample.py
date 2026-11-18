"""
Descarrega uma amostra de avaliação de imagens Mapillary de Aveiro, com:
  - a imagem (miniatura 2048 px ou 1024 px)
  - os metadados de pose (posição, rumo, câmara, autor -> atribuição CC BY-SA)
  - as deteções da própria Mapillary nessa imagem (classe + caixa), que usamos
    como REFERÊNCIA para avaliar os nossos modelos (nunca entram no catálogo).

A amostra é espalhada: dentro de cada sequência fica 1 imagem em cada `--step`
(frames seguidos são quase iguais) e depois sorteia-se `--n` imagens.

Uso:
    python scripts/download_sample.py --n 1000 --since 2024
    python scripts/download_sample.py --n 1000 --since 2024 --pano   # só 360°

As imagens ficam só em disco local (data/ está no .gitignore) e não são
mostradas na aplicação. A Mapillary já desfoca rostos/matrículas; antes de a
aplicação guardar/mostrar qualquer recorte passa sempre pelo nosso EgoBlur.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import random
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import mapbox_vector_tile
import requests

sys.path.insert(0, os.path.dirname(__file__))
from estimate_mapillary import DATA_DIR, UA  # noqa: E402
from fetch_ground_truth import categorize_mly  # noqa: E402

IMG_DIR = DATA_DIR / "images"
META_CSV = DATA_DIR / "meta" / "sample_images.csv"
DET_JSONL = DATA_DIR / "ground_truth" / "mly_detections.jsonl"
GRAPH = "https://graph.mapillary.com"
IMAGE_FIELDS = ("id,captured_at,compass_angle,computed_compass_angle,geometry,"
                "computed_geometry,camera_type,camera_parameters,width,height,"
                "creator,sequence,is_pano,{thumb}")
_local = threading.local()


def session():
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers.update(UA)
    return _local.s


def select(index_csv, n, since, pano, step, seed):
    by_seq = defaultdict(list)
    with open(index_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row["is_pano"] == "True") != pano or not row["captured_at"]:
                continue
            ts = int(row["captured_at"])
            if datetime.fromtimestamp(ts / 1000, tz=timezone.utc).year < since:
                continue
            by_seq[row["sequence_id"]].append((ts, row["id"]))
    pool = []
    for frames in by_seq.values():
        frames.sort()
        pool.extend(fid for _, fid in frames[::step])
    random.Random(seed).shuffle(pool)
    print(f"Candidatas: {len(pool):,} imagens em {len(by_seq)} sequências "
          f"(1 em cada {step} frames) -> a sortear {min(n, len(pool))}")
    return pool[:n]


def decode_detection(geom_b64, width, height):
    """Geometria Mapillary (MVT em base64, extent 4096) -> bbox em píxeis."""
    tile = mapbox_vector_tile.decode(base64.b64decode(geom_b64),
                                     default_options={"y_coord_down": True})
    xs, ys = [], []
    for layer in tile.values():
        extent = layer.get("extent", 4096)
        for feat in layer["features"]:
            g = feat["geometry"]
            rings = g["coordinates"] if g["type"] == "Polygon" else \
                [r for poly in g["coordinates"] for r in poly]
            for ring in rings:
                for x, y in ring:
                    xs.append(x / extent * width)
                    ys.append(y / extent * height)
    if not xs:
        return None
    return [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)]


def fetch_one(image_id, token, thumb):
    s = session()
    r = s.get(f"{GRAPH}/{image_id}", timeout=30,
              params={"access_token": token, "fields": IMAGE_FIELDS.format(thumb=thumb)})
    r.raise_for_status()
    m = r.json()
    url = m.get(thumb)
    if not url:
        raise RuntimeError("sem URL de miniatura")

    path = IMG_DIR / f"{image_id}.jpg"
    if not path.exists():
        img = s.get(url, timeout=60)
        img.raise_for_status()
        path.write_bytes(img.content)

    # dimensões da miniatura (a Mapillary devolve as do original)
    from PIL import Image
    with Image.open(path) as im:
        tw, th = im.size

    d = s.get(f"{GRAPH}/{image_id}/detections", timeout=30,
              params={"access_token": token, "fields": "id,value,geometry"})
    dets = []
    if d.status_code == 200:
        for det in d.json().get("data", []):
            cat = categorize_mly(det.get("value", ""))
            if not cat:
                continue
            bbox = decode_detection(det["geometry"], tw, th) if det.get("geometry") else None
            if bbox:
                dets.append({"value": det["value"], "category": cat[0],
                             "subtype": cat[1], "bbox": bbox})

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--since", type=int, default=2024)
    ap.add_argument("--pano", action="store_true", help="só panorâmicas 360°")
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--size", choices=["1024", "2048", "original"], default="2048")
    ap.add_argument("--ids-from", help="CSV com coluna 'id' (ex.: data/meta/urban_candidates.csv)")
    ap.add_argument("--only-pano", choices=["yes", "no"], help="filtra o --ids-from por tipo")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    token = os.getenv("MAPILLARY_TOKEN", "").strip()
    if not token or "COLA_AQUI" in token:
        sys.exit("Falta o MAPILLARY_TOKEN no ficheiro .env")

    if args.ids_from:
        with open(args.ids_from, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if args.only_pano:
            rows = [r for r in rows if (r.get("is_pano") == "True") == (args.only_pano == "yes")]
        ids = [r["id"] for r in rows][: args.n]
        print(f"{len(ids)} imagens de {args.ids_from}")
    else:
        ids = select(DATA_DIR / "meta" / "aveiro_index.csv",
                     args.n, args.since, args.pano, args.step, args.seed)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    DET_JSONL.parent.mkdir(parents=True, exist_ok=True)

    metas, n_det, errors = [], 0, 0
    thumb = f"thumb_{args.size}_url"
    with ThreadPoolExecutor(args.workers) as ex, DET_JSONL.open("w", encoding="utf-8") as fdet:
        futs = {ex.submit(fetch_one, i, token, thumb): i for i in ids}
        for k, fut in enumerate(as_completed(futs), 1):
            try:
                meta, dets = fut.result()
            except Exception as e:  # noqa: BLE001
                errors += 1
                print(f"  erro {futs[fut]}: {e}")
                continue
            metas.append(meta)
            for det in dets:
                fdet.write(json.dumps({"image_id": meta["id"], **det}) + "\n")
            n_det += len(dets)
            if k % 100 == 0 or k == len(ids):
                print(f"  {k}/{len(ids)} imagens, {n_det} deteções de referência")

    with META_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(metas[0].keys()))
        w.writeheader()
        w.writerows(metas)

    size_mb = sum((IMG_DIR / m["file"]).stat().st_size for m in metas) / 1e6
    print(f"\nImagens: {len(metas)} ({size_mb:.0f} MB) em {IMG_DIR}")
    print(f"Metadados: {META_CSV}")
    print(f"Deteções de referência Mapillary: {n_det} em {DET_JSONL}")
    if errors:
        print(f"Erros: {errors}")


if __name__ == "__main__":
    main()
