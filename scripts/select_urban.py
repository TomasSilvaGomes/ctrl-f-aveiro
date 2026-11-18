"""
Seleciona imagens URBANAS onde, à partida, aparecem as 3 categorias do desafio
(sinalização, iluminação, resíduos), usando só o índice e os pontos de
referência já descarregados (sem gastar pedidos de imagem).

Critérios por imagem:
  - perspetiva (não 360°) e capturada desde --since
  - a mais de --motorway-m metros de autoestradas/vias rápidas (OSM)
  - dentro de --radius metros E à frente da câmara (±--fov/2 do rumo) há pelo
    menos 1 contentor/ecoponto, 1 sinal e 1 candeeiro (Mapillary + OSM)
  - desbaste: dentro da mesma sequência, imagens a mais de --spacing metros

Saída: data/meta/urban_candidates.csv (ordenado: mais objetos primeiro)

Uso:
    python scripts/select_urban.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

import requests
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

sys.path.insert(0, os.path.dirname(__file__))
from estimate_mapillary import DATA_DIR, UA  # noqa: E402

LAT0 = 40.64  # projeção local equiretangular (erro desprezável à escala do concelho)
KX, KY = 111320 * math.cos(math.radians(LAT0)), 110540


def xy(lon, lat):
    return lon * KX, lat * KY


def motorways():
    q = """
    [out:json][timeout:120];
    area["boundary"="administrative"]["admin_level"="7"]["name"="Aveiro"]->.a;
    way["highway"~"^(motorway|motorway_link|trunk|trunk_link)$"](area.a);
    out geom;
    """
    r = requests.post("https://overpass-api.de/api/interpreter", data={"data": q},
                      headers=UA, timeout=180)
    r.raise_for_status()
    return [LineString([xy(p["lon"], p["lat"]) for p in w["geometry"]])
            for w in r.json()["elements"] if len(w.get("geometry", [])) > 1]


def reference_points():
    pts = []  # (x, y, categoria, subtipo, fonte)
    for name in ("mapillary_features_aveiro.geojson", "osm_aveiro.geojson"):
        with open(DATA_DIR / "ground_truth" / name, encoding="utf-8") as f:
            for feat in json.load(f)["features"]:
                p = feat["properties"]
                if p["subtype"] in ("poste", "poste_eletrico"):  # postes de eletricidade não contam
                    continue
                lon, lat = feat["geometry"]["coordinates"]
                pts.append((*xy(lon, lat), p["category"], p["subtype"], p["source"]))
    return pts


def angle_diff(a, b):
    return abs((a - b + 180) % 360 - 180)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=int, default=2019)
    ap.add_argument("--bbox", default="-8.680,40.615,-8.615,40.660",
                    help="W,S,E,N — por omissão a cidade de Aveiro")
    ap.add_argument("--radius", type=float, default=30)
    ap.add_argument("--fov", type=float, default=90)
    ap.add_argument("--motorway-m", type=float, default=40)
    ap.add_argument("--spacing", type=float, default=15)
    args = ap.parse_args()

    print("Autoestradas/vias rápidas (OSM)...")
    mw = motorways()
    mw_tree = STRtree(mw)
    print(f"  {len(mw)} troços")

    refs = reference_points()
    ref_tree = STRtree([Point(x, y) for x, y, *_ in refs])
    print(f"Pontos de referência: {len(refs):,}")

    W, S, E, N = map(float, args.bbox.split(","))
    cands, stats = [], defaultdict(int)
    with open(DATA_DIR / "meta" / "aveiro_index.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stats["total"] += 1
            if not row["captured_at"] or not row["compass_angle"]:
                continue
            if not (W <= float(row["lon"]) <= E and S <= float(row["lat"]) <= N):
                continue
            year = datetime.fromtimestamp(int(row["captured_at"]) / 1000, tz=timezone.utc).year
            if year < args.since:
                continue
            is_pano = row["is_pano"] == "True"
            stats["recentes_cidade"] += 1
            x, y = xy(float(row["lon"]), float(row["lat"]))
            cam = Point(x, y)
            near_mw = mw_tree.query(cam.buffer(args.motorway_m))
            if any(mw[i].distance(cam) < args.motorway_m for i in near_mw):
                stats["autoestrada"] += 1
                continue
            heading = float(row["compass_angle"])
            counts = defaultdict(int)
            subs = set()
            for i in ref_tree.query(cam.buffer(args.radius)):
                rx, ry, cat, sub, src = refs[i]
                d = math.hypot(rx - x, ry - y)
                if d > args.radius or d < 2:
                    continue
                bearing = math.degrees(math.atan2(rx - x, ry - y)) % 360
                # panorâmica 360° vê em todas as direções
                if is_pano or angle_diff(bearing, heading) <= args.fov / 2:
                    counts[cat] += 1
                    if cat == "residuos" and src == "osm":
                        subs.add(sub)
            if all(counts[c] for c in ("sinalizacao", "iluminacao", "residuos")):
                stats["com_3_categorias"] += 1
                stats[f"com_3_{'360' if is_pano else 'normal'}"] += 1
                cands.append({**row, "x": x, "y": y, "year": year,
                              "n_sinal": counts["sinalizacao"], "n_ilum": counts["iluminacao"],
                              "n_resid": counts["residuos"],
                              "subtipos_osm": "|".join(sorted(subs))})

    # desbaste por sequência (frames seguidos são quase iguais)
    cands.sort(key=lambda c: -(c["n_resid"] * 3 + c["n_sinal"] + c["n_ilum"]))
    kept, by_seq = [], defaultdict(list)
    for c in cands:
        if all(math.hypot(c["x"] - k["x"], c["y"] - k["y"]) >= args.spacing
               for k in by_seq[c["sequence_id"]]):
            by_seq[c["sequence_id"]].append(c)
            kept.append(c)

    out = DATA_DIR / "meta" / "urban_candidates.csv"
    cols = ["id", "lon", "lat", "captured_at", "year", "is_pano", "compass_angle", "sequence_id",
            "n_sinal", "n_ilum", "n_resid", "subtipos_osm"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(kept)

    print("\n=== Seleção ===")
    print(f"Índice total:                 {stats['total']:>8,}")
    print(f"Cidade, desde {args.since}:          {stats['recentes_cidade']:>8,}")
    print(f"  excluídas (autoestrada):    {stats['autoestrada']:>8,}")
    print(f"  com as 3 categorias à vista:{stats['com_3_categorias']:>8,}"
          f"  (360°: {stats['com_3_360']:,} | normais: {stats['com_3_normal']:,})")
    print(f"  após desbaste ({args.spacing:.0f} m):       {len(kept):>8,}  em {len(by_seq)} sequências")
    yp = defaultdict(int)
    for k in kept:
        yp[(k["year"], "360" if k["is_pano"] == "True" else "normal")] += 1
    print("  por ano/tipo: " + ", ".join(f"{y} {t}: {n:,}" for (y, t), n in sorted(yp.items())))
    with_sub = sum(1 for k in kept if k["subtipos_osm"])
    print(f"  com subtipo de resíduo (OSM): {with_sub:>6,}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
