"""
Escolhe TODAS as fotos seguidas (frames consecutivos) das sequências que passam
numa rua, para a triangulação multivista (ver triangulacao.py).

A rua é dada por 2 pontos (início e direção) e continua em frente até --comprimento
metros. Ficam as fotos a <= --largura metros do eixo da rua, das sequências com
pelo menos --min-fotos fotos (por omissão só 360°).

Prepara uma pasta de dados própria para o piloto (não mexe na amostra principal):
  <saida>/meta/rua_ids.csv           ids a descarregar (download_sample.py --ids-from)
  <saida>/ground_truth/*.geojson      cópia das referências Mapillary/OSM (só para medir)

Uso (PowerShell):
    python scripts/seleciona_rua.py --inicio 40.643331,-8.645127 --direcao 40.643185,-8.645803
    $env:DATA_DIR = "data\\piloto_rua"
    python scripts/download_sample.py --ids-from data\\piloto_rua\\meta\\rua_ids.csv --n 100000 --size original
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN_DATA = ROOT / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inicio", required=True, help="lat,lon do início da rua")
    ap.add_argument("--direcao", required=True, help="lat,lon de um ponto mais à frente na rua")
    ap.add_argument("--comprimento", type=float, default=400, help="metros a partir do início")
    ap.add_argument("--largura", type=float, default=12, help="distância máxima ao eixo (m)")
    ap.add_argument("--min-fotos", type=int, default=30, help="sequências com menos fotos ficam de fora")
    ap.add_argument("--todas", action="store_true", help="incluir fotos normais (não 360°)")
    ap.add_argument("--saida", default=str(MAIN_DATA / "piloto_rua"))
    args = ap.parse_args()

    a = [float(v) for v in args.inicio.split(",")]
    b = [float(v) for v in args.direcao.split(",")]
    kx, ky = 111320 * math.cos(math.radians(a[0])), 110540

    def xy(lat, lon):
        return (lon - a[1]) * kx, (lat - a[0]) * ky

    bx, by = xy(*b)
    n = math.hypot(bx, by)
    ux, uy = bx / n, by / n
    print(f"Rua: rumo {math.degrees(math.atan2(ux, uy)) % 360:.0f}°, {args.comprimento:.0f} m")

    seqs = defaultdict(list)
    with open(MAIN_DATA / "meta" / "aveiro_index.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not args.todas and r["is_pano"] != "True":
                continue
            x, y = xy(float(r["lat"]), float(r["lon"]))
            t, d = x * ux + y * uy, abs(-x * uy + y * ux)  # ao longo / ao lado da rua
            if -20 <= t <= args.comprimento and d <= args.largura:
                seqs[r["sequence_id"]].append((t, r))

    out = Path(args.saida)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"{'sequência':<24}{'fotos':>6}{'espaço':>9}  data")
    for s, v in sorted(seqs.items(), key=lambda kv: -len(kv[1])):
        if len(v) < args.min_fotos:
            continue
        v.sort(key=lambda z: z[0])
        gaps = sorted(q - p for (p, _), (q, _) in zip(v, v[1:]))
        date = datetime.fromtimestamp(int(v[0][1]["captured_at"]) / 1000, timezone.utc).date()
        print(f"{s:<24}{len(v):>6}{gaps[len(gaps) // 2]:>7.1f} m  {date}")
        rows += [{"id": r["id"], "sequence_id": s, "ao_longo_m": round(t, 1), "is_pano": r["is_pano"]}
                 for t, r in v]
    if not rows:
        sys.exit("Nenhuma sequência encontrada nesta rua.")
    with open(out / "meta" / "rua_ids.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # referências (só para medir) e desenhos de sinais: cópia da pasta principal
    (out / "ground_truth").mkdir(exist_ok=True)
    for name in ("osm_aveiro.geojson", "mapillary_features_aveiro.geojson"):
        shutil.copyfile(MAIN_DATA / "ground_truth" / name, out / "ground_truth" / name)
    print(f"\n{len(rows)} fotos -> {out / 'meta' / 'rua_ids.csv'}")


if __name__ == "__main__":
    main()
