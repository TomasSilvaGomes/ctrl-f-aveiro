"""
Avalia a SUBCLASSE das deteções finais:

  Sinais     -> anotações Mapillary por imagem; a classe Mapillary é convertida
                para o nosso subtipo através de config/taxonomia.yaml.
                Para limites de velocidade compara também o número (OCR).
  Resíduos   -> a Mapillary não tem tipo; compara com o contentor OSM mais
                próximo (<= --osm-m metros) que tenha tipo.

Uso:
    python scripts/evaluate_subtypes.py --model owlv2
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(__file__))
from estimate_mapillary import DATA_DIR  # noqa: E402
from evaluate import load_jsonl, matches  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OSM_TO_OURS = {"ecoponto": "ilha_reciclagem", "indiferenciado": "indiferenciado",
               "subterraneo": "subterraneo"}


def mapillary_to_subtype(tax):
    rules = []  # (padrão, subtipo) pela ordem da taxonomia: específico primeiro, "outro" no fim
    for sub in tax["categorias"]["sinalizacao"]["subtipos"]:
        for pat in sub.get("mapillary", []):
            rules.append((pat, sub["id"]))

    def f(value):
        for pat, sid in rules:
            if fnmatch.fnmatch(value, pat):
                return sid
        return None
    return f


def table(conf, labels, title):
    print(f"\n{title}")
    cols = labels + ["nao_detetado"]
    w = max(len(c) for c in cols) + 2
    print(f"{'referência ↓ / nosso →':>26}" + "".join(f"{c[:w - 2]:>{w}}" for c in cols) + f"{'acerto':>9}{'n':>6}")
    tot_ok = tot = 0
    for r in labels:
        row = [conf[(r, c)] for c in cols]
        det = sum(row[:-1])
        if not det and not row[-1]:
            continue
        acc = conf[(r, r)] / det if det else 0
        tot_ok += conf[(r, r)]
        tot += det
        print(f"{r:>26}" + "".join(f"{v:>{w}}" for v in row) + f"{acc:>9.0%}{det:>6}")
    if tot:
        print(f"{'TOTAL (detetados)':>26}{'':>{w * len(cols)}}{tot_ok / tot:>9.0%}{tot:>6}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="owlv2")
    ap.add_argument("--min-px", type=float, default=24)
    ap.add_argument("--osm-m", type=float, default=10)
    args = ap.parse_args()

    with open(ROOT / "config" / "taxonomia.yaml", encoding="utf-8") as f:
        tax = yaml.safe_load(f)
    to_sub = mapillary_to_subtype(tax)
    sign_ids = [s["id"] for s in tax["categorias"]["sinalizacao"]["subtipos"]]
    bin_ids = [s["id"] for s in tax["categorias"]["residuos"]["subtipos"]]

    with open(DATA_DIR / "meta" / "sample_images.csv", encoding="utf-8") as f:
        imgs = {r["id"]: r for r in csv.DictReader(f)}
    refs = load_jsonl(DATA_DIR / "ground_truth" / "mly_detections.jsonl")
    ours = load_jsonl(DATA_DIR / "preds" / f"{args.model}_classified.jsonl")

    # ---------------- sinais
    conf, speed = Counter(), Counter()
    for iid in ours:
        meta = imgs[iid]
        area = float(meta["thumb_w"]) * float(meta["thumb_h"])
        ps = [p for p in ours[iid] if p["category"] == "sinalizacao"]
        for r in refs.get(iid, []):
            if r["category"] != "sinalizacao":
                continue
            b = r["bbox"]
            if max(b[2] - b[0], b[3] - b[1]) < args.min_px:
                continue
            ref_sub = to_sub(r["value"])
            if not ref_sub:
                continue
            cand = [p for p in ps if matches({"bbox": p["bbox"]}, r, area)]
            if not cand:
                conf[(ref_sub, "nao_detetado")] += 1
                continue
            best = max(cand, key=lambda p: p["confidence"])
            conf[(ref_sub, best["subtype"])] += 1
            if ref_sub == "limite_velocidade" and best["subtype"] == "limite_velocidade":
                m = re.search(r"limit-(\d+)", r["value"])
                if m:
                    speed["certo" if best.get("valor") == int(m.group(1)) else
                          ("sem_leitura" if best.get("valor") is None else "errado")] += 1
    table(conf, sign_ids, "SINAIS — subclasse vs anotações Mapillary (sinais detetados)")
    if speed:
        n = sum(speed.values())
        print(f"\nNúmero do limite de velocidade (OCR): certo {speed['certo']}/{n} "
              f"({speed['certo'] / n:.0%}), errado {speed['errado']}, sem leitura {speed['sem_leitura']}")

    # ---------------- resíduos (OSM)
    geo_path = DATA_DIR / "preds" / f"{args.model}_geo.jsonl"
    if not geo_path.exists():
        return
    with open(DATA_DIR / "ground_truth" / "osm_aveiro.geojson", encoding="utf-8") as f:
        osm = [(ft["geometry"]["coordinates"], OSM_TO_OURS[ft["properties"]["subtype"]])
               for ft in json.load(f)["features"]
               if ft["properties"]["category"] == "residuos" and ft["properties"]["subtype"] in OSM_TO_OURS]
    kx, ky = 111320 * math.cos(math.radians(40.64)), 110540
    conf_b = Counter()
    with open(geo_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["category"] != "residuos":
                continue
            best, bd = None, args.osm_m
            for (lon, lat), sub in osm:
                dist = math.hypot((lon - d["lon"]) * kx, (lat - d["lat"]) * ky)
                if dist < bd:
                    best, bd = sub, dist
            if best:
                conf_b[(best, d["subtype"])] += 1
    table(conf_b, bin_ids, f"RESÍDUOS — subclasse vs contentor OSM mais próximo (<= {args.osm_m:.0f} m)")


if __name__ == "__main__":
    main()
