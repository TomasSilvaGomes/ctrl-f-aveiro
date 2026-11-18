"""
Auditoria visual da CLASSIFICAÇÃO (SigLIP): compara as frases atuais da taxonomia com as
frases escolhidas pela fase 1 do escolhe_prompts.py (CCAS), em recortes reais da rua.

A Mapillary não diz o tipo (candeeiro/poste, ilha/indiferenciado/subterrâneo), por isso
a "verdade" é dada por uma pessoa a ver as imagens.

  1. preparar: tira --n recortes ao acaso por categoria (deteções do sistema), classifica
     com os 2 conjuntos de frases e cria:
       data/relatorios/auditoria_classificacao/folha_<categoria>.jpg  (recortes numerados)
       data/relatorios/auditoria_classificacao/auditoria.csv          (coluna "verdade" vazia)
  2. (pessoa) preencher "verdade" no CSV com o id da classe certa, ou "nao" se não for
     mobiliário da categoria (ex.: papeleira, poste elétrico, armário).
  3. medir: acerto de cada conjunto de frases.

Uso:
    python scripts/audita_classificacao.py preparar --pastas data/piloto_rua data/piloto_sequencia
    python scripts/audita_classificacao.py medir
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
from classify_subtype import Classifier, load_taxonomy, square_crop  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "relatorios" / "auditoria_classificacao"
CATS = ("iluminacao", "residuos")
Image.MAX_IMAGE_PIXELS = None


def tax_ccas(tax):
    chosen = json.loads((ROOT / "data" / "relatorios" / "escolha_prompts.json").read_text(encoding="utf-8"))
    t = copy.deepcopy(tax)
    for cat, classes in chosen["classificador"].items():
        for s in t["categorias"][cat]["subtipos"]:
            if s["id"] in classes:
                s["prompts"] = classes[s["id"]]
    return t


def preparar(args):
    rng = random.Random(args.seed)
    pool = {c: [] for c in CATS}
    for base in args.pastas:
        base = Path(base)
        files = {r["id"]: r["file"] for r in csv.DictReader(open(base / "meta" / "sample_images.csv", encoding="utf-8"))}
        for line in open(base / "preds" / "owlv2.jsonl", encoding="utf-8"):
            d = json.loads(line)
            b = d["bbox"]
            if d["category"] in CATS and d["score"] >= 0.2 and max(b[2] - b[0], b[3] - b[1]) >= 32:
                pool[d["category"]].append((base, files[d["image_id"]], d))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tax = load_taxonomy()
    clfs = {"atual": Classifier(tax, device), "ccas": Classifier(tax_ccas(tax), device)}
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("arialbd.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    rows = []
    for cat in CATS:
        sample = rng.sample(pool[cat], min(args.n, len(pool[cat])))
        crops, views = [], []
        for base, fname, d in sample:
            im = Image.open(base / "images" / fname).convert("RGB")
            crops.append(square_crop(im, d["bbox"])[0])           # o que o modelo vê
            x0, y0, x1, y1 = d["bbox"]                             # o que a pessoa vê: foto à volta
            cx, cy, s = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * 0.65
            v = im.crop((int(cx - s), int(cy - s), int(cx + s), int(cy + s)))
            ox, oy = int(cx - s), int(cy - s)
            ImageDraw.Draw(v).rectangle([x0 - ox, y0 - oy, x1 - ox, y1 - oy], outline=(255, 0, 0),
                                        width=max(2, int(s / 60)))
            views.append(v)
        preds = {k: clf(crops, [cat] * len(crops)) for k, clf in clfs.items()}
        cols = 8
        sheet = Image.new("RGB", (cols * 200, ((len(crops) + cols - 1) // cols) * 200), "white")
        dr = ImageDraw.Draw(sheet)
        for i, ((base, fname, d), view) in enumerate(zip(sample, views)):
            x, y = (i % cols) * 200, (i // cols) * 200
            sheet.paste(view.resize((196, 196)), (x + 2, y + 2))
            dr.rectangle([x + 2, y + 2, x + 34, y + 22], fill=(0, 0, 0))
            dr.text((x + 5, y + 3), str(i + 1), fill=(255, 255, 0), font=font)
            rows.append({"categoria": cat, "n": i + 1, "pasta": base.name, "image_id": d["image_id"],
                         "bbox": json.dumps([round(v) for v in d["bbox"]]),
                         "atual": preds["atual"][i][0] or "nao", "ccas": preds["ccas"][i][0] or "nao",
                         "verdade": ""})
        sheet.save(OUT / f"folha_{cat}.jpg", quality=90)
        print(f"{cat}: {len(crops)} recortes -> {OUT / f'folha_{cat}.jpg'}")
    with open(OUT / "auditoria.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {OUT / 'auditoria.csv'}  (preencher a coluna 'verdade')")


def medir(args):
    rows = [r for r in csv.DictReader(open(OUT / "auditoria.csv", encoding="utf-8-sig")) if r["verdade"].strip()]
    lines = ["# Auditoria da classificação (SigLIP): frases atuais vs CCAS", ""]
    for cat in CATS:
        rs = [r for r in rows if r["categoria"] == cat]
        if not rs:
            continue
        verd = Counter(r["verdade"] for r in rs)
        print(f"\n{cat}: {len(rs)} recortes · verdade: {dict(verd)}")
        lines += [f"## {cat} ({len(rs)} recortes)", "",
                  "Verdade (a olho): " + ", ".join(f"{k} {v}" for k, v in verd.most_common()), "",
                  "| Frases | Tipo certo | Mobiliário verdadeiro descartado | Não-mobiliário aceite |",
                  "|---|---|---|---|"]
        for k in ("atual", "ccas"):
            ok = sum(r[k] == r["verdade"] for r in rs)
            lost = sum(r[k] == "nao" and r["verdade"] != "nao" for r in rs)
            fp = sum(r[k] != "nao" and r["verdade"] == "nao" for r in rs)
            print(f"  {k:<6} certo {ok}/{len(rs)} ({ok / len(rs):.0%}) · descartou verdadeiros {lost} · aceitou falsos {fp}")
            lines.append(f"| {k} | {ok}/{len(rs)} ({ok / len(rs):.0%}) | {lost} | {fp} |")
        lines += ["", "*Tipo certo = o sistema deu a mesma resposta que a pessoa (incluindo 'não é mobiliário').*", ""]
    (OUT / "auditoria.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n-> {OUT / 'auditoria.md'}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="modo", required=True)
    p = sub.add_parser("preparar")
    p.add_argument("--pastas", nargs="+", default=["data/piloto_rua", "data/piloto_sequencia"])
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    sub.add_parser("medir")
    args = ap.parse_args()
    preparar(args) if args.modo == "preparar" else medir(args)


if __name__ == "__main__":
    main()
