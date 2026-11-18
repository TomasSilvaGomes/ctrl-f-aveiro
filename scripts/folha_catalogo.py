"""
Folha de imagens do catálogo: exemplos de um tipo (subtipo), cada um com a caixa a
vermelho e a foto à volta. Serve para ver se o sistema acerta (revisão pelo técnico).

Saída: data/relatorios/folhas/<subtipo>.jpg  (+ <subtipo>.csv com id do catálogo por número)

Uso:
    python scripts/folha_catalogo.py --subtipo candeeiro
    python scripts/folha_catalogo.py --subtipo subterraneo ilha_reciclagem indiferenciado --n 24
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
from estimate_mapillary import DATA_DIR  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subtipo", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(DATA_DIR / "catalogo" / "catalogo.geojson", encoding="utf-8") as f:
        feats = [ft["properties"] for ft in json.load(f)["features"]]
    files = {r["id"]: r["file"] for r in csv.DictReader(open(DATA_DIR / "meta" / "sample_images.csv", encoding="utf-8"))}
    out = DATA_DIR / "relatorios" / "folhas"
    out.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("arialbd.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    for sub in args.subtipo:
        es = [e for e in feats if e["subtipo"] == sub]
        sample = random.Random(args.seed).sample(es, min(args.n, len(es)))
        cols, size = 8, 200
        sheet = Image.new("RGB", (cols * size, ((len(sample) + cols - 1) // cols) * size), "white")
        dr = ImageDraw.Draw(sheet)
        rows = []
        for i, e in enumerate(sample):
            im = Image.open(DATA_DIR / "images" / files[e["imagem_id"]]).convert("RGB")
            x0, y0, x1, y1 = e["imagem_bbox"]
            cx, cy, s = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * 0.8 + 20
            ox, oy = int(cx - s), int(cy - s)
            v = im.crop((ox, oy, int(cx + s), int(cy + s)))
            ImageDraw.Draw(v).rectangle([x0 - ox, y0 - oy, x1 - ox, y1 - oy], outline=(255, 0, 0),
                                        width=max(2, int(s / 50)))
            x, y = (i % cols) * size, (i // cols) * size
            sheet.paste(v.resize((size - 4, size - 4)), (x + 2, y + 2))
            dr.rectangle([x + 2, y + 2, x + 34, y + 21], fill=(0, 0, 0))
            dr.text((x + 5, y + 3), str(i + 1), fill=(255, 255, 0), font=font)
            rows.append({"n": i + 1, "id": e["id"], "label": e["label"], "confianca": e["confianca"],
                         "n_vistas": e["n_vistas"], "imagem_id": e["imagem_id"]})
        sheet.save(out / f"{sub}.jpg", quality=88)
        with open(out / f"{sub}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"{sub}: {len(sample)} de {len(es)} -> {out / f'{sub}.jpg'}")


if __name__ == "__main__":
    main()
