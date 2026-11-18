"""
Desenha as deteções finais (classe/subclasse, confiança, distância) sobre as
imagens, com a atribuição CC BY-SA no rodapé.

Entrada:  data/preds/<modelo>_geo.jsonl
Saída:    data/viz/<image_id>.jpg

Uso:
    python scripts/visualize.py --model owlv2 --n 12            # imagens com mais categorias
    python scripts/visualize.py --model owlv2 --ids 123,456
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
from estimate_mapillary import DATA_DIR  # noqa: E402

COLORS = {"sinalizacao": (230, 40, 40), "iluminacao": (255, 200, 0), "residuos": (40, 200, 70)}
Image.MAX_IMAGE_PIXELS = None


def font(size):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_label(d):
    # Só o nome (o código RST/OSM fica nos dados exportados, não no que o utilizador vê)
    return f"{d['label']} · {d['confidence']:.0%} · {d['dist_m']:.0f} m"


def render(iid, dets, meta, out_dir, min_conf):
    im = Image.open(DATA_DIR / "images" / f"{iid}.jpg").convert("RGB")
    W, H = im.size
    pano = meta["is_pano"] == "True"
    top, bottom = (int(H * 0.22), int(H * 0.68)) if pano else (0, H)
    im = im.crop((0, top, W, bottom))
    dr = ImageDraw.Draw(im)
    fs = max(14, W // 160)
    f = font(fs)
    lw = max(2, W // 900)
    for d in sorted(dets, key=lambda d: d["confidence"]):
        if d["confidence"] < min_conf:
            continue
        x0, y0, x1, y1 = d["bbox"]
        y0, y1 = y0 - top, y1 - top
        c = COLORS[d["category"]]
        dr.rectangle([x0, y0, x1, y1], outline=c, width=lw)
        t = text_label(d)
        tw = dr.textlength(t, font=f)
        ty = max(0, y0 - fs - 6)
        dr.rectangle([x0, ty, x0 + tw + 8, ty + fs + 6], fill=c)
        dr.text((x0 + 4, ty + 2), t, fill=(0, 0, 0), font=f)
    attr = f"Imagem: {meta.get('creator_username') or '?'} / Mapillary, CC BY-SA 4.0 — {meta['source_url']}"
    dr.rectangle([0, im.height - fs - 10, dr.textlength(attr, font=f) + 12, im.height], fill=(0, 0, 0))
    dr.text((6, im.height - fs - 6), attr, fill=(255, 255, 255), font=f)
    im.thumbnail((3000, 3000))
    path = out_dir / f"{iid}.jpg"
    im.save(path, quality=88)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="owlv2")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--ids", default="")
    ap.add_argument("--min-conf", type=float, default=0.3)
    args = ap.parse_args()

    with open(DATA_DIR / "meta" / "sample_images.csv", encoding="utf-8") as f:
        imgs = {r["id"]: r for r in csv.DictReader(f)}
    by_img = defaultdict(list)
    with open(DATA_DIR / "preds" / f"{args.model}_geo.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            by_img[d["image_id"]].append(d)

    if args.ids:
        ids = args.ids.split(",")
    else:
        def score(i):
            good = [d for d in by_img[i] if d["confidence"] >= args.min_conf]
            return (len({d["category"] for d in good}), len({d["subtype"] for d in good}))
        ids = sorted(by_img, key=score, reverse=True)[: args.n]

    out_dir = DATA_DIR / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    for iid in ids:
        print(render(iid, by_img[iid], imgs[iid], out_dir, args.min_conf))


if __name__ == "__main__":
    main()
