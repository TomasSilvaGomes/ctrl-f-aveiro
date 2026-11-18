"""
Desenha caixas sobre as fotos, com a atribuição CC BY-SA no rodapé. Três modos:

  final   resultado FINAL: tipo · confiança · distância   (data/preds/owlv2_geo.jsonl)
          -> data/viz/<image_id>.jpg
  owl     OWLv2 EM BRUTO: a frase do detetor que encontrou cada caixa + pontuação, antes do
          SigLIP (a frase só diz a categoria; o tipo é o SigLIP que decide)
          -> data/viz/owl_<image_id>.jpg
  foto    corre a cadeia OWLv2 -> SigLIP numa foto QUALQUER (frontal; ex.: para comparar com
          outro modelo). Tracejado cinzento = descartado pelo SigLIP (ganhou frase negativa)
          -> data/viz/teste_<nome>.jpg

Sem --ids escolhe as fotos com mais categorias (--frontal / --pano para filtrar).

Uso:
    python scripts/visualize.py final --n 12
    python scripts/visualize.py owl --ids 1866069988084542 --min-score 0.15
    python scripts/visualize.py foto minha_foto.jpg outra.jpg
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
from comum import COLORS, DATA_DIR, font, load_jsonl, load_taxonomy, nms_by_category  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
NOMES = {"sinalizacao": "Sinal", "iluminacao": "Iluminação", "residuos": "Resíduos"}
OUT_DIR = DATA_DIR / "viz"


# ------------------------------------------------------------------ desenho
def label(dr, f, fs, box, text, color, lw, dashed=False):
    x0, y0, x1, y1 = box
    if dashed:
        dash = 12
        for ax, ay, bx, by in [(x0, y0, x1, y0), (x0, y1, x1, y1), (x0, y0, x0, y1), (x1, y0, x1, y1)]:
            n = max(1, int(max(abs(bx - ax), abs(by - ay)) // dash))
            for i in range(0, n, 2):
                t0, t1 = i / n, min(1, (i + 1) / n)
                dr.line([ax + (bx - ax) * t0, ay + (by - ay) * t0, ax + (bx - ax) * t1, ay + (by - ay) * t1],
                        fill=color, width=lw)
    else:
        dr.rectangle([x0, y0, x1, y1], outline=color, width=lw)
    tw = dr.textlength(text, font=f)
    ty = max(0, y0 - fs - 6)
    dr.rectangle([x0, ty, x0 + tw + 8, ty + fs + 6], fill=color)
    dr.text((x0 + 4, ty + 2), text, fill=(0, 0, 0), font=f)


def banner(dr, f, fs, y, text, color, fg=(0, 0, 0)):
    dr.rectangle([0, y, dr.textlength(text, font=f) + 12, y + fs + 8], fill=color)
    dr.text((6, y + 3), text, fill=fg, font=f)
    return y + fs + 10


def open_band(meta):
    """Foto da amostra; nas 360° corta o céu e o carro (fica a faixa do meio)."""
    im = Image.open(DATA_DIR / "images" / meta["file"]).convert("RGB")
    W, H = im.size
    top, bottom = (int(H * 0.22), int(H * 0.68)) if meta["is_pano"] == "True" else (0, H)
    return im.crop((0, top, W, bottom)), top


def attribution(im, dr, f, fs, meta):
    t = f"Imagem: {meta.get('creator_username') or '?'} / Mapillary, CC BY-SA 4.0 — {meta['source_url']}"
    dr.rectangle([0, im.height - fs - 10, dr.textlength(t, font=f) + 12, im.height], fill=(0, 0, 0))
    dr.text((6, im.height - fs - 6), t, fill=(255, 255, 255), font=f)


def save(im, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    im.thumbnail((3000, 3000))
    path = OUT_DIR / name
    im.save(path, quality=88)
    return path


def pick(by_img, imgs, args, good):
    """--ids, ou as fotos com mais categorias (e mais tipos) acima do limiar."""
    if args.ids:
        return args.ids.split(",")

    def ok(i):
        p = imgs[i]["is_pano"] == "True"
        return not ((args.frontal and p) or (args.pano and not p))

    def score(i):
        g = good(by_img[i])
        return (len({d["category"] for d in g}), len({d.get("subtype") or d.get("prompt") for d in g}))
    return sorted((i for i in by_img if i in imgs and ok(i)), key=score, reverse=True)[: args.n]


def load_imgs():
    with open(DATA_DIR / "meta" / "sample_images.csv", encoding="utf-8") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


# ------------------------------------------------------------------ modos
def cmd_final(args):
    imgs, by_img = load_imgs(), load_jsonl(DATA_DIR / "preds" / "owlv2_geo.jsonl")
    good = lambda ds: [d for d in ds if d["confidence"] >= args.min_conf]  # noqa: E731
    for iid in pick(by_img, imgs, args, good):
        im, top = open_band(imgs[iid])
        dr = ImageDraw.Draw(im)
        fs, lw = max(14, im.width // 160), max(2, im.width // 900)
        f = font(fs)
        for d in sorted(good(by_img[iid]), key=lambda d: d["confidence"]):
            x0, y0, x1, y1 = d["bbox"]
            # só o nome (o código RST/OSM fica nos dados exportados)
            label(dr, f, fs, [x0, y0 - top, x1, y1 - top],
                  f"{d['label']} · {d['confidence']:.0%} · {d['dist_m']:.0f} m", COLORS[d["category"]], lw)
        attribution(im, dr, f, fs, imgs[iid])
        print(save(im, f"{iid}.jpg"))


def cmd_owl(args):
    prompts = load_taxonomy()["detetor"]
    imgs, by_img = load_imgs(), load_jsonl(DATA_DIR / "preds" / "owlv2.jsonl")
    good = lambda ds: nms_by_category([d for d in ds if d["score"] >= args.min_score])  # noqa: E731
    for iid in pick(by_img, imgs, args, good):
        im, top = open_band(imgs[iid])
        dr = ImageDraw.Draw(im)
        fs, lw = max(14, im.width // 150), max(2, im.width // 800)
        f = font(fs)
        shown = good(by_img[iid])
        for d in sorted(shown, key=lambda d: d["score"]):
            x0, y0, x1, y1 = d["bbox"]
            label(dr, f, fs, [x0, y0 - top, x1, y1 - top], f'"{d["prompt"]}" {d["score"]:.2f}',
                  COLORS[d["category"]], lw)
        y = 6
        for cat, ps in prompts.items():   # legenda: frases usadas por categoria
            y = banner(dr, f, fs, y, f"{NOMES.get(cat, cat)}: " + " · ".join(ps), COLORS[cat])
        banner(dr, f, fs, y, f"OWLv2 em bruto · pontuação ≥ {args.min_score} · {len(shown)} caixas",
               (0, 0, 0), (255, 255, 255))
        attribution(im, dr, f, fs, imgs[iid])
        print(save(im, f"owl_{iid}.jpg"))
        for d in sorted(shown, key=lambda d: -d["score"]):
            print(f'   {d["score"]:.2f}  {NOMES[d["category"]]:<10}  "{d["prompt"]}"')


def cmd_foto(args):
    import numpy as np
    import torch
    from classify_subtype import Classifier, square_crop, suppress_contained
    from run_detector import PROMPTS, TILE_W, Owlv2, letterbox, to_original

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tax = load_taxonomy()
    labels = {s["id"]: s["label"].replace(" {n} km/h", "") for c in tax["categorias"].values() for s in c["subtipos"]}
    det, clf = Owlv2(device), Classifier(tax, device)
    for path in args.fotos:
        im = Image.open(path).convert("RGB")
        arr = np.asarray(im)
        H, W = arr.shape[:2]
        tw = int(W * TILE_W)
        views = []
        for a, b in [(0, W), (0, tw), (W - tw, W)]:   # mesmo tiling das frontais
            canvas, s, ox = letterbox(arr[:, a:b], a)
            views.append((canvas, {"pano": False, "W": W, "H": H, "scale": s, "ox": ox}))
        dets = []
        for (_, info), res in zip(views, det([v for v, _ in views], 0.1)):
            for box, score, prompt in res:
                ob = to_original(box, info)
                if ob and score >= args.det_threshold:
                    dets.append({"bbox": ob, "score": score, "prompt": prompt, "category": PROMPTS[prompt][0]})
        # mesmos filtros da corrida principal
        dets = [d for d in nms_by_category(dets)
                if max(d["bbox"][2] - d["bbox"][0], d["bbox"][3] - d["bbox"][1]) >= 16]
        dets = suppress_contained(dets)
        res = clf([square_crop(im, d["bbox"])[0] for d in dets], [d["category"] for d in dets]) if dets else []
        dr = ImageDraw.Draw(im)
        fs, lw = max(14, W // 110), max(2, W // 700)
        f = font(fs)
        print(Path(path).name)
        for d, (sub, p_sub, p_neg) in sorted(zip(dets, res), key=lambda x: x[0]["score"]):
            owl = f'OWL: "{d["prompt"]}" {d["score"]:.2f}'
            if sub is None:
                t = f"descartado ({p_neg:.0%} negativo) | {owl}"
                label(dr, f, fs, d["bbox"], t, (150, 150, 150), lw, dashed=True)
            else:
                t = f"{labels[sub]} {p_sub:.0%} | {owl}"
                label(dr, f, fs, d["bbox"], t, COLORS[d["category"]], lw)
            print(f"   {t}")
        print(f"-> {save(im, f'teste_{Path(path).stem}.jpg')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    for name, extra in (("final", ("--min-conf", 0.3)), ("owl", ("--min-score", 0.2))):
        p = sub.add_parser(name)
        p.add_argument("--ids", default="", help="ids separados por vírgula")
        p.add_argument("--n", type=int, default=12)
        p.add_argument(extra[0], type=float, default=extra[1])
        p.add_argument("--frontal", action="store_true", help="só fotos frontais")
        p.add_argument("--pano", action="store_true", help="só fotos 360°")
    p = sub.add_parser("foto")
    p.add_argument("fotos", nargs="+")
    p.add_argument("--det-threshold", type=float, default=0.2)
    argv = sys.argv[1:]
    if not argv or argv[0] not in ("final", "owl", "foto", "-h", "--help"):
        argv = ["final", *argv]   # compatível com o uso antigo: visualize.py --n 12
    args = ap.parse_args(argv)
    {"final": cmd_final, "owl": cmd_owl, "foto": cmd_foto}[args.cmd](args)


if __name__ == "__main__":
    main()
