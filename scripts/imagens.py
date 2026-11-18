"""
IMAGENS para ver os resultados:

  final     fotos com caixas: nome · confiança (--todas: rua inteira por ordem + index.html)
  owl       detetor em bruto (frase + pontuação), antes do SigLIP
  foto      corre detetor + SigLIP numa foto qualquer
  contorno  caixa ao lado do contorno SAM 2 (pé real e pontos 3D usados)
  folha     folha de recortes numerados de um tipo do catálogo (para rever à mão)
  guardar   guarda recortes escolhidos da folha como referências de Aveiro

Uso:
    python scripts/imagens.py final --todas
    python scripts/imagens.py contorno --n 9
    python scripts/imagens.py folha --subtipo candeeiro
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(__file__))
from comum import COLORS, DATA_DIR, ROOT, font, ink, limiares, load_jsonl, load_taxonomy, nms_by_category  # noqa: E402


####################################################################################################
# VER  (antes: scripts/visualize.py)
####################################################################################################
# Desenha caixas sobre as fotos, com a atribuição CC BY-SA no rodapé. Três modos:
#
#   final   resultado FINAL: tudo o que foi visto, nome · confiança (owlv2_classified;
#           --min-conf filtra; a distância fica só nos dados exportados)
#           -> data/viz/<image_id>.jpg  (--todas: rua inteira por ordem + index.html)
#   owl     OWLv2 EM BRUTO: a frase do detetor que encontrou cada caixa + pontuação, antes do
#           SigLIP (a frase só diz a categoria; o tipo é o SigLIP que decide)
#           -> data/viz/owl_<image_id>.jpg
#   foto    corre a cadeia OWLv2 -> SigLIP numa foto QUALQUER (frontal; ex.: para comparar com
#           outro modelo). Tracejado cinzento = descartado pelo SigLIP (ganhou frase negativa)
#           -> data/viz/teste_<nome>.jpg
#   contorno  caixa (OWLv2) ao lado do contorno (SAM 2): linha vermelha = pé pela caixa,
#           ponto verde = pé real, pontos azuis = pontos 3D da Mapillary usados na distância
#           -> data/viz/contorno/NN_<image_id>.jpg + index.html + contorno.json
#
# Sem --ids escolhe as fotos com mais categorias (--frontal / --pano para filtrar).
#
# Uso:
#     python scripts/imagens.py final --n 12
#     python scripts/imagens.py owl --ids 1866069988084542 --min-score 0.15
#     python scripts/imagens.py foto minha_foto.jpg outra.jpg


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
    dr.text((x0 + 4, ty + 2), text, fill=ink(color), font=f)


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
def gallery(names, title):
    """Página simples (data/viz/index.html) para ver as fotos por ordem, a fazer scroll."""
    rows = "\n".join(f'<figure><img loading="lazy" src="{n}"><figcaption>{i}. {n}</figcaption></figure>'
                     for i, n in enumerate(names, 1))
    (OUT_DIR / "index.html").write_text(
        f"<!doctype html><meta charset=utf-8><title>{title}</title><style>body{{background:#111;color:#ddd;"
        f"font:14px sans-serif;margin:16px}}img{{width:100%;max-width:1800px}}figure{{margin:0 0 24px}}</style>"
        f"<h1>{title}</h1>{rows}", encoding="utf-8")
    return OUT_DIR / "index.html"


def contour_key(d):
    """Mesma chave que posicao.py contorno usa na cache."""
    return f"{d['image_id']}|" + ",".join(str(int(round(v))) for v in d["bbox"])


def load_polys():
    """Contornos guardados pelo passo contorno (preds/contorno_cache.jsonl): chave -> polígonos."""
    p = DATA_DIR / "preds" / "contorno_cache.jsonl"
    out = {}
    if p.exists():
        for ln in open(p, encoding="utf-8"):
            r = json.loads(ln)
            if r.get("poly"):
                out[r["k"]] = r["poly"]
    return out


def shifts(poly, W):
    """Nas 360° o contorno pode passar a costura da panorâmica: desenha também deslocado de W."""
    us = [u for u, _ in poly]
    return [0] + ([-W] if max(us) > W else []) + ([W] if min(us) < 0 else [])


def tag(dr, f, fs, x, y, text, color, below=False):
    """Etiqueta pequena com fundo da cor da categoria, por cima (ou por baixo) de (x, y)."""
    tw = dr.textlength(text, font=f)
    ty = y if below else max(0, y - fs - 6)
    dr.rectangle([x, ty, x + tw + 8, ty + fs + 6], fill=color)
    dr.text((x + 4, ty + 2), text, fill=ink(color), font=f)


_ICONS = {}


def icon(d):
    """Desenho oficial do sinal (data/referencias/sinais/<código>.png, Guia da DGV) com o fundo
    transparente. Limites de velocidade: círculo vermelho com o número lido. None se não houver."""
    import cv2
    import numpy as np
    code = (d.get("codigo_rst") or (d.get("codigo") or "").split(":")[-1] or "").strip()
    if code == "C13" and d.get("valor"):
        k = f"C13_{d['valor']}"
        if k not in _ICONS:
            ic = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
            di = ImageDraw.Draw(ic)
            di.ellipse([4, 4, 196, 196], fill=(255, 255, 255, 255), outline=(200, 20, 30, 255), width=24)
            fnt = font(86 if d["valor"] < 100 else 70)
            t = str(d["valor"])
            di.text((100 - di.textlength(t, font=fnt) / 2, 52), t, fill=(0, 0, 0, 255), font=fnt)
            _ICONS[k] = ic
        return _ICONS[k]
    for c in (code, code + "a"):
        if not c:
            continue
        if c not in _ICONS:
            p = ROOT / "data" / "referencias" / "sinais" / f"{c}.png"
            if not p.exists():
                _ICONS[c] = None
                continue
            a = np.asarray(Image.open(p).convert("RGB")).copy()
            h, w = a.shape[:2]
            m = np.zeros((h + 2, w + 2), np.uint8)
            for x, y in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):   # fundo branco -> transparente
                cv2.floodFill(a, m, (x, y), (0, 0, 0), (12, 12, 12), (12, 12, 12),
                              cv2.FLOODFILL_MASK_ONLY | (255 << 8) | 4)
            alpha = np.where(m[1:-1, 1:-1] > 0, 0, 255).astype(np.uint8)
            _ICONS[c] = Image.fromarray(np.dstack([a, alpha]), "RGBA")
        if _ICONS[c] is not None:
            return _ICONS[c]
    return None


def cmd_final(args):
    """Tudo o que foi visto (owlv2_classified, com os dados de owlv2_geo quando há posição):
    só nome e confiança."""
    imgs = load_imgs()
    geo = {(d["image_id"], tuple(d["bbox"])): d
           for ds in load_jsonl(DATA_DIR / "preds" / "owlv2_geo.jsonl").values() for d in ds}
    by_img = load_jsonl(DATA_DIR / "preds" / "owlv2_classified.jsonl")
    for ds in by_img.values():
        for i, d in enumerate(ds):
            ds[i] = geo.get((d["image_id"], tuple(d["bbox"])), d)
    thr = limiares(args.limiares, args.min_conf)
    good = lambda ds: [d for d in ds if d["confidence"] >= thr[d["category"]]]  # noqa: E731
    if args.todas:   # rua seguida: todas as fotos, pela ordem de captura (com ou sem deteções)
        ids = sorted(imgs, key=lambda i: (str(imgs[i].get("sequence")), int(imgs[i]["captured_at"] or 0)))
    else:
        ids = pick(by_img, imgs, args, good)
    polys = load_polys()
    names = []
    for k, iid in enumerate(ids, 1):
        im, top = open_band(imgs[iid])
        fs, lw = max(14, im.width // 160), max(2, im.width // 900)
        f = font(fs)
        ds = sorted(good(by_img[iid]), key=lambda d: d["confidence"])
        # 1) contentores: contorno (SAM 2) pintado por cima da foto. Na iluminação o contorno
        #    apanha muitas vezes só parte do poste: fica a caixa
        overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        with_poly = []
        for d in ds:
            ps = polys.get(contour_key(d)) if d["category"] == "residuos" else None
            if ps:
                col = COLORS[d["category"]]
                for p in ps:
                    for shift in shifts(p, im.width):
                        pts = [(u + shift, v - top) for u, v in p]
                        od.polygon(pts, fill=col + (105,), outline=col + (255,), width=lw)
                with_poly.append((d, ps))
        im = Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")
        dr = ImageDraw.Draw(im)
        drawn = {id(d) for d, _ in with_poly}
        for d, ps in with_poly:
            u0 = min(u for p in ps for u, _ in p) % im.width
            v0 = min(v for p in ps for _, v in p) - top
            tag(dr, f, fs, u0, v0, f"{d['label']} · {d['confidence']:.0%}", COLORS[d["category"]])
        # 2) sinais: caixa + desenho oficial do sinal (Guia da DGV) em vez do texto
        for d in ds:
            if id(d) in drawn:
                continue
            x0, y0, x1, y1 = d["bbox"]
            box = [x0, y0 - top, x1, y1 - top]
            ic = icon(d) if d["category"] == "sinalizacao" else None
            if ic is None:
                label(dr, f, fs, box, f"{d['label']} · {d['confidence']:.0%}", COLORS[d["category"]], lw)
                continue
            dr.rectangle(box, outline=COLORS[d["category"]], width=lw)
            s = int(max(fs * 2.4, min(box[3] - box[1], fs * 5)))
            ic = ic.resize((s, s), Image.LANCZOS)
            ix, iy = int(box[2] + lw * 2), int(max(0, box[1]))
            if ix + s > im.width:
                ix = int(box[0] - lw * 2 - s)
            im.paste(ic, (ix, iy), ic)
            tag(dr, font(int(fs * 0.8)), int(fs * 0.8), ix, iy + s + 2, f"{d['confidence']:.0%}",
                COLORS[d["category"]], below=True)
        attribution(im, dr, f, fs, imgs[iid])
        name = f"{k:04d}_{iid}.jpg" if args.todas else f"{iid}.jpg"
        print(save(im, name))
        names.append(name)
    if args.todas:
        print(f"-> {gallery(names, f'{DATA_DIR.name}: {len(names)} fotos seguidas')}")


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
    from classificar import Classifier, square_crop, suppress_contained
    from detetar import PROMPTS, TILE_W, Owlv2, letterbox, to_original

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


def cmd_contorno(args):
    """Caixa (OWLv2) lado a lado com o contorno (SAM 2): pé pela caixa vs pé real e pontos 3D."""
    import json

    import numpy as np
    import torch
    from posicao import Sam, segment

    imgs, by_img = load_imgs(), load_jsonl(DATA_DIR / "preds" / "owlv2_geo.jsonl")

    def moved(d):                     # quanto o pé mudou, em fração da altura da caixa
        x0, y0, x1, y1 = d["bbox"]
        return (y1 - d["base_y"]) / max(1, y1 - y0)
    cands = [d for ds in by_img.values() for d in ds
             if d["category"] in ("iluminacao", "residuos") and d.get("base_y") is not None
             and d["confidence"] >= args.min_conf and d["bbox"][3] - d["bbox"][1] >= 60
             and (not args.ids or d["image_id"] in args.ids.split(","))]
    cands = [d for d in cands if args.ids or 0.06 <= abs(moved(d)) <= 0.5]
    cands.sort(key=lambda d: (d.get("d3_mask") is None, -d["confidence"]))
    pick_, seen = [], set()
    quota = {"iluminacao": args.n - args.n // 3, "residuos": args.n // 3}
    for d in cands:
        if d["image_id"] in seen or quota[d["category"]] <= 0:
            continue
        seen.add(d["image_id"])
        quota[d["category"]] -= 1
        pick_.append(d)

    out = OUT_DIR / "contorno"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.jpg"):                                # tira as da corrida anterior
        old.unlink()
    sam = Sam("cuda" if torch.cuda.is_available() else "cpu")
    PH = 460                                                     # altura de cada painel
    f, fs = font(20), 20
    rows = []
    for k, d in enumerate(pick_, 1):
        meta = imgs[d["image_id"]]
        im = np.asarray(Image.open(DATA_DIR / "images" / meta["file"]).convert("RGB"))
        pts = None
        p = DATA_DIR / "sfm" / f"{d['image_id']}.npz"
        if p.exists():
            with np.load(p) as z:
                pts = dict(z)
        _, _, v, mask = segment(sam, d, meta, im, pts, keep=True)
        bx0, by0, bx1, by1 = v["box"]
        bw, bh = bx1 - bx0, by1 - by0
        h, w = mask.shape
        mx, my = max(40, min(bw * 1.2, bh * 0.5)), max(30, bh * 0.25)      # margem à volta da caixa
        cx0, cy0 = int(max(0, bx0 - mx)), int(max(0, by0 - my))
        cx1, cy1 = int(min(w, bx1 + mx)), int(min(h, by1 + my))
        base = Image.fromarray(v["img"][cy0:cy1, cx0:cx1])
        s = PH / base.height
        size = (max(160, int(base.width * s)), PH)
        sx = size[0] / base.width
        col = COLORS[d["category"]]
        T = lambda x, y: ((x - cx0) * sx, (y - cy0) * s)  # noqa: E731

        # pontos 3D da Mapillary nesta vista
        P = []
        if pts is not None:
            px, py, ok = v["to_view"](pts["u"], pts["v"])
            ins = ok & (px >= cx0) & (px < cx1) & (py >= cy0) & (py < cy1)
            P = list(zip(px[ins], py[ins]))

        # painel 1: caixa
        a = base.resize(size)
        da = ImageDraw.Draw(a)
        for x, y in P:
            if bx0 <= x <= bx1 and by0 <= y <= by1:
                X, Y = T(x, y)
                da.ellipse([X - 4, Y - 4, X + 4, Y + 4], fill=(0, 220, 255), outline=(0, 0, 0))
        (X0, Y0), (X1, Y1) = T(bx0, by0), T(bx1, by1)
        da.rectangle([X0, Y0, X1, Y1], outline=col, width=4)
        da.line([X0 - 14, Y1, X1 + 14, Y1], fill=(255, 60, 60), width=5)

        # painel 2: contorno
        m = Image.fromarray((mask[cy0:cy1, cx0:cx1] * 255).astype(np.uint8)).resize(size)
        tint = Image.new("RGB", size, col)
        b = Image.composite(Image.blend(base.resize(size), tint, 0.45), base.resize(size), m)
        edge = np.asarray(m) > 127
        edge = edge & ~(np.roll(edge, 1, 0) & np.roll(edge, -1, 0) & np.roll(edge, 1, 1) & np.roll(edge, -1, 1))
        ba = np.asarray(b).copy()
        ba[np.asarray(Image.fromarray(edge.astype(np.uint8) * 255).filter(
            ImageFilter.MaxFilter(3))) > 0] = col
        b = Image.fromarray(ba)
        db = ImageDraw.Draw(b)
        for x, y in P:
            xi, yi = int(x), int(y)
            if 0 <= yi < h and 0 <= xi < w and mask[yi, xi]:
                X, Y = T(x, y)
                db.ellipse([X - 4, Y - 4, X + 4, Y + 4], fill=(0, 220, 255), outline=(0, 0, 0))
        ys, xs = np.nonzero(mask)
        j = np.argmax(ys)
        X, Y = T(xs[j], ys[j])
        db.ellipse([X - 9, Y - 9, X + 9, Y + 9], fill=(40, 220, 90), outline=(0, 0, 0), width=2)

        gap, top, bot = 8, fs + 14, fs + 12
        tile = Image.new("RGB", (size[0] * 2 + gap, PH + top + bot), (20, 20, 20))
        tile.paste(a, (0, top))
        tile.paste(b, (size[0] + gap, top))
        dt = ImageDraw.Draw(tile)
        dt.text((6, 5), "Caixa", fill=(255, 255, 255), font=f)
        dt.text((size[0] + gap + 6, 5), "Contorno", fill=(255, 255, 255), font=f)
        cred = f"{meta.get('creator_username') or '?'} / Mapillary, CC BY-SA 4.0"
        dt.text((6, PH + top + 4), cred, fill=(200, 200, 200), font=font(15))
        name = f"{k:02d}_{d['image_id']}.jpg"
        tile.thumbnail((1400, 1400))
        tile.save(out / name, quality=88)
        n_box = sum(bx0 <= x <= bx1 and by0 <= y <= by1 for x, y in P)
        n_in = sum(1 for x, y in P if 0 <= int(y) < h and 0 <= int(x) < w and mask[int(y), int(x)])
        rows.append({"file": name, "label": d["label"], "category": d["category"], "dist_m": d["dist_m"],
                     "metodo": d.get("geo_method"), "pts_caixa": int(n_box), "pts_contorno": int(n_in),
                     "autor": meta.get("creator_username") or "?", "url": meta["source_url"]})
        print(out / name, f"· {d['label']} · {d['dist_m']} m · pontos 3D caixa {n_box} -> contorno {n_in}")
    (out / "contorno.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    figs = "\n".join(f'<figure><img src="{r["file"]}"><figcaption>{r["label"]} · {r["dist_m"]} m · '
                     f'<a href="{r["url"]}">{r["autor"]} / Mapillary</a>, CC BY-SA 4.0</figcaption></figure>'
                     for r in rows)
    (out / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Caixa vs contorno</title><style>body{background:#111;"
        "color:#ddd;font:14px sans-serif;margin:16px}a{color:#8cf}img{max-width:100%}figure{margin:0 0 24px}"
        f"</style><h1>Caixa vs contorno (SAM 2)</h1>{figs}", encoding="utf-8")
    print(f"-> {out / 'index.html'}")


def main_ver(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    for name, extra in (("final", ("--min-conf", None)), ("owl", ("--min-score", 0.2))):
        p = sub.add_parser(name)
        p.add_argument("--ids", default="", help="ids separados por vírgula")
        p.add_argument("--n", type=int, default=12)
        p.add_argument(extra[0], type=float, default=extra[1])
        p.add_argument("--frontal", action="store_true", help="só fotos frontais")
        p.add_argument("--pano", action="store_true", help="só fotos 360°")
        p.add_argument("--todas", action="store_true", help="todas as fotos por ordem de captura + index.html")
        p.add_argument("--saida", default="", help="subpasta de data/viz (ex.: limiar_0.70)")
        p.add_argument("--limiares", default="", help="por categoria, ex.: iluminacao=0.7,residuos=0.5")
    p = sub.add_parser("foto")
    p.add_argument("fotos", nargs="+")
    p.add_argument("--det-threshold", type=float, default=0.2)
    p = sub.add_parser("contorno")
    p.add_argument("--ids", default="", help="ids das fotos separados por vírgula")
    p.add_argument("--n", type=int, default=9)
    p.add_argument("--min-conf", type=float, default=0.3)
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] not in ("final", "owl", "foto", "contorno", "-h", "--help"):
        argv = ["final", *argv]   # compatível com o uso antigo: imagens.py --n 12
    args = ap.parse_args(argv)
    global OUT_DIR
    if getattr(args, "saida", ""):
        OUT_DIR = OUT_DIR / args.saida
    {"final": cmd_final, "owl": cmd_owl, "foto": cmd_foto, "contorno": cmd_contorno}[args.cmd](args)


####################################################################################################
# FOLHAS  (antes: scripts/folha_catalogo.py)
####################################################################################################
# Revisão a olho do catálogo, em 2 passos:
#
#   folha     folha de imagens de um tipo: exemplos numerados, cada um com a caixa a vermelho
#             e a foto à volta -> data/relatorios/folhas/<subtipo>.jpg (+ .csv: número -> id)
#   guardar   guarda os números escolhidos numa folha como FOTOS DE REFERÊNCIA (recorte com
#             margem) -> data/referencias/aveiro/<categoria>/<classe>/<id>.jpg
#             ou, para "parece mas não é", data/referencias/aveiro/nao/<tipo>/<id>.jpg
#             (créditos em data/referencias/aveiro/fontes.csv)
#
# Uso:
#     python scripts/imagens.py folha --subtipo candeeiro subterraneo --n 24
#     python scripts/imagens.py guardar --folha candeeiro --numeros 3,4,10 --classe iluminacao/candeeiro
#     python scripts/imagens.py guardar --folha indiferenciado --numeros 7,13 --classe nao/papeleira


Image.MAX_IMAGE_PIXELS = None
FOLHAS = DATA_DIR / "relatorios" / "folhas"
REFS = DATA_DIR / "referencias" / "aveiro"


def catalog():
    with open(DATA_DIR / "catalogo" / "catalogo.geojson", encoding="utf-8") as f:
        return [ft["properties"] for ft in json.load(f)["features"]]


def files():
    with open(DATA_DIR / "meta" / "sample_images.csv", encoding="utf-8") as f:
        return {r["id"]: r["file"] for r in csv.DictReader(f)}


def cmd_folha(args):
    feats, fs = catalog(), files()
    FOLHAS.mkdir(parents=True, exist_ok=True)
    f15 = font(15)
    for sub in args.subtipo:
        es = [e for e in feats if e["subtipo"] == sub]
        if not es:
            print(f"{sub}: nenhum elemento no catálogo")
            continue
        sample = random.Random(args.seed).sample(es, min(args.n, len(es)))
        cols, size = 8, 200
        sheet = Image.new("RGB", (cols * size, ((len(sample) + cols - 1) // cols) * size), "white")
        dr = ImageDraw.Draw(sheet)
        rows = []
        for i, e in enumerate(sample):
            im = Image.open(DATA_DIR / "images" / fs[e["imagem_id"]]).convert("RGB")
            x0, y0, x1, y1 = e["imagem_bbox"]
            cx, cy, s = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * 0.8 + 20
            ox, oy = int(cx - s), int(cy - s)
            v = im.crop((ox, oy, int(cx + s), int(cy + s)))
            ImageDraw.Draw(v).rectangle([x0 - ox, y0 - oy, x1 - ox, y1 - oy], outline=(255, 0, 0),
                                        width=max(2, int(s / 50)))
            x, y = (i % cols) * size, (i // cols) * size
            sheet.paste(v.resize((size - 4, size - 4)), (x + 2, y + 2))
            dr.rectangle([x + 2, y + 2, x + 34, y + 21], fill=(0, 0, 0))
            dr.text((x + 5, y + 3), str(i + 1), fill=(255, 255, 0), font=f15)
            rows.append({"n": i + 1, "id": e["id"], "label": e["label"], "confianca": e["confianca"],
                         "n_vistas": e["n_vistas"], "imagem_id": e["imagem_id"]})
        sheet.save(FOLHAS / f"{sub}.jpg", quality=88)
        with open(FOLHAS / f"{sub}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"{sub}: {len(sample)} de {len(es)} -> {FOLHAS / f'{sub}.jpg'}")


def cmd_guardar(args):
    with open(FOLHAS / f"{args.folha}.csv", encoding="utf-8-sig") as f:
        rows = {int(r["n"]): r for r in csv.DictReader(f)}
    cat, fs = {e["id"]: e for e in catalog()}, files()
    out = REFS / args.classe
    out.mkdir(parents=True, exist_ok=True)
    log = REFS / "fontes.csv"
    new = not log.exists()
    nums = [int(x) for x in args.numeros.split(",")]
    with open(log, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ficheiro", "classe", "id_catalogo", "imagem_id", "fonte", "licenca"])
        for n in nums:
            e = cat[rows[n]["id"]]
            im = Image.open(DATA_DIR / "images" / fs[e["imagem_id"]]).convert("RGB")
            x0, y0, x1, y1 = e["imagem_bbox"]
            m = 0.15 * max(x1 - x0, y1 - y0)
            p = out / f"{e['id']}.jpg"
            im.crop((int(x0 - m), int(y0 - m), int(x1 + m), int(y1 + m))).save(p, quality=92)
            w.writerow([p.relative_to(REFS).as_posix(), args.classe, e["id"], e["imagem_id"],
                        e.get("imagem_url"), "CC BY-SA 4.0 (Mapillary)"])
    print(f"{args.classe}: +{len(nums)} -> {out}")


def main_folhas(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("folha", help="folha de imagens numeradas de um ou mais tipos")
    p.add_argument("--subtipo", nargs="+", required=True)
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("guardar", help="guarda números de uma folha como fotos de referência")
    p.add_argument("--folha", required=True, help="nome da folha (subtipo) em data/relatorios/folhas")
    p.add_argument("--numeros", required=True, help="números na folha, separados por vírgula")
    p.add_argument("--classe", required=True, help="<categoria>/<classe> ou nao/<tipo>")
    args = ap.parse_args(argv)
    {"folha": cmd_folha, "guardar": cmd_guardar}[args.cmd](args)


####################################################################################################
SECOES = {
    "final": (lambda a, c="final": main_ver([c, *a]), "final"),
    "owl": (lambda a, c="owl": main_ver([c, *a]), "owl"),
    "foto": (lambda a, c="foto": main_ver([c, *a]), "foto"),
    "contorno": (lambda a, c="contorno": main_ver([c, *a]), "contorno"),
    "folha": (lambda a, c="folha": main_folhas([c, *a]), "folha"),
    "guardar": (lambda a, c="guardar": main_folhas([c, *a]), "guardar"),
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in SECOES:
        print(__doc__)
        print("Comandos: " + ", ".join(SECOES))
        sys.exit(0 if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help") else 1)
    cmd = sys.argv[1]
    fn = SECOES[cmd][0]
    # nos comandos que já têm subcomando próprio (lambda) o argparse acrescenta o nome sozinho
    sys.argv[0] = os.path.basename(sys.argv[0]) + ("" if fn.__name__ == "<lambda>" else f" {cmd}")
    fn(sys.argv[2:])


if __name__ == "__main__":
    main()
