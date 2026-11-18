"""
Código EXATO de cada sinal por semelhança imagem-imagem com os desenhos oficiais
(ex.: C11a "Proibição de virar à direita" vs C11b "... à esquerda"). Experiência: ainda não
entra na pipeline. Dois comandos:

  extrair   desenhos dos sinais do Guia de Sinalização Rodoviária (DGV, 2003), com o código e o
            nome lidos da legenda (ex.: "A1a – Curva à direita")
            pdfs/SinalizacaoRodoviaria.pdf -> data/referencias/sinais/<código>.png + sinais.csv
            Uso interno para teste — confirmar com os mentores se os desenhos podem ser usados.
  testar    cada recorte da rua -> "impressão digital" -> código do desenho mais parecido.
            Banco de referências = desenhos com variações (rodado, perspetiva, desfocado,
            pequeno, mais escuro, fundo trocado). Modelos: SigLIP (omissão) ou DINOv2.
            Teste a 17/11 (400 sinais): SigLIP 78% · DINOv2 23–38% (estranha desenhos vs fotos).
            Pares espelhados (esquerda/direita) detetados automaticamente; nesses pares a escolha
            é refeita píxel a píxel ("pixeis") ou por zonas do sinal ("zonas").
            Medição: sinais do sistema que a Mapillary também marcou (classe Mapillary -> código
            RST, tabela MLY_TO_RST, só para medir) -> data/relatorios/semelhanca_<modelo>.json

Uso:
    python scripts/semelhanca_sinais.py extrair
    python scripts/semelhanca_sinais.py testar
    python scripts/semelhanca_sinais.py testar --modelo dinov2 --limite 400
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(__file__))
from comum import DATA_DIR, ROOT, load_jsonl, matches  # noqa: E402
from ler_limites import flat_print, pano_print  # noqa: E402


# ================================================================== extrair (PDF da DGV)
PDF = ROOT / "pdfs" / "SinalizacaoRodoviaria.pdf"
REF_OUT = ROOT / "data" / "referencias"
SCALE = 4  # render a 4x (288 dpi)
# código RST no início da legenda: A1a, B2, C13, D4, H7, C11a, E1, ...
CODE = re.compile(r"^\s*([A-Z]{1,2}\d{1,2}[a-z]?)\s*[–—-]\s*(.+)$")
FAMILIAS = {"A": "perigo", "B": "cedencia", "C": "proibicao", "D": "obrigacao", "E": "prescricao",
            "F": "prescricao", "G": "zona", "H": "informacao", "I": "pre_sinalizacao",
            "J": "direcao", "L": "confirmacao", "N": "localidade", "O": "outros", "M": "adicional"}


def caption(tp, box, rects):
    """Texto das linhas por baixo do desenho, na mesma coluna."""
    l, b, r, t = box
    cx = (l + r) / 2
    sel = [q for q in rects if b - 60 <= q[3] <= b + 2 and q[0] - 15 <= cx <= q[2] + 15]
    if not sel:
        return ""
    # só as linhas seguidas (sem salto grande) por baixo do desenho
    sel.sort(key=lambda q: -q[3])
    keep = [sel[0]]
    for q in sel[1:]:
        if keep[-1][1] - q[3] > 8:
            break
        keep.append(q)
    # texto do bloco inteiro de uma vez (evita palavras partidas entre retângulos)
    box = (min(q[0] for q in keep) - 1, min(q[1] for q in keep) - 1,
           max(q[2] for q in keep) + 1, max(q[3] for q in keep) + 1)
    return re.sub(r"\s+", " ", tp.get_text_bounded(*box)).strip()


def cmd_extrair(args):
    """Desenhos + código + nome dos sinais do PDF da DGV."""
    import pypdfium2 as pdfium  # Apache/BSD (não usar PyMuPDF: AGPL)
    if not PDF.exists():
        sys.exit(f"Falta {PDF}")
    (REF_OUT / "sinais").mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(str(PDF))
    rows, seen = [], set()
    for pi in range(len(pdf)):
        page = pdf[pi]
        tp = page.get_textpage()
        rects = [tp.get_rect(k) for k in range(tp.count_rects())]
        H = page.get_size()[1]
        bitmap = None
        for o in page.get_objects():
            if o.type != 3:  # 3 = imagem
                continue
            box = o.get_pos()
            m = CODE.match(caption(tp, box, rects))
            if not m:
                continue
            code, name = m.group(1), re.sub(r"\s+", " ", m.group(2)).strip()
            if code in seen:
                continue
            seen.add(code)
            if bitmap is None:
                bitmap = page.render(scale=SCALE).to_pil().convert("RGB")
            l, b, r, t = box
            crop = bitmap.crop((int(l * SCALE), int((H - t) * SCALE), int(r * SCALE), int((H - b) * SCALE)))
            fname = f"{code}.png"
            crop.save(REF_OUT / "sinais" / fname)
            rows.append({"codigo": code, "nome": name, "familia": FAMILIAS.get(code[0], "outros"),
                         "pagina": pi + 1, "ficheiro": f"sinais/{fname}"})
    with open(REF_OUT / "sinais.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print(f"{len(rows)} sinais extraídos -> {REF_OUT / 'sinais.csv'}")
    print("Por família: " + ", ".join(f"{k}: {v}" for k, v in Counter(r["familia"] for r in rows).items()))


# ================================================================== testar (semelhança)
MODELOS = {"siglip": "google/siglip-so400m-patch14-384",  # Apache 2.0
           "dinov2": "facebook/dinov2-base"}                # Apache 2.0
FAMILIAS_RUA = ("A", "B", "C", "D", "E", "G", "H")  # sinais verticais
REF_INNER, CROP_INNER = 0.8, 0.8 / 1.4  # interior do sinal (o recorte da rua tem margem 1,4)
METODOS = ("modelo", "+pixeis", "+zonas", "+pixeis_se_duvida")
TAUS = (0.005, 0.01, 0.02, 0.03, 0.05)  # margens de "dúvida" testadas

# Classe Mapillary -> códigos RST aceites (SÓ para medir; nunca entra no catálogo)
MLY_TO_RST = {
    "information--pedestrians-crossing": ["H7"],
    "regulatory--no-entry": ["C1"],
    "regulatory--one-way-straight": ["H3"],
    "regulatory--yield": ["B1"],
    "regulatory--no-stopping": ["C16"],
    "regulatory--stop": ["B2"],
    "regulatory--keep-right": ["D3a"],
    "regulatory--keep-left": ["D3b"],
    "warning--road-bump": ["A2a"],
    "regulatory--bicycles-only": ["D7a"],
    "regulatory--roundabout": ["D4"],
    "information--parking": ["H1a", "H1b"],
    "regulatory--no-right-turn": ["C11a"],
    "regulatory--no-left-turn": ["C11b"],
    "regulatory--maximum-speed-limit-": ["C13"],
    "information--dead-end": ["H4"],
    "warning--two-way-traffic": ["A25"],
    "warning--roundabout": ["B7"],
    "regulatory--no-parking": ["C15"],
    "regulatory--turn-right-ahead": ["D1e", "D1a"],
    "regulatory--turn-left-ahead": ["D1d", "D1b"],
    "regulatory--turn-right": ["D1a", "D1e"],
    "regulatory--turn-left": ["D1b", "D1d"],
    "regulatory--go-straight": ["D1c"],
    "regulatory--go-straight-or-turn-right": ["D2b"],
    "regulatory--go-straight-or-turn-left": ["D2a"],
    "regulatory--turn-left-or-right": ["D2c"],
    "regulatory--road-closed-to-vehicles": ["C2"],
    "regulatory--buses-only": ["D6"],
    "regulatory--height-limit": ["C9"],
    "regulatory--weight-limit": ["C6", "C5"],
    "regulatory--length-limit": ["C7"],
    "regulatory--no-u-turn": ["C12"],
    "warning--pedestrians-crossing": ["A16a", "A16b"],
    "regulatory--end-of-bicycles-only": ["D13a"],
    "regulatory--no-heavy-goods-vehicles": ["C3c", "C3b"],
    "warning--roadworks": ["A23"],
    "regulatory--dual-path-bicycles-and-pedestrians": ["D7e", "D7f"],
    "warning--children": ["A14"],
    "warning--traffic-signals": ["A22"],
    "regulatory--no-overtaking": ["C14a"],
    "warning--bicycles-crossing": ["A17"],
    "warning--curve-right": ["A1a"],
    "warning--curve-left": ["A1b"],
    "regulatory--pedestrians-only": ["D7b"],
    "warning--other-danger": ["A29"],
    "regulatory--end-of-parking-zone": ["G6"],
    "regulatory--end-of-no-parking": ["C21"],
}


def mly_codes(value):
    base = value.rsplit("--", 1)[0]
    if base.startswith("regulatory--maximum-speed-limit-"):
        base = "regulatory--maximum-speed-limit-"
    return MLY_TO_RST.get(base)


# ------------------------------------------------------------------ referências
def load_refs():
    with open(DATA_DIR / "referencias" / "sinais.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["codigo"].startswith(FAMILIAS_RUA)
                and not r["codigo"].startswith("ET")]
    return [(r["codigo"], r["nome"], Image.open(DATA_DIR / "referencias" / r["ficheiro"]).convert("RGB"))
            for r in rows]


def background_mask(img):
    """Fundo branco à volta do desenho (enchimento a partir dos cantos)."""
    a = np.asarray(img).copy()
    h, w = a.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    for x, y in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        cv2.floodFill(a, mask, (x, y), (0, 0, 0), (12, 12, 12), (12, 12, 12),
                      cv2.FLOODFILL_MASK_ONLY | (255 << 8) | 4)
    return mask[1:-1, 1:-1] > 0


def augment(img, rng):
    """Uma variação do desenho com aspeto de foto da rua."""
    a = np.asarray(img).astype(np.float32)
    bg = background_mask(img)
    # fundo: cor lisa ao acaso + ruído (céu, fachada, árvore...)
    col = rng.choice([(150, 180, 210), (120, 120, 120), (90, 110, 70), (200, 200, 195), (60, 60, 60)])
    noise = rng.normal(0, 18, a.shape)
    a[bg] = (np.array(col, np.float32) + noise[bg]).clip(0, 255)
    im = Image.fromarray(a.astype(np.uint8))
    fill = tuple(int(c) for c in col)
    # margem, rotação e perspetiva ligeiras
    w, h = im.size
    pad = int(0.2 * max(w, h))
    canvas = Image.new("RGB", (w + 2 * pad, h + 2 * pad), fill)
    canvas.paste(im, (pad, pad))
    im = canvas.rotate(rng.uniform(-8, 8), resample=Image.BICUBIC, fillcolor=fill)
    W, H = im.size
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    dst = src + np.float32(rng.uniform(-0.08, 0.08, (4, 2)) * [W, H])
    M = cv2.getPerspectiveTransform(src, dst)
    im = Image.fromarray(cv2.warpPerspective(np.asarray(im), M, (W, H), borderMode=cv2.BORDER_REPLICATE))
    # tamanho pequeno (como sinais longe) e desfocagem
    s = int(rng.choice([24, 32, 48, 64, 96, 160]))
    im = im.resize((s, s), Image.BILINEAR).resize((224, 224), Image.BICUBIC)
    if rng.random() < 0.5:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.5)))
    # luz
    arr = np.asarray(im).astype(np.float32) * rng.uniform(0.6, 1.2) + rng.uniform(-20, 20)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def framed(im):
    """Desenho com margem branca, como os recortes da rua (sinal ~70% do quadro)."""
    w, h = im.size
    pad = int(0.2 * max(w, h))
    c = Image.new("RGB", (w + 2 * pad, h + 2 * pad), (255, 255, 255))
    c.paste(im, (pad, pad))
    return c.resize((224, 224), Image.BICUBIC)


def gray64(im, inner=1.0):
    """Interior do sinal em cinzento 64x64, normalizado (média 0, desvio 1)."""
    a = cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2GRAY)
    if inner < 1:
        h, w = a.shape
        m = (1 - inner) / 2
        a = a[int(h * m):int(h * (1 - m)), int(w * m):int(w * (1 - m))]
    a = cv2.resize(a, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    return (a - a.mean()) / (a.std() + 1e-6)


def ncc(a, b):
    """Semelhança píxel a píxel (1 = igual, 0 = nada a ver)."""
    return float((a * b).mean())


def mirror_pairs(refs, thr=0.7, gap=0.1):
    """Pares (X, Y) em que o interior de Y é o espelho do de X."""
    g = {c: gray64(im, REF_INNER) for c, _, im in refs}
    pairs = {}
    codes = list(g)
    for i, a in enumerate(codes):
        for b in codes[i + 1:]:
            if a[0] != b[0] or a in pairs or b in pairs:
                continue
            m = ncc(np.fliplr(g[a]), g[b])
            if m > thr and m > ncc(g[a], g[b]) + gap:
                pairs[a], pairs[b] = b, a
    return pairs


# ------------------------------------------------------------------ modelos
class Encoder:
    """Impressões digitais de imagens: vetor global e vetores por zona, normalizados."""

    def __init__(self, nome, device):
        from transformers import AutoImageProcessor, AutoModel
        self.nome, self.device = nome, device
        self.proc = AutoImageProcessor.from_pretrained(MODELOS[nome])
        self.model = AutoModel.from_pretrained(MODELOS[nome], dtype=torch.float16).to(device).eval()

    @torch.inference_mode()
    def __call__(self, images, batch=32):
        gs, ps = [], []
        for i in range(0, len(images), batch):
            if self.nome == "dinov2":
                x = self.proc(images=images[i:i + batch], return_tensors="pt",
                              do_center_crop=False, size={"height": 224, "width": 224})
                out = self.model(pixel_values=x["pixel_values"].to(self.device).half())
                g, p = out.pooler_output, out.last_hidden_state[:, 1:]
            else:
                x = self.proc(images=images[i:i + batch], return_tensors="pt")
                out = self.model.vision_model(pixel_values=x["pixel_values"].to(self.device).half())
                g, p = out.pooler_output, out.last_hidden_state
            g, p = g.float(), p.float()
            gs.append(g / g.norm(dim=-1, keepdim=True))
            # zonas guardadas na memória normal (no SigLIP: 729 zonas x 1152 números por
            # imagem; 2 000 recortes encheriam a placa gráfica e tudo ficaria muito lento)
            ps.append((p / p.norm(dim=-1, keepdim=True)).half().cpu())
        return torch.cat(gs), torch.cat(ps)


# ------------------------------------------------------------------ recortes da rua
def street_cases(limit):
    imgs = {r["id"]: r for r in csv.DictReader(open(DATA_DIR / "meta" / "sample_images.csv", encoding="utf-8"))}
    refs = load_jsonl(DATA_DIR / "ground_truth" / "mly_detections.jsonl")
    ours = load_jsonl(DATA_DIR / "preds" / "owlv2_classified.jsonl")
    cases = []
    for iid, rs in refs.items():
        if iid not in imgs:
            continue
        area = float(imgs[iid]["thumb_w"]) * float(imgs[iid]["thumb_h"])
        ps = [p for p in ours.get(iid, []) if p["category"] == "sinalizacao"]
        for r in rs:
            b = r["bbox"]
            codes = mly_codes(r["value"]) if r["category"] == "sinalizacao" else None
            if not codes or max(b[2] - b[0], b[3] - b[1]) < 24:
                continue
            cand = [p for p in ps if matches({"bbox": p["bbox"]}, r, area)]
            if cand:
                best = max(cand, key=lambda p: p["confidence"])
                cases.append({"image_id": iid, "bbox": best["bbox"], "codes": codes,
                              "mly": r["value"], "pano": imgs[iid]["is_pano"] == "True",
                              "w": best["bbox"][2] - best["bbox"][0]})
    random.Random(0).shuffle(cases)
    return cases[:limit] if limit else cases


def load_crops(cases):
    """Print de cada sinal (vista normal, ampliada), 224x224, sinal ~70% do quadro."""
    by_img = defaultdict(list)
    for i, c in enumerate(cases):
        by_img[c["image_id"]].append(i)

    def work(iid):
        img = np.asarray(Image.open(DATA_DIR / "images" / f"{iid}.jpg").convert("RGB"))
        out = []
        for i in by_img[iid]:
            c = cases[i]
            v = pano_print(img, c["bbox"]) if c["pano"] else flat_print(img, c["bbox"])
            out.append((i, Image.fromarray(v).resize((224, 224), Image.BICUBIC)))
        return out

    crops = [None] * len(cases)
    with ThreadPoolExecutor(os.cpu_count() or 8) as pool:
        for res in pool.map(work, list(by_img)):
            for i, im in res:
                crops[i] = im
    return crops


# ------------------------------------------------------------------ principal
def cmd_testar(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)

    refs = load_refs()
    pairs = mirror_pairs(refs)
    print(f"{len(refs)} desenhos de referência · pares espelhados: "
          + ", ".join(f"{a}/{b}" for a, b in sorted(pairs.items()) if a < b))

    enc = Encoder(args.modelo, device)
    t = time.time()
    bank_imgs, bank_codes = [], []
    for code, _, im in refs:
        bank_imgs.append(framed(im))
        bank_codes.append(code)
        for _ in range(args.variacoes):
            bank_imgs.append(augment(im, rng))
            bank_codes.append(code)
    bank_g, _ = enc(bank_imgs)
    ref_img = {c: im for c, _, im in refs}
    ref_gray = {c: gray64(im, REF_INNER) for c, im in ref_img.items()}
    ref_idx = {c: i for i, c in enumerate(ref_img)}
    _, ref_p = enc([framed(ref_img[c]) for c in ref_img])
    print(f"Banco: {len(bank_imgs)} imagens ({time.time() - t:.0f} s)")

    cases = street_cases(args.limite)
    t = time.time()
    crops = load_crops(cases)
    g, p = enc(crops)
    print(f"{len(cases)} recortes da rua ({time.time() - t:.0f} s)")

    codes_arr = np.array(bank_codes)
    uniq = np.unique(codes_arr)
    sims = (g @ bank_g.T).cpu().numpy()
    best = np.stack([sims[:, codes_arr == u].max(1) for u in uniq], 1)  # melhor por código
    col = {str(u): i for i, u in enumerate(uniq)}
    res, lr, lr_tau = Counter(), Counter(), Counter()
    by_mly = defaultdict(Counter)
    for k, c in enumerate(cases):
        top = str(uniq[best[k].argmax()])
        preds = {m: top for m in METODOS}
        if top in pairs:
            other = pairs[top]
            cg = gray64(crops[k], CROP_INNER)
            px = top if ncc(cg, ref_gray[top]) >= ncc(cg, ref_gray[other]) else other
            preds["+pixeis"] = px
            # combinado: o píxel só decide quando o modelo hesita entre os dois lados
            margin = best[k, col[top]] - best[k, col[other]]
            preds["+pixeis_se_duvida"] = px if margin < args.duvida else top
            for tau in TAUS:
                lr_tau[tau] += (px if margin < tau else top) in c["codes"] if any(
                    x in pairs for x in c["codes"]) else 0
            pk = p[k].float()
            za = float((pk * ref_p[ref_idx[top]].float()).sum(-1).mean())
            zb = float((pk * ref_p[ref_idx[other]].float()).sum(-1).mean())
            preds["+zonas"] = top if za >= zb else other
        res["n"] += 1
        for m, v in preds.items():
            ok = v in c["codes"]
            res[m] += ok
            by_mly[c["mly"].rsplit("--", 1)[0]][m] += ok
        # esquerda/direita: casos cuja resposta certa está num par espelhado
        if any(x in pairs for x in c["codes"]):
            lr["n"] += 1
            for m, v in preds.items():
                lr[m] += v in c["codes"]

    n = res["n"]
    print(f"\nModelo: {args.modelo}. Código exato certo ({n} sinais):")
    for m in METODOS:
        print(f"  {m:<18} {res[m] / n:.0%}")
    if lr["n"]:
        print(f"\nSó sinais de pares esquerda/direita ({lr['n']} sinais):")
        for m in METODOS:
            print(f"  {m:<18} {lr[m] / lr['n']:.0%}")
        print("  píxel decide se a diferença do modelo entre os dois lados for menor que: "
              + " · ".join(f"{t}: {lr_tau[t] / lr['n']:.0%}" for t in TAUS))
    print("\nPor classe Mapillary (n · " + " · ".join(METODOS) + "):")
    tot = Counter(c["mly"].rsplit("--", 1)[0] for c in cases)
    for k, v in tot.most_common(30):
        b = by_mly[k]
        print(f"  {v:>4}  " + " ".join(f"{b[m] / v:>5.0%}" for m in METODOS) + f"  {k}")
    out = DATA_DIR / "relatorios" / f"semelhanca_{args.modelo}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "modelo": args.modelo, "n": n,
        "certo": {m: round(res[m] / n, 3) for m in METODOS},
        "pares_esq_dir": {"n": lr["n"], **{m: round(lr[m] / max(1, lr["n"]), 3) for m in METODOS}},
        "por_classe": {k: {"n": v, **{m: round(by_mly[k][m] / v, 3) for m in METODOS}}
                       for k, v in tot.most_common()},
        "pares": sorted(f"{a}/{b}" for a, b in pairs.items() if a < b)},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("extrair", help="desenhos + códigos dos sinais do PDF da DGV")
    p = sub.add_parser("testar", help="medir a semelhança recorte da rua -> desenho")
    p.add_argument("--modelo", choices=list(MODELOS), default="siglip")
    p.add_argument("--variacoes", type=int, default=10, help="variações por desenho")
    p.add_argument("--limite", type=int, default=0, help="n.º de recortes a testar (0 = todos)")
    p.add_argument("--duvida", type=float, default=0.02,
                   help="o teste do espelho só decide se o modelo hesitar menos que isto")
    args = ap.parse_args()
    {"extrair": cmd_extrair, "testar": cmd_testar}[args.cmd](args)


if __name__ == "__main__":
    main()
