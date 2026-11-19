"""
Detalhes dos SINAIS (passos da pipeline depois de classificar):

  limites   número dos limites de velocidade, lido pelo SigLIP num "print" do sinal
  codigo    código e nome oficial do Guia de Sinalização da DGV (perigo -> "Trânsito nos dois
            sentidos", A25 ...), por semelhança com os desenhos do PDF
  extrair   desenhos + códigos dos sinais do PDF da DGV -> data/referencias/sinais/ (uma vez)

Uso:
    python scripts/sinais.py limites
    python scripts/sinais.py codigo
    python scripts/sinais.py extrair
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageFilter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from comum import DATA_DIR, ROOT  # noqa: E402


####################################################################################################
# LIMITES  (antes: scripts/ler_limites.py)
####################################################################################################
# Lê o número dos sinais de limite de velocidade (passo entre classificar.py e posicao.py).
#
# Para cada sinal "limite_velocidade":
#   1. "print" do sinal: vista normal centrada nele, tirada da 360° com a resolução
#      original (sem a deformação da 360°), ampliada 4x. Nas fotos normais, recorte ampliado.
#   2. SigLIP lê o número: compara o print com frases "sinal de limite com o número N"
#      e escolhe a mais parecida. "valor_conf" = confiança dessa escolha (0–1);
#      abaixo de --rever (0.6) o sinal fica marcado para o técnico confirmar.
#
# Teste a 17/11 (64 sinais com o número confirmado por pontos Mapillary):
#   OCR (EasyOCR) na foto 31% certos · OCR no print 55% · SigLIP no print 97%.
#   Por isso o OCR foi retirado do projeto. A medição repete-se com avaliar.py erros.
#
# Entrada/saída:  data/preds/<modelo>_classified.jsonl (atualizado no lugar)
#
# Uso:
#     python scripts/sinais.py limites --model owlv2



Image.MAX_IMAGE_PIXELS = None
SIGLIP = "google/siglip-so400m-patch14-384"
SPEEDS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120]
SCALE, MARGIN = 4, 1.4


def pano_print(pano, box):
    """Vista normal centrada na caixa (rumo e inclinação), densidade = original x SCALE."""
    H, W = pano.shape[:2]
    x0, y0, x1, y1 = box
    yaw = ((x0 + x1) / 2 / W - 0.5) * 2 * math.pi
    pitch = (0.5 - (y0 + y1) / 2 / H) * math.pi
    half = max(x1 - x0, y1 - y0) / W * math.pi * MARGIN
    f = W / (2 * math.pi) * SCALE
    size = max(32, int(2 * f * math.tan(half)))
    c = (np.arange(size, dtype=np.float32) + 0.5 - size / 2) / f
    x, y = np.meshgrid(c, c)
    cp, sp = math.cos(pitch), math.sin(pitch)
    y2, z2 = y * cp - sp, y * sp + cp
    lon = yaw + np.arctan2(x, z2)
    lat = -np.arctan2(y2, np.sqrt(x * x + z2 * z2))
    u = ((lon / (2 * math.pi) + 0.5) % 1.0) * W
    v = (0.5 - lat / math.pi) * H
    return cv2.remap(pano, u.astype(np.float32), v.astype(np.float32),
                     cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def flat_print(img, box):
    x0, y0, x1, y1 = box
    cx, cy, s = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * MARGIN / 2
    c = img[max(0, int(cy - s)):int(cy + s), max(0, int(cx - s)):int(cx + s)]
    return cv2.resize(c, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)


class SpeedReader:
    def __init__(self, device, classifier=None):
        """classifier: um classificar.Classifier já carregado -> reaproveita o mesmo SigLIP (a app
        carrega o modelo uma só vez)."""
        if classifier is not None:
            self.proc, self.model, self.dtype = classifier.proc, classifier.model, classifier.dtype
        else:
            from transformers import AutoModel, AutoProcessor
            self.dtype = torch.float16 if torch.device(device).type == "cuda" else torch.float32
            self.proc = AutoProcessor.from_pretrained(SIGLIP)
            self.model = AutoModel.from_pretrained(SIGLIP, dtype=self.dtype).to(device).eval()
        self.device = device
        with torch.inference_mode():
            tok = self.proc(text=[f"This is a photo of a round speed limit traffic sign with the number {n}."
                                  for n in SPEEDS], padding="max_length", return_tensors="pt").to(device)
            t = self._t(self.model.get_text_features(**tok))
        self.text = t / t.norm(dim=-1, keepdim=True)

    @staticmethod
    def _t(x):
        return (x if torch.is_tensor(x) else x.pooler_output).float()

    @torch.inference_mode()
    def __call__(self, view):
        im = Image.fromarray(view).resize((384, 384), Image.BICUBIC)
        pv = self.proc(images=im, return_tensors="pt")["pixel_values"].to(self.device).to(self.dtype)
        e = self._t(self.model.get_image_features(pixel_values=pv))
        p = torch.softmax((e / e.norm(dim=-1, keepdim=True)) @ self.text.T * self.model.logit_scale.exp().float(), -1)[0]
        i = int(p.argmax())
        return SPEEDS[i], float(p[i])


def main_limites(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="owlv2")
    ap.add_argument("--rever", type=float, default=0.6, help="abaixo desta confiança: a rever")
    args = ap.parse_args(argv)

    with open(DATA_DIR / "meta" / "sample_images.csv", encoding="utf-8") as f:
        imgs = {r["id"]: r for r in csv.DictReader(f)}
    path = DATA_DIR / "preds" / f"{args.model}_classified.jsonl"
    with path.open(encoding="utf-8") as f:
        recs = [json.loads(line) for line in f]
    todo = sorted((i for i, r in enumerate(recs) if r["subtype"] == "limite_velocidade"),
                  key=lambda i: recs[i]["image_id"])
    print(f"{len(todo)} sinais de limite de velocidade")
    with open(ROOT / "config" / "taxonomia.yaml", encoding="utf-8") as f:
        subs = yaml.safe_load(f)["categorias"]["sinalizacao"]["subtipos"]
    template = next(s["label"] for s in subs if s["id"] == "limite_velocidade")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    siglip = SpeedReader(device)

    stats, cur_id, img = Counter(), None, None
    for i in tqdm(todo, unit="sinal"):
        r = recs[i]
        if r["image_id"] != cur_id:
            cur_id = r["image_id"]
            img = np.asarray(Image.open(DATA_DIR / "images" / f"{cur_id}.jpg").convert("RGB"))
        pano = imgs.get(cur_id, {}).get("is_pano") == "True"
        view = pano_print(img, r["bbox"]) if pano else flat_print(img, r["bbox"])
        val, conf = siglip(view)
        stats[val] += 1
        stats["a_rever"] += conf < args.rever
        r.pop("valor_ocr", None)
        r.update({"valor": val, "valor_conf": round(conf, 3), "valor_rever": conf < args.rever,
                  "label": template.replace("{n}", str(val))})

    with path.open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
    print("Números lidos: " + ", ".join(f"{n} km/h: {stats[n]}" for n in SPEEDS if stats[n]))
    print(f"A rever pelo técnico (confiança < {args.rever:.0%}): {stats['a_rever']}")
    print(f"-> {path}")


####################################################################################################
# DESENHOS  (antes: scripts/semelhanca_sinais.py)
####################################################################################################
# Código EXATO de cada sinal por semelhança imagem-imagem com os desenhos oficiais
# (ex.: C11a "Proibição de virar à direita" vs C11b "... à esquerda"). Dois comandos:
#
#   extrair   desenhos dos sinais do Guia de Sinalização Rodoviária (DGV, 2003), com o código e o
#             nome lidos da legenda (ex.: "A1a – Curva à direita")
#             pdfs/SinalizacaoRodoviaria.pdf -> data/referencias/sinais/<código>.png + sinais.csv
#             Uso interno para teste — confirmar com os mentores se os desenhos podem ser usados.
#   (o teste de 17–18/11 escolheu o SigLIP: 78–84% de código exato; DINOv2 23–38%. O subcomando
#   'testar' e o DINOv2 saíram a 19/11.)
#   codigo    PASSO DA PIPELINE (depois de sinais.py limites): os sinais de tipo genérico (perigo,
#             obrigação, proibido virar, estacionamento, outro) ganham o código e o nome oficial
#             do desenho mais parecido DENTRO da família do tipo (ex.: perigo -> só códigos A):
#             "Sinal de perigo" -> "Sinal Trânsito nos dois sentidos" (PT:A25)
#
# Uso:
#     python scripts/sinais.py extrair
#     python scripts/sinais.py codigo


# ================================================================== extrair (PDF da DGV)
PDF = ROOT / "pdfs" / "SinalizacaoRodoviaria.pdf"
# nomes que o texto do PDF corta mal (confirmados no RST, Decreto Regulamentar 22-A/98)
CORRECOES = {"C12": "Proibição de inversão do sentido de marcha"}
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
            name = CORRECOES.get(code, name)
            rows.append({"codigo": code, "nome": name, "familia": FAMILIAS.get(code[0], "outros"),
                         "pagina": pi + 1, "ficheiro": f"sinais/{fname}"})
    with open(REF_OUT / "sinais.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print(f"{len(rows)} sinais extraídos -> {REF_OUT / 'sinais.csv'}")
    print("Por família: " + ", ".join(f"{k}: {v}" for k, v in Counter(r["familia"] for r in rows).items()))


# ================================================================== semelhança com os desenhos
MODELOS = {"siglip": "google/siglip-so400m-patch14-384"}  # Apache 2.0
FAMILIAS_RUA = ("A", "B", "C", "D", "E", "G", "H")  # sinais verticais
REF_INNER, CROP_INNER = 0.8, 0.8 / 1.4  # interior do sinal (o recorte da rua tem margem 1,4)
def load_refs():
    with open(REF_OUT / "sinais.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["codigo"].startswith(FAMILIAS_RUA)
                and not r["codigo"].startswith("ET")]
    return [(r["codigo"], r["nome"], Image.open(REF_OUT / r["ficheiro"]).convert("RGB"))
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


# ================================================================== codigo (passo da pipeline)
# Tipo do SigLIP -> códigos possíveis no Guia (o desenho mais parecido é escolhido SÓ entre estes;
# None = qualquer sinal vertical). Tipos com um só código ficam como estão.
PERMITIDOS = {
    "perigo": ("A",), "obrigacao": ("D",), "proibido_virar": ("C11a", "C11b", "C12"),
    "estacionamento": ("H1a", "H1b"), "outro": None,
}


def bank_embeddings(enc, refs, variacoes):
    """Impressões digitais dos desenhos (+ variações), guardadas em data/referencias/."""
    cache = ROOT / "data" / "referencias" / f"banco_{enc.nome}_{variacoes}.npz"
    if cache.exists():
        z = np.load(cache)
        if list(z["refs"]) == [c for c, _, _ in refs]:
            return torch.from_numpy(z["g"]).to(enc.device), list(z["codes"])
    rng = np.random.default_rng(0)
    imgs, codes = [], []
    for code, _, im in refs:
        imgs.append(framed(im))
        codes.append(code)
        for _ in range(variacoes):
            imgs.append(augment(im, rng))
            codes.append(code)
    g, _ = enc(imgs)
    np.savez(cache, g=g.cpu().numpy(), codes=np.array(codes), refs=np.array([c for c, _, _ in refs]))
    return g, codes


def cmd_aplicar(args):
    """Código e nome oficial (Guia da DGV) de cada sinal: owlv2_classified.jsonl ganha
    codigo_rst / nome_rst; os tipos genéricos (perigo, obrigação, ...) passam a mostrar o nome."""
    path = DATA_DIR / "preds" / f"{args.model}_classified.jsonl"
    dets = [json.loads(ln) for ln in open(path, encoding="utf-8")]
    imgs = {r["id"]: r for r in csv.DictReader(open(DATA_DIR / "meta" / "sample_images.csv", encoding="utf-8"))}
    todo = [k for k, d in enumerate(dets) if d["category"] == "sinalizacao" and d.get("subtype") in PERMITIDOS]
    for d in dets:
        for f in ("codigo_rst", "nome_rst", "codigo_sim"):
            d.pop(f, None)
    print(f"Sinais com tipo genérico: {len(todo)} de {sum(d['category'] == 'sinalizacao' for d in dets)}")
    if todo:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        refs = load_refs()
        names = {c: n for c, n, _ in refs}
        pairs = mirror_pairs(refs)
        ref_gray = {c: gray64(im, REF_INNER) for c, _, im in refs}
        enc = Encoder("siglip", device)
        t = time.time()
        bank_g, bank_codes = bank_embeddings(enc, refs, args.variacoes)
        codes_arr = np.array(bank_codes)
        uniq = [str(u) for u in np.unique(codes_arr)]
        cases = [{"image_id": dets[k]["image_id"], "bbox": dets[k]["bbox"],
                  "pano": imgs[dets[k]["image_id"]]["is_pano"] == "True"} for k in todo]
        crops = load_crops(cases)
        g, _ = enc(crops)
        sims = (g @ bank_g.T).cpu().numpy()
        best = np.stack([sims[:, codes_arr == u].max(1) for u in uniq], 1)
        col = {u: i for i, u in enumerate(uniq)}
        n_mud = Counter()
        for j, k in enumerate(todo):
            d = dets[k]
            allow = PERMITIDOS[d["subtype"]]
            if allow is None:
                ok = uniq
            elif all(len(a) == 1 for a in allow):       # família inteira (ex.: "A" = perigo)
                ok = [u for u in uniq if u.startswith(allow)]
            else:
                ok = [u for u in uniq if u in allow]
            sc = np.array([best[j, col[u]] for u in ok])
            top = ok[int(sc.argmax())]
            if top in pairs and pairs[top] in col:     # esquerda/direita: píxel decide se houver dúvida
                other = pairs[top]
                if best[j, col[top]] - best[j, col[other]] < args.duvida:
                    cg = gray64(crops[j], CROP_INNER)
                    top = top if ncc(cg, ref_gray[top]) >= ncc(cg, ref_gray[other]) else other
            if d["subtype"] == "outro" and best[j, col[top]] < args.min_outro:
                continue                                    # "outro" pouco parecido: fica genérico
            d.update({"codigo_rst": top, "nome_rst": names[top], "codigo_sim": round(float(best[j, col[top]]), 3),
                      "codigo": f"PT:{top}", "label": f"Sinal {names[top]}"})
            n_mud[d["subtype"]] += 1
        print(f"Código oficial dado a {sum(n_mud.values())} sinais ({time.time() - t:.0f} s): "
              + ", ".join(f"{k} {v}" for k, v in n_mud.most_common()))
        top_names = Counter(dets[k]["nome_rst"] for k in todo if "nome_rst" in dets[k]).most_common(12)
        print("Mais frequentes: " + " · ".join(f"{n} {c}" for n, c in top_names))
    with open(path, "w", encoding="utf-8") as fo:
        for d in dets:
            fo.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"-> {path}")


def main_desenhos(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("extrair", help="desenhos + códigos dos sinais do PDF da DGV")
    p = sub.add_parser("codigo", help="passo da pipeline: código e nome oficial de cada sinal")
    p.add_argument("--model", default="owlv2")
    p.add_argument("--variacoes", type=int, default=10)
    p.add_argument("--duvida", type=float, default=0.02)
    p.add_argument("--min-outro", type=float, default=0.84,
                   help="tipo 'outro' só recebe nome oficial com semelhança >= isto")
    args = ap.parse_args(argv)
    {"extrair": cmd_extrair, "codigo": cmd_aplicar}[args.cmd](args)


####################################################################################################
SECOES = {
    "limites": (main_limites, "número dos limites de velocidade"),
    "codigo": (lambda a, c="codigo": main_desenhos([c, *a]), "código e nome oficial"),
    "extrair": (lambda a, c="extrair": main_desenhos([c, *a]), "desenhos do PDF da DGV"),
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
