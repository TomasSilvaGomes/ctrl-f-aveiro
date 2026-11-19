"""
2.ª fase: classifica o SUBTIPO de cada deteção com SigLIP zero-shot sobre o
recorte, usando a taxonomia em config/taxonomia.yaml. Se ganhar uma descrição
"negativa" (árvore, fachada, ...) a deteção é descartada. O número dos limites
de velocidade é lido a seguir, por scripts/ler_limites.py.

Contentores: a ilha de reciclagem pode ser classificada por fração (azul/papel,
amarelo/embalagens, verde/vidro); os restantes tipos são decididos pelo SigLIP.

Entrada:  data/preds/<modelo>.jsonl         (saída do run_detector.py)
Saída:    data/preds/<modelo>_classified.jsonl

Uso:
    python scripts/classify_subtype.py --model owlv2 --det-threshold 0.2
    python scripts/classify_subtype.py --categories residuos   # só refaz os contentores
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from estimate_mapillary import DATA_DIR  # noqa: E402
from evaluate import load_jsonl, nms_by_category  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SIGLIP = "google/siglip-so400m-patch14-384"  # Apache 2.0
CROP = 384


def load_taxonomy():
    with open(ROOT / "config" / "taxonomia.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def suppress_contained(dets, thr=0.7):
    """Remove caixas da mesma categoria quase contidas noutra com score maior
    (ex.: poste inteiro + só a parte de cima do mesmo poste)."""
    keep = []
    for d in sorted(dets, key=lambda d: -d["score"]):
        a = d["bbox"]
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        dup = False
        for k in keep:
            if k["category"] != d["category"]:
                continue
            b = k["bbox"]
            ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
            iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            small = min(area_a, (b[2] - b[0]) * (b[3] - b[1]))
            if small and ix * iy / small >= thr:
                dup = True
                break
        if not dup:
            keep.append(d)
    return keep


def square_crop(im: Image.Image, box, pad=0.15):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    x0, x1 = max(0, x0 - pad * w), min(im.width, x1 + pad * w)
    y0, y1 = max(0, y0 - pad * h), min(im.height, y1 + pad * h)
    c = im.crop((int(x0), int(y0), int(x1), int(y1)))
    side = max(c.size)
    canvas = Image.new("RGB", (side, side), (127, 127, 127))
    canvas.paste(c, ((side - c.width) // 2, (side - c.height) // 2))
    return canvas.resize((CROP, CROP), Image.BICUBIC), c


class Classifier:
    def __init__(self, tax, device):
        from transformers import AutoModel, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(SIGLIP)
        self.dtype = torch.float16 if torch.device(device).type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(SIGLIP, dtype=self.dtype).to(device).eval()
        self.device = device
        self.tables = {}  # categoria -> (embeddings de texto, [(subtipo|None, ...)])
        for cat, spec in tax["categorias"].items():
            texts, owners = [], []
            for sub in spec["subtipos"]:
                for p in sub["prompts"]:
                    texts.append(f"This is a photo of {p}.")
                    owners.append(sub["id"])
            for neg in tax["negativos"]:
                texts.append(f"This is a photo of {neg}.")
                owners.append(None)
            self.tables[cat] = (self._text(texts), owners)
        self.scale = self.model.logit_scale.exp().float()
        self.bias = self.model.logit_bias.float()

    @staticmethod
    def _tensor(x):
        return x if torch.is_tensor(x) else x.pooler_output

    @torch.inference_mode()
    def _text(self, texts):
        tok = self.proc(text=texts, padding="max_length", return_tensors="pt").to(self.device)
        t = self._tensor(self.model.get_text_features(**tok)).float()
        return t / t.norm(dim=-1, keepdim=True)

    @torch.inference_mode()
    def __call__(self, crops, cats):
        # recortes já vêm 384x384: normalização SigLIP (média 0.5, desvio 0.5) na GPU
        arr = torch.from_numpy(np.stack([np.asarray(c) for c in crops])).to(self.device)
        pv = ((arr.permute(0, 3, 1, 2).float() / 255 - 0.5) / 0.5).to(self.dtype)
        img = self._tensor(self.model.get_image_features(pixel_values=pv)).float()
        img = img / img.norm(dim=-1, keepdim=True)
        out = []
        for e, cat in zip(img, cats):
            txt, owners = self.tables[cat]
            logits = e @ txt.T * self.scale + self.bias
            probs = torch.softmax(logits, dim=-1)
            agg = defaultdict(float)  # soma das probabilidades por subtipo
            for p, o in zip(probs.tolist(), owners):
                agg[o] += p
            best = max(agg, key=agg.get)
            out.append((best, agg[best], agg.get(None, 0.0)))
        return out


def main():
    Image.MAX_IMAGE_PIXELS = None
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="owlv2")
    ap.add_argument("--det-threshold", type=float, default=0.2)
    ap.add_argument("--min-px", type=float, default=16, help="ignora caixas com lado maior menor")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 8,
                    help="threads a abrir/recortar fotos (o limite costuma ser a GPU)")
    ap.add_argument("--limit", type=int, default=0, help="n.º de imagens (0 = todas)")
    ap.add_argument("--categories", nargs="*", default=None,
                    help="só reclassifica estas categorias e mantém as outras do ficheiro anterior")
    args = ap.parse_args()

    tax = load_taxonomy()
    subinfo = {(c, s["id"]): s for c, spec in tax["categorias"].items() for s in spec["subtipos"]}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    preds = load_jsonl(DATA_DIR / "preds" / f"{args.model}.jsonl")
    ids = sorted(preds)
    if args.limit:
        ids = ids[: args.limit]
    work = {}
    for iid in ids:
        dets = [p for p in preds[iid] if p["score"] >= args.det_threshold]
        kept = nms_by_category(dets)
        kept = [p for p in kept if max(p["bbox"][2] - p["bbox"][0], p["bbox"][3] - p["bbox"][1]) >= args.min_px]
        kept = suppress_contained(kept)
        if args.categories:
            kept = [p for p in kept if p["category"] in args.categories]
        if kept:
            work[iid] = kept
    n_dets = sum(len(v) for v in work.values())
    print(f"{len(work)} imagens, {n_dets} deteções (score >= {args.det_threshold}, após NMS)")

    clf = Classifier(tax, device)

    def crops_for(iid):
        im = Image.open(DATA_DIR / "images" / f"{iid}.jpg").convert("RGB")
        return iid, [square_crop(im, p["bbox"]) for p in work[iid]]

    out_path = DATA_DIR / "preds" / f"{args.model}_classified.jsonl"
    stats, t0 = Counter(), time.time()
    queue = []  # (iid, det, crop_quadrado, crop_original)

    def flush(f, items):
        res = clf([q[2] for q in items], [q[1]["category"] for q in items])
        for (iid, det, _, raw), (sub, p_sub, p_neg) in zip(items, res):
            if sub is None:
                stats["descartadas_negativo"] += 1
                continue
            info = subinfo[(det["category"], sub)]
            label, value = info["label"], None
            if sub == "limite_velocidade":  # o número é lido depois (ler_limites.py)
                label = label.replace(" {n} km/h", "")
            cat_label = tax["categorias"][det["category"]]["label"]
            rec = {"image_id": iid, "bbox": det["bbox"], "category": det["category"],
                   "category_label": cat_label, "subtype": sub, "label": label,
                   "codigo": info.get("codigo"), "valor": value,
                   "det_score": det["score"], "sub_score": round(p_sub, 4),
                   # confiança global = confiança da deteção (score OWLv2 normalizado:
                   # >= 0.5 conta como certeza) x confiança do subtipo (SigLIP)
                   "confidence": round(min(1.0, det["score"] / 0.5) * p_sub, 4),
                   "det_prompt": det["prompt"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats[(det["category"], sub)] += 1

    previous = []  # registos das outras categorias (modo --categories)
    if args.categories and out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            previous = [ln for ln in f if json.loads(ln)["category"] not in args.categories]
        print(f"Mantidos {len(previous)} registos de outras categorias")

    with out_path.open("w", encoding="utf-8") as f, ThreadPoolExecutor(args.threads) as pool:
        f.writelines(previous)
        for iid, crops in tqdm(pool.map(crops_for, list(work)), total=len(work), unit="img"):
            for det, (sq, raw) in zip(work[iid], crops):
                queue.append((iid, det, sq, raw))
            while len(queue) >= args.batch:
                flush(f, queue[: args.batch])
                queue = queue[args.batch:]
        if queue:
            flush(f, queue)

    dt = time.time() - t0
    print(f"\nFeito em {dt:.0f}s -> {out_path}")
    print(f"Descartadas (não-mobiliário): {stats.pop('descartadas_negativo', 0)}")
    by_cat = defaultdict(list)
    for (c, s), n in stats.items():
        by_cat[c].append((n, s))
    for c, items in by_cat.items():
        items.sort(reverse=True)
        print(f"  {c}: " + ", ".join(f"{s} {n}" for n, s in items))


if __name__ == "__main__":
    main()
