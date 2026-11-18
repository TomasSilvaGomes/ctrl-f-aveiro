"""
2.ª fase: classifica o SUBTIPO de cada deteção com SigLIP zero-shot sobre o
recorte, usando a taxonomia em config/taxonomia.yaml. Se ganhar uma descrição
"negativa" (árvore, fachada, ...) a deteção é descartada. O número dos limites
de velocidade é lido a seguir, por scripts/sinais.py limites.

Contentores: só 3 tipos (ilha de reciclagem, indiferenciado, subterrâneo),
decididos só pelo SigLIP (sem regras de cor).

Entrada:  data/preds/<modelo>.jsonl         (saída do detetar.py)
Saída:    data/preds/<modelo>_classified.jsonl

Uso:
    python scripts/classificar.py --model owlv2 --det-threshold 0.2
    python scripts/classificar.py --categories residuos   # só refaz os contentores
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
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from comum import DATA_DIR, limit_vram, load_jsonl, load_taxonomy, nms_by_category  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[1]
SIGLIP = "google/siglip-so400m-patch14-384"  # Apache 2.0
CROP = 384
Image.MAX_IMAGE_PIXELS = None


def is_vertical(b):
    return (b[3] - b[1]) > 1.5 * (b[2] - b[0])


def suppress_contained(dets, thr=0.7):
    """Remove caixas da mesma categoria quase contidas noutra (ex.: poste inteiro + só a
    luminária do mesmo poste). Normalmente fica a de score maior; na ILUMINAÇÃO ficam
    primeiro as caixas verticais (o poste completo): a caixa só da luminária tem o fundo
    no ar, a distância pelo chão falha e o objeto perdia-se (29% dos casos, 18/11)."""
    keep = []
    order = sorted(dets, key=lambda d: (not (d["category"] == "iluminacao" and is_vertical(d["bbox"])),
                                        -d["score"]))
    for d in order:
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


class EmbeddingCache:
    """Impressões digitais SigLIP já calculadas, por (foto, caixa). Corridas seguintes só
    calculam os recortes novos (mudar frases ou referências não obriga a ver tudo de novo)."""

    def __init__(self, path):
        self.path, self.new = path, {}
        self.keys, self.emb = {}, None
        if path.exists():
            z = np.load(path, allow_pickle=False)
            if str(z["model"]) == SIGLIP:
                self.emb = z["emb"]
                self.keys = {k: i for i, k in enumerate(z["keys"].tolist())}

    @staticmethod
    def key(iid, bbox):
        return f"{iid}|" + ",".join(str(int(round(v))) for v in bbox)

    def get(self, k):
        if k in self.new:
            return self.new[k]
        i = self.keys.get(k)
        return None if i is None else self.emb[i]

    def save(self):
        if not self.new:
            return
        keys = list(self.keys) + list(self.new)
        new = np.stack(list(self.new.values())).astype(np.float16)
        emb = new if self.emb is None else np.concatenate([self.emb, new])
        np.savez(self.path, model=np.array(SIGLIP), keys=np.array(keys), emb=emb)


class Classifier:
    def __init__(self, tax, device):
        from transformers import AutoModel, AutoProcessor
        limit_vram()
        self.proc = AutoProcessor.from_pretrained(SIGLIP)
        self.model = AutoModel.from_pretrained(SIGLIP, dtype=torch.float16).to(device).eval()
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
    def encode(self, crops):
        """Recortes 384x384 -> impressões digitais normalizadas (na GPU)."""
        # normalização SigLIP (média 0.5, desvio 0.5) feita na GPU
        arr = torch.from_numpy(np.stack([np.asarray(c) for c in crops])).to(self.device)
        pv = ((arr.permute(0, 3, 1, 2).float() / 255 - 0.5) / 0.5).half()
        img = self._tensor(self.model.get_image_features(pixel_values=pv)).float()
        return img / img.norm(dim=-1, keepdim=True)

    @torch.inference_mode()
    def __call__(self, crops, cats):
        return self.classify(self.encode(crops), cats)

    @torch.inference_mode()
    def classify(self, img, cats):
        """Impressões digitais (tensor [N, D]) -> [(subtipo ou None, p_subtipo, p_negativo)]."""
        img = torch.as_tensor(img, device=self.device).float()
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
    cache = EmbeddingCache(DATA_DIR / "preds" / "siglip_cache.npz")
    missing = {iid: [d for d in ds if cache.get(cache.key(iid, d["bbox"])) is None]
               for iid, ds in work.items()}
    missing = {iid: ds for iid, ds in missing.items() if ds}
    n_miss = sum(len(v) for v in missing.values())
    print(f"Já calculados (cache): {n_dets - n_miss} · a calcular: {n_miss} recortes em {len(missing)} fotos")

    def crops_for(iid):
        im = Image.open(DATA_DIR / "images" / f"{iid}.jpg").convert("RGB")
        return iid, [square_crop(im, p["bbox"])[0] for p in missing[iid]]

    # 1) impressões digitais só dos recortes novos (é aqui que a GPU trabalha)
    t0 = time.time()
    pend, last = [], (list(missing)[-1] if missing else None)
    with ThreadPoolExecutor(args.threads) as pool:
        for iid, crops in tqdm(pool.map(crops_for, list(missing)), total=len(missing), unit="img"):
            pend += [(cache.key(iid, d["bbox"]), c) for d, c in zip(missing[iid], crops)]
            while len(pend) >= args.batch or (pend and iid == last):
                chunk, pend = pend[: args.batch], pend[args.batch:]
                emb = clf.encode([c for _, c in chunk]).half().cpu().numpy()
                cache.new.update({k: e for (k, _), e in zip(chunk, emb)})
    cache.save()

    out_path = DATA_DIR / "preds" / f"{args.model}_classified.jsonl"
    stats = Counter()

    def flush(f, items):
        emb = np.stack([cache.get(cache.key(iid, det["bbox"])) for iid, det in items])
        res = clf.classify(emb, [det["category"] for _, det in items])
        for (iid, det), (sub, p_sub, p_neg) in zip(items, res):
            if sub is None:
                stats["descartadas_negativo"] += 1
                continue
            info = subinfo[(det["category"], sub)]
            label, value = info["label"], None
            if sub == "limite_velocidade":  # o número é lido depois (sinais.py limites)
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

    # 2) classificar todos (frases contra impressões digitais: rápido)
    items = [(iid, d) for iid, ds in work.items() for d in ds]
    with out_path.open("w", encoding="utf-8") as f:
        f.writelines(previous)
        for i in range(0, len(items), 1024):
            flush(f, items[i:i + 1024])

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
