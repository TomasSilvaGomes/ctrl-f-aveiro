"""
Corre um detetor zero-shot sobre a amostra e grava as deteções em
data/preds/<modelo>.jsonl  (image_id, prompt, category, subtype, score, bbox).

- Imagens normais (frontais): a foto inteira em quadrado 960x960 + metade esquerda e
  direita (60% da largura, sobreposição de 20%), para apanhar sinais pequenos ao fundo
  (tiling; desligar com --sem-tiling).
- Panorâmicas 360°: 6 vistas perspetiva de 90° (yaw = 30°, 90°, ..., 330°),
  deteção em cada vista e conversão das caixas para coordenadas da panorâmica.
- Pré-processamento feito por nós (o pré-processador do OWLv2 é muito lento).

Modelos:
    owlv2   google/owlv2-base-patch16-ensemble   (Apache 2.0)

Uso:
    python scripts/detetar.py --model owlv2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from comum import DATA_DIR, limit_vram, load_taxonomy  # noqa: E402
from comum import perspective_view, view_box_to_pano  # noqa: E402

def load_prompts():
    """Frases do detetor (config/taxonomia.yaml, secção 'detetor'): frase -> (categoria, '')."""
    return {p: (cat, "") for cat, ps in load_taxonomy()["detetor"].items() for p in ps}


# prompt de texto -> (categoria, subtipo provisório; o subtipo é decidido pelo SigLIP)
PROMPTS = load_prompts()
MODELS = {"owlv2": "google/owlv2-base-patch16-ensemble"}
SIZE = 960
FOV = 90
YAWS = [30, 90, 150, 210, 270, 330]
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


TILE_W, TILE_OVERLAP = 0.6, 0.2   # frontais: metades com 60% da largura (20% de sobreposição)


def letterbox(im, x0=0):
    """Imagem (array) -> quadrado 960x960 (cinzento à volta) + info para desfazer."""
    H, W = im.shape[:2]
    s = SIZE / max(W, H)
    small = np.asarray(Image.fromarray(im).resize((round(W * s), round(H * s)), Image.BILINEAR))
    canvas = np.full((SIZE, SIZE, 3), 127, dtype=np.uint8)
    canvas[: small.shape[0], : small.shape[1]] = small
    return canvas, s, x0


def prepare(row, tiling=True):
    """Lê a imagem e devolve uma lista de vistas 960x960 + info para desfazer.
    360°: 6 vistas de 90°. Frontal: a foto inteira + (tiling) metade esquerda e direita
    com sobreposição, para os sinais pequenos ao fundo terem mais resolução."""
    im = Image.open(DATA_DIR / "images" / row["file"]).convert("RGB")
    return row, prepare_views(im, panoramic=row["is_pano"] == "True", tiling=tiling)


def prepare_views(image, *, panoramic=False, tiling=True):
    """Imagem (PIL ou array RGB) já em memória -> [(vista 960x960, info)]. Usado pela pipeline
    (prepare) e pela app (fotos enviadas)."""
    im = np.asarray(image)
    H, W = im.shape[:2]
    views = []
    if panoramic:
        for yaw in YAWS:
            views.append((perspective_view(im, yaw, FOV, SIZE), {"pano": True, "W": W, "H": H, "yaw": yaw}))
    else:
        parts = [(0, W)]
        if tiling:
            tw = int(W * TILE_W)
            parts += [(0, tw), (W - tw, W)]
        for a, b in parts:
            canvas, s, ox = letterbox(im[:, a:b], a)
            views.append((canvas, {"pano": False, "W": W, "H": H, "scale": s, "ox": ox}))
    return views


def to_original(box, info):
    if info["pano"]:
        return view_box_to_pano(box, info["W"], info["H"], info["yaw"], FOV, SIZE)
    s, ox = info["scale"], info.get("ox", 0)
    x0, y0, x1, y1 = (b / s for b in box)
    x0, x1 = x0 + ox, x1 + ox
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(info["W"], x1), min(info["H"], y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)]


class Owlv2:
    def __init__(self, device):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        name = MODELS["owlv2"]
        limit_vram()  # máx. 95% da VRAM (evita o Windows usar a RAM normal)
        self.proc = Owlv2Processor.from_pretrained(name)
        # meia precisão só na placa gráfica (no processador o float16 é lento ou não existe)
        self.dtype = torch.float16 if torch.device(device).type == "cuda" else torch.float32
        self.model = Owlv2ForObjectDetection.from_pretrained(name, dtype=self.dtype).to(device).eval()
        self.device = device
        tok = self.proc(text=[list(PROMPTS)], return_tensors="pt")
        self.ids, self.mask = tok["input_ids"].to(device), tok["attention_mask"].to(device)
        self.mean, self.std = MEAN.to(device), STD.to(device)

    @torch.inference_mode()
    def __call__(self, arrays, threshold):
        x = torch.from_numpy(np.stack(arrays)).to(self.device).permute(0, 3, 1, 2).float() / 255
        x = ((x - self.mean) / self.std).to(self.dtype)
        B = x.shape[0]
        out = self.model(input_ids=self.ids.repeat(B, 1), attention_mask=self.mask.repeat(B, 1),
                         pixel_values=x)
        res = self.proc.image_processor.post_process_object_detection(
            out, threshold=threshold, target_sizes=[(SIZE, SIZE)] * B)
        prompts = list(PROMPTS)
        return [[(b, s, prompts[lab]) for b, s, lab in
                 zip(r["boxes"].tolist(), r["scores"].tolist(), r["labels"].tolist())] for r in res]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="owlv2")
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=12, help="vistas por lote na GPU")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 8,
                    help="threads a abrir/recortar fotos (o limite costuma ser a GPU)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sem-tiling", action="store_true", help="frontais: só a foto inteira (como antes)")
    ap.add_argument("--apenas-frontais", action="store_true",
                    help="refaz só as fotos frontais e mantém as deteções das 360° do ficheiro anterior")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")
    with open(DATA_DIR / "meta" / "sample_images.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]
    out_path = DATA_DIR / "preds" / f"{args.model}.jsonl"
    keep_lines = []
    if args.apenas_frontais:
        pano_ids = {r["id"] for r in rows if r["is_pano"] == "True"}
        if out_path.exists():
            with out_path.open(encoding="utf-8") as f:
                keep_lines = [ln for ln in f if json.loads(ln)["image_id"] in pano_ids]
        rows = [r for r in rows if r["is_pano"] != "True"]
        print(f"Só frontais: {len(rows)} fotos (mantidas {len(keep_lines)} deteções das 360°)")

    det = Owlv2(device)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0, n_det, n_views, dropped = time.time(), 0, 0, 0
    pending = []  # (row, info, array)

    def flush(f, items):
        nonlocal n_det, dropped
        results = det([a for _, _, a in items], args.threshold)
        for (row, info, _), dets in zip(items, results):
            for box, score, prompt in dets:
                ob = to_original(box, info)
                if ob is None:
                    dropped += 1
                    continue
                cat, sub = PROMPTS[prompt]
                f.write(json.dumps({"image_id": row["id"], "prompt": prompt, "category": cat,
                                    "subtype": sub, "score": round(score, 4), "bbox": ob}) + "\n")
                n_det += 1

    from functools import partial
    with out_path.open("w", encoding="utf-8") as f, ThreadPoolExecutor(args.threads) as pool:
        f.writelines(keep_lines)
        prep = partial(prepare, tiling=not args.sem_tiling)
        for row, views in tqdm(pool.map(prep, rows), total=len(rows), unit="img"):
            for arr, info in views:
                pending.append((row, info, arr))
                n_views += 1
            while len(pending) >= args.batch:
                flush(f, pending[: args.batch])
                pending = pending[args.batch:]
        if pending:
            flush(f, pending)

    dt = time.time() - t0
    print(f"{len(rows)} imagens ({n_views} vistas) em {dt:.0f}s "
          f"({len(rows) / dt:.2f} img/s, {n_views / dt:.1f} vistas/s), {n_det} deteções "
          f"({dropped} descartadas na costura) -> {out_path}")


if __name__ == "__main__":
    main()
