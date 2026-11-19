"""
Corre um detetor zero-shot sobre a amostra e grava as deteções em
data/preds/<modelo>.jsonl  (image_id, prompt, category, subtype, score, bbox).

- Imagens normais: letterbox para quadrado 960x960.
- Panorâmicas 360°: 6 vistas perspetiva de 90° (yaw = 30°, 90°, ..., 330°),
  deteção em cada vista e conversão das caixas para coordenadas da panorâmica.
- Pré-processamento feito por nós (o pré-processador do OWLv2 é muito lento).

Modelos:
    owlv2   google/owlv2-base-patch16-ensemble   (Apache 2.0)

Uso:
    python scripts/run_detector.py --model owlv2
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
from estimate_mapillary import DATA_DIR  # noqa: E402
from panorama import perspective_view, view_box_to_pano  # noqa: E402

def load_prompts():
    """Frases do detetor (config/taxonomia.yaml, secção 'detetor'): frase -> (categoria, '')."""
    import yaml
    from pathlib import Path
    with open(Path(__file__).resolve().parents[1] / "config" / "taxonomia.yaml", encoding="utf-8") as f:
        det = yaml.safe_load(f)["detetor"]
    return {p: (cat, "") for cat, ps in det.items() for p in ps}


# prompt de texto -> (categoria, subtipo provisório; o subtipo é decidido pelo SigLIP)
PROMPTS = load_prompts()
MODELS = {"owlv2": "google/owlv2-base-patch16-ensemble"}
SIZE = 960
FOV = 90
YAWS = [30, 90, 150, 210, 270, 330]
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def prepare(row):
    """Lê a imagem e devolve uma lista de vistas 960x960 + info para desfazer."""
    im = np.asarray(Image.open(DATA_DIR / "images" / row["file"]).convert("RGB"))
    H, W = im.shape[:2]
    views = []
    if row["is_pano"] == "True":
        for yaw in YAWS:
            views.append((perspective_view(im, yaw, FOV, SIZE), {"pano": True, "W": W, "H": H, "yaw": yaw}))
    else:
        views.append(prepare_flat(Image.fromarray(im)))
    return row, views


def prepare_flat(image):
    width, height = image.size
    scale = SIZE / max(width, height)
    size = max(1, round(width * scale)), max(1, round(height * scale))
    small = np.asarray(image.resize(size, Image.Resampling.BILINEAR))
    canvas = np.full((SIZE, SIZE, 3), 127, dtype=np.uint8)
    canvas[:size[1], :size[0]] = small
    return canvas, {"pano": False, "W": width, "H": height, "scale": scale}


def to_original(box, info):
    if info["pano"]:
        return view_box_to_pano(box, info["W"], info["H"], info["yaw"], FOV, SIZE)
    s = info["scale"]
    x0, y0, x1, y1 = (b / s for b in box)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(info["W"], x1), min(info["H"], y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)]


class Owlv2:
    def __init__(self, device):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        name = MODELS["owlv2"]
        self.proc = Owlv2Processor.from_pretrained(name)
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
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")
    with open(DATA_DIR / "meta" / "sample_images.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    det = Owlv2(device)
    out_path = DATA_DIR / "preds" / f"{args.model}.jsonl"
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

    with out_path.open("w", encoding="utf-8") as f, ThreadPoolExecutor(args.threads) as pool:
        for row, views in tqdm(pool.map(prepare, rows), total=len(rows), unit="img"):
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
