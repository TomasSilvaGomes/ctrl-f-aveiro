"""
Lê o número dos sinais de limite de velocidade (passo entre classify_subtype e geolocate).

Para cada sinal "limite_velocidade":
  1. "print" do sinal: vista normal centrada nele, tirada da 360° com a resolução
     original (sem a deformação da 360°), ampliada 4x. Nas fotos normais, recorte ampliado.
  2. SigLIP lê o número: compara o print com frases "sinal de limite com o número N"
     e escolhe a mais parecida. "valor_conf" = confiança dessa escolha (0–1);
     abaixo de --rever (0.6) o sinal fica marcado para o técnico confirmar.

Teste a 17/11 (64 sinais com o número confirmado por pontos Mapillary):
  OCR (EasyOCR) na foto 31% certos · OCR no print 55% · SigLIP no print 97%.
  Por isso o OCR foi retirado do projeto. A medição repete-se com analisa_erros.py.

Entrada/saída:  data/preds/<modelo>_classified.jsonl (atualizado no lugar)

Uso:
    python scripts/ler_limites.py --model owlv2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter

import cv2
import numpy as np
import torch
import yaml
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from estimate_mapillary import DATA_DIR  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

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
    def __init__(self, device):
        from transformers import AutoModel, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(SIGLIP)
        self.model = AutoModel.from_pretrained(SIGLIP, dtype=torch.float16).to(device).eval()
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
        pv = self.proc(images=im, return_tensors="pt")["pixel_values"].to(self.device).half()
        e = self._t(self.model.get_image_features(pixel_values=pv))
        p = torch.softmax((e / e.norm(dim=-1, keepdim=True)) @ self.text.T * self.model.logit_scale.exp().float(), -1)[0]
        i = int(p.argmax())
        return SPEEDS[i], float(p[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="owlv2")
    ap.add_argument("--rever", type=float, default=0.6, help="abaixo desta confiança: a rever")
    args = ap.parse_args()

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


if __name__ == "__main__":
    main()
