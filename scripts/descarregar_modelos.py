"""
Descarrega os modelos de IA do projeto (para a cache do Hugging Face) e verifica que
correm nesta máquina. Corre-se uma vez, numa instalação nova.

  OWLv2   google/owlv2-base-patch16-ensemble   Apache 2.0   encontra objetos
  SigLIP  google/siglip-so400m-patch14-384     Apache 2.0   tipo do objeto e número dos limites
  DINOv2  facebook/dinov2-base                 Apache 2.0   semelhança imagem-imagem (em teste)

Verificação do DINOv2: transforma recortes de sinais em "impressões digitais"
(vetores) e mede a semelhança (0 = nada parecido, 1 = igual) entre:
  - o recorte e o mesmo recorte espelhado (esquerda <-> direita);
  - o recorte e sinais de outros tipos.
Se o espelho der quase 1, o DINOv2 sozinho não distingue esquerda de direita
(é preciso o teste do espelho píxel a píxel).

Uso:
    python scripts/descarregar_modelos.py               # todos
    python scripts/descarregar_modelos.py --so dinov2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import torch
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(__file__))
from comum import DATA_DIR  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
MODELOS = {
    "owlv2": "google/owlv2-base-patch16-ensemble",
    "siglip": "google/siglip-so400m-patch14-384",
    "dinov2": "facebook/dinov2-base",
}


def descarregar(nome):
    from huggingface_hub import snapshot_download
    t = time.time()
    path = snapshot_download(MODELOS[nome], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
    size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs) / 1e6
    print(f"{nome:<7} {MODELOS[nome]:<38} {size:>7.0f} MB  ({time.time() - t:.0f} s)")


def sign_crops(n_per_type=3):
    """Alguns recortes de sinais com confiança alta, por tipo."""
    by_type = defaultdict(list)
    with open(DATA_DIR / "preds" / "owlv2_classified.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["category"] == "sinalizacao" and d["confidence"] > 0.6 and len(by_type[d["subtype"]]) < n_per_type:
                x0, y0, x1, y1 = d["bbox"]
                if x1 - x0 >= 50:
                    by_type[d["subtype"]].append(d)
    crops = []
    for sub, ds in by_type.items():
        for d in ds:
            im = Image.open(DATA_DIR / "images" / f"{d['image_id']}.jpg").convert("RGB")
            crops.append((sub, im.crop(tuple(int(v) for v in d["bbox"]))))
    return crops


@torch.inference_mode()
def verificar_dinov2():
    from transformers import AutoImageProcessor, AutoModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(MODELOS["dinov2"])
    model = AutoModel.from_pretrained(MODELOS["dinov2"]).to(device).eval()

    def emb(images):
        x = proc(images=images, return_tensors="pt").to(device)
        e = model(**x).pooler_output  # vetor do token [CLS]
        return e / e.norm(dim=-1, keepdim=True)

    crops = sign_crops()
    if not crops:
        print("Sem recortes de sinais para testar (falta data/preds/owlv2_classified.jsonl).")
        return
    t = time.time()
    e = emb([c for _, c in crops])
    em = emb([ImageOps.mirror(c) for _, c in crops])
    dt = (time.time() - t) / (2 * len(crops)) * 1000
    print(f"\nDINOv2 a correr em {device}: vetor de {e.shape[1]} números por imagem, "
          f"{dt:.0f} ms por imagem")

    subs = [s for s, _ in crops]
    sim = e @ e.T
    mirror = (e * em).sum(-1)
    same = [float(sim[i, j]) for i in range(len(subs)) for j in range(i + 1, len(subs)) if subs[i] == subs[j]]
    diff = [float(sim[i, j]) for i in range(len(subs)) for j in range(i + 1, len(subs)) if subs[i] != subs[j]]
    avg = lambda v: sum(v) / len(v) if v else float("nan")  # noqa: E731
    print(f"Semelhança média (0 = nada parecido, 1 = igual), {len(crops)} recortes:")
    print(f"  sinal vs o mesmo espelhado:  {avg(mirror.tolist()):.2f}")
    print(f"  sinal vs outro do mesmo tipo: {avg(same):.2f}")
    print(f"  sinal vs sinal de outro tipo: {avg(diff):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", nargs="*", choices=list(MODELOS), default=list(MODELOS))
    args = ap.parse_args()
    for nome in args.so:
        descarregar(nome)
    if "dinov2" in args.so:
        verificar_dinov2()


if __name__ == "__main__":
    main()
