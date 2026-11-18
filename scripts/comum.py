"""
Peças partilhadas por todos os scripts (não se corre sozinho).

  ROOT, DATA_DIR       pasta do projeto e pasta de dados. DATA_DIR vem do ambiente
                       (ex.: $env:DATA_DIR = "data\\piloto_rua") ou do .env; por omissão data/
  load_taxonomy()      config/taxonomia.yaml (classes, frases, regras)
  load_jsonl()         ficheiro .jsonl -> {image_id: [registos]}
  iou(), matches()     sobreposição de caixas / correspondência com uma marcação Mapillary
  nms_by_category()    tira caixas repetidas (várias frases a apontar ao mesmo objeto)
  mapillary_to_subtype classe Mapillary -> nosso subtipo de sinal (SÓ para medir)
  limit_vram()         máx. 95% da memória da placa gráfica
  font(), COLORS       desenho das imagens
"""

from __future__ import annotations

import fnmatch
import json
import os
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
DATA_DIR = Path(os.getenv("DATA_DIR") or (ROOT / "data"))
CATS = ("sinalizacao", "iluminacao", "residuos")
COLORS = {"sinalizacao": (230, 40, 40), "iluminacao": (255, 200, 0), "residuos": (40, 200, 70)}


def load_taxonomy():
    import yaml
    with open(ROOT / "config" / "taxonomia.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path):
    by_img = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            by_img[d["image_id"]].append(d)
    return by_img


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


def matches(pred, ref, img_area):
    """A nossa caixa corresponde à marcação Mapillary? IoU >= 0.3, ou o centro da marcação
    dentro da nossa caixa (a Mapillary marca só a cabeça do candeeiro; nós o poste todo)."""
    pb, rb = pred["bbox"], ref["bbox"]
    if iou(pb, rb) >= 0.3:
        return True
    cx, cy = (rb[0] + rb[2]) / 2, (rb[1] + rb[3]) / 2
    p_area = (pb[2] - pb[0]) * (pb[3] - pb[1])
    return pb[0] <= cx <= pb[2] and pb[1] <= cy <= pb[3] and p_area <= 0.15 * img_area


def nms_by_category(dets, thr=0.5):
    """Vários prompts podem apontar ao mesmo objeto: fica a caixa com maior score."""
    keep = []
    for d in sorted(dets, key=lambda d: -d["score"]):
        if all(k["category"] != d["category"] or iou(k["bbox"], d["bbox"]) < thr for k in keep):
            keep.append(d)
    return keep


def mapillary_to_subtype(tax):
    """Classe Mapillary (ex.: regulatory--stop--g1) -> nosso subtipo, pelos padrões 'mapillary'
    da taxonomia (pela ordem: específicos primeiro, 'outro' no fim)."""
    rules = [(pat, sub["id"]) for sub in tax["categorias"]["sinalizacao"]["subtipos"]
             for pat in sub.get("mapillary", [])]

    def f(value):
        for pat, sid in rules:
            if fnmatch.fnmatch(value, pat):
                return sid
        return None
    return f


def limit_vram(fraction=0.95):
    """Não deixa o processo passar de 95% da VRAM: no Windows, encher a placa faz o driver
    usar a RAM normal e tudo fica muito lento (aconteceu a 17/11). Não acelera; evita isso."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction)


def font(size):
    from PIL import ImageFont
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()
