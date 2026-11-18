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
# azul #0072B2 · âmbar #E69F00 · roxo #CC79A7 (paleta Okabe-Ito, segura para daltónicos)
COLORS = {"sinalizacao": (0, 114, 178), "iluminacao": (230, 159, 0), "residuos": (204, 121, 167)}


def ink(color):
    """Cor do texto por cima de uma cor de fundo: branco nas escuras, preto nas claras."""
    r, g, b = color[:3]
    return (255, 255, 255) if 0.299 * r + 0.587 * g + 0.114 * b < 140 else (0, 0, 0)


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


def limiares(texto="", base=None):
    """Confiança mínima por categoria: config/taxonomia.yaml (confianca_minima); base (se dada)
    substitui todas; texto "iluminacao=0.7,residuos=0.5" substitui as que nomeia."""
    conf = load_taxonomy().get("confianca_minima", {})
    out = {k: (base if base is not None else conf.get(k, 0.3)) for k in ("sinalizacao", "iluminacao", "residuos")}
    for par in filter(None, (texto or "").split(",")):
        k, v = par.split("=")
        out[k.strip()] = float(v)
    return out


####################################################################################################
# PANORÂMICAS E ROTAÇÕES (antes: scripts/panorama.py)
####################################################################################################
# Utilitários para panorâmicas equiretangulares (360°).
#
# Convenção: a coluna central da panorâmica (u = W/2) corresponde a yaw = 0
# (direção do compass_angle da imagem). yaw cresce para a direita (sentido horário).
# Vistas perspetiva com pitch 0: cobrem latitudes de -fov/2 a +fov/2.
import math
from functools import lru_cache

import cv2
import numpy as np


def _focal(size: int, fov_deg: float) -> float:
    return (size / 2) / math.tan(math.radians(fov_deg) / 2)


@lru_cache(maxsize=64)
def _maps(W: int, H: int, yaw_deg: float, fov_deg: float, size: int):
    f = _focal(size, fov_deg)
    c = (np.arange(size, dtype=np.float32) + 0.5 - size / 2) / f
    x, y = np.meshgrid(c, c)  # x para a direita, y para baixo, z = 1
    lon = math.radians(yaw_deg) + np.arctan(x)
    lat = -np.arctan2(y, np.sqrt(x * x + 1))
    u = ((lon / (2 * math.pi) + 0.5) % 1.0) * W
    v = (0.5 - lat / math.pi) * H
    return u.astype(np.float32), v.astype(np.float32)


def perspective_view(pano: np.ndarray, yaw_deg: float, fov_deg: float = 90, size: int = 960):
    """Recorta uma vista perspetiva (size x size) da panorâmica (array HxWx3)."""
    H, W = pano.shape[:2]
    mx, my = _maps(W, H, float(yaw_deg), float(fov_deg), size)
    return cv2.remap(pano, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def view_box_to_pano(box, W: int, H: int, yaw_deg: float, fov_deg: float = 90, size: int = 960):
    """Converte uma caixa da vista para coordenadas da panorâmica.
    Devolve None se a caixa atravessar a costura (u=0/W)."""
    f = _focal(size, fov_deg)
    x0, y0, x1, y1 = box
    n = 12
    xs = np.concatenate([np.linspace(x0, x1, n), np.full(n, x1), np.linspace(x1, x0, n), np.full(n, x0)])
    ys = np.concatenate([np.full(n, y0), np.linspace(y0, y1, n), np.full(n, y1), np.linspace(y1, y0, n)])
    x = (xs - size / 2) / f
    y = (ys - size / 2) / f
    lon = math.radians(yaw_deg) + np.arctan(x)
    lat = -np.arctan2(y, np.sqrt(x * x + 1))
    u = (lon / (2 * math.pi) + 0.5) * W  # sem módulo: coordenada "desenrolada"
    v = (0.5 - lat / math.pi) * H
    u0, u1 = float(u.min()), float(u.max())
    if u1 <= 0:
        u0, u1 = u0 + W, u1 + W
    elif u0 >= W:
        u0, u1 = u0 - W, u1 - W
    if u0 < 0 or u1 > W:
        return None
    return [round(u0, 1), round(float(v.min()), 1), round(u1, 1), round(float(v.max()), 1)]


def pixel_to_bearing(u: float, W: int, compass_angle: float) -> float:
    """Rumo (graus, 0 = norte) de uma coluna da panorâmica."""
    return (compass_angle + (u / W - 0.5) * 360) % 360


def rodrigues(v):
    """Vetor ângulo-eixo (computed_rotation da Mapillary) -> matriz de rotação 3x3."""
    v = np.asarray(v, float)
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K
