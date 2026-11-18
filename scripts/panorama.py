"""
Utilitários para panorâmicas equiretangulares (360°).

Convenção: a coluna central da panorâmica (u = W/2) corresponde a yaw = 0
(direção do compass_angle da imagem). yaw cresce para a direita (sentido horário).
Vistas perspetiva com pitch 0: cobrem latitudes de -fov/2 a +fov/2.
"""

from __future__ import annotations

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
