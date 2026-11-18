"""
Cor de um contentor (regras HSV, sem IA, determinístico).

Em Aveiro (verificado nas imagens):
  - ecopontos   = "iglus" azul-escuro; a FRAÇÃO vê-se pela cor da abertura
                  (amarelo = embalagens, verde = vidro, azul = papel)
  - contentores com rodas: preto/cinzento/verde-escuro = indiferenciado;
                  tampa amarela/azul = embalagens/papel; castanho = biorresíduos
  - subterrâneos = colunas cinzentas (metal/betão) com tampa escura

body_colour(crop)  -> "azul" | "escuro" | "cinzento_claro" | "verde" | "castanho" | "outro"
openings(crop)     -> frações com abertura/tampa visível, ex. ["embalagens", "vidro"]
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def _hsv(crop: Image.Image):
    a = np.asarray(crop.convert("HSV"), dtype=np.float32)
    return a[..., 0] * (360 / 255), a[..., 1] / 255, a[..., 2] / 255


def _centre(crop: Image.Image, frac=0.6):
    w, h = crop.size
    dx, dy = w * (1 - frac) / 2, h * (1 - frac) / 2
    return crop.crop((int(dx), int(dy), int(w - dx), int(h - dy)))


def body_stats(crop: Image.Image):
    """Fração de píxeis de cada cor no centro do recorte (corpo do contentor).
    Limiares calibrados em recortes de Aveiro: iglus azul-escuro têm S 0,39–0,67;
    contentores pretos têm reflexo azulado mas S <= 0,33."""
    H, S, V = _hsv(_centre(crop))
    n = H.size or 1
    blue = (H >= 195) & (H <= 250) & (S >= 0.36) & (V >= 0.10)
    green = (H >= 75) & (H <= 165) & (S >= 0.30) & (V >= 0.15)
    brown = (H >= 10) & (H <= 40) & (S >= 0.40) & (V >= 0.12) & (V <= 0.55)
    other = blue | green | brown
    light = ~other & (S < 0.20) & (V >= 0.55)
    dark = ~other & ~light & (V < 0.55)
    return {"azul": blue.sum() / n, "verde": green.sum() / n, "castanho": brown.sum() / n,
            "escuro": dark.sum() / n, "cinzento_claro": light.sum() / n}


def body_colour(crop: Image.Image, blue_frac=0.30, min_frac=0.25):
    """Azul ganha a partir de 30% dos píxeis (calibrado: iglus mediana 82%,
    contentores pretos mediana 0%, percentil 75 = 4%); senão, a cor maioritária."""
    s = body_stats(crop)
    if s["azul"] >= blue_frac:
        return "azul", s
    best = max(s, key=s.get)
    return (best if s[best] >= min_frac else "outro"), s


def openings(crop: Image.Image, min_frac=0.004):
    """Aberturas/tampas de cor viva (todo o recorte): amarelo, verde vivo."""
    H, S, V = _hsv(crop)
    n = H.size or 1
    yellow = (H >= 38) & (H <= 70) & (S >= 0.45) & (V >= 0.45)
    green = (H >= 85) & (H <= 160) & (S >= 0.40) & (V >= 0.35)
    out = []
    if yellow.sum() / n >= min_frac:
        out.append("embalagens")
    if green.sum() / n >= min_frac:
        out.append("vidro")
    return out
