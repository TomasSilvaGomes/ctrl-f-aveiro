"""Pictogramas dos sinais classificados pelo catálogo."""

from __future__ import annotations

import re


RED = "#cf2943"
BLUE = "#1762b8"
INK = "#182433"
WHITE = "#fff"
ROUND = 'stroke-linecap="round" stroke-linejoin="round"'
STRAIGHT_ARROW = '<path fill="#fff" d="M12 5.5 18 12h-3.6v6.5H9.6V12H6z"/>'


SIGN_CODES = {
    "B1": "cedencia",
    "B2": "stop",
    "C1": "sentido_proibido",
    "C13": "limite_velocidade",
    "C15": "estacionamento_proibido",
    "C16": "paragem_estacionamento_proibidos",
    "H1": "estacionamento",
    "H1a": "estacionamento",
    "H7": "passagem_peoes",
}


def circle(fill, stroke=None):
    border = f' stroke="{stroke}" stroke-width="2.5"' if stroke else ""
    return f'<circle cx="12" cy="12" r="{9 if stroke else 10.25}" fill="{fill}"{border}/>'


def line(path, color, width=2.2):
    return f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{width}" {ROUND}/>'


def text(value, fill, size):
    return (
        f'<text x="12" y="12" dy=".34em" fill="{fill}" font-family="system-ui,sans-serif" '
        f'font-weight="800" font-size="{size}" text-anchor="middle">{value}</text>'
    )


def square(fill):
    return f'<rect x="2" y="2" width="20" height="20" rx="2.4" fill="{fill}"/>'


SIGN_DRAWINGS = {
    "stop": (
        f'<path fill="{RED}" d="M8 1.8h8L22.2 8v8L16 22.2H8L1.8 16V8z"/>'
        + text("STOP", WHITE, 6.25)
    ),
    "cedencia": (
        f'<path d="M2.5 4.5h19L12 21z" fill="{WHITE}" stroke="{RED}" stroke-width="2.65" {ROUND}/>'
    ),
    "sentido_proibido": circle(RED) + '<rect x="5" y="10.2" width="14" height="3.6" rx=".6" fill="#fff"/>',
    "estacionamento_proibido": circle(BLUE, RED) + line("M5.8 5.8 18.2 18.2", RED, 2.5),
    "paragem_estacionamento_proibidos": circle(BLUE, RED) + line("M5.8 5.8 18.2 18.2m0-12.4L5.8 18.2", RED, 2.5),
    "proibido_virar": (
        circle(WHITE, RED)
        + line("M16 16v-3a3.5 3.5 0 0 0-3.5-3.5H8m2.5-2.5L8 9.5l2.5 2.5", INK, 2.05)
        + line("M5.8 18.2 18.2 5.8", RED, 2.5)
    ),
    "obrigacao": circle(BLUE) + STRAIGHT_ARROW,
    "sentido_unico": square(BLUE) + STRAIGHT_ARROW,
    "passagem_peoes": (
        square(BLUE) + '<path fill="#fff" stroke-linejoin="round" d="M12 5.1 20 19H4z"/>'
        f'<circle cx="13" cy="9.4" r="1.1" fill="{INK}"/>'
        + line("m9.4 13.8 1.5-2.1 1.7-.8 1.4 2 1.5.2m-3-2.2-1.1 3.4-1.5 2.1m1.5-2.1 2.2.7 1.2 1.5", INK, 1.3)
        + line("M7.2 17.7h1.35m1.55 0h1.35m1.55 0h1.35m1.55 0h1.35", INK, .9)
    ),
    "estacionamento": square(BLUE) + '<path fill="#fff" d="M8.5 5.6h4.3a4.2 4.2 0 1 1 0 8.4h-1.3v4.5h-3zm3 2.8v2.8h1.3a1.4 1.4 0 1 0 0-2.8z"/>',
    "perigo": (
        f'<path d="M12 3 22 20.5H2z" fill="{WHITE}" stroke="{RED}" stroke-width="2.5" {ROUND}/>'
        + line("M12 9.5v4.5", INK)
        + f'<circle cx="12" cy="17" r="1.05" fill="{INK}"/>'
    ),
    "direcao": (
        '<path fill="#165a98" d="M2.5 6.2h14.1L22 12l-5.4 5.8H2.5a1 1 0 0 1-1-1V7.2a1 1 0 0 1 1-1z"/>'
        + line("M6.4 12h10.3m-3-3 3 3-3 3", WHITE, 1.8)
    ),
    "outro": (
        '<rect x="5" y="3" width="14" height="13" rx="2.7" fill="#eaf0f5" stroke="#526579" stroke-width="1.8"/>'
        + line("M12 16v5M8.8 8h6.4m-6.4 3h4.3", "#526579", 1.7)
    ),
}


def sign_svg(subtype, value):
    if subtype == "limite_velocidade":
        speed = str(value or "")
        number = speed if re.fullmatch(r"[0-9]{1,3}", speed) else "–"
        drawing = circle(WHITE, RED) + text(number, INK, 7.5 if len(number) == 3 else 9)
        key = f"{subtype}-{number}"
    else:
        key = subtype if subtype in SIGN_DRAWINGS else "outro"
        drawing = SIGN_DRAWINGS[key]
    return f"sign-{key}", (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" aria-hidden="true" focusable="false">'
        f"{drawing}</svg>"
    )
