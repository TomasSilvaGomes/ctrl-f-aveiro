"""
Extrai os desenhos dos sinais do Guia de Sinalização Rodoviária (DGV, 2003), com o
código e o nome lidos da legenda por baixo de cada desenho (ex.: "A1a – Curva à direita").

Servem de REFERÊNCIA para o modelo de semelhança (DINOv2): cada recorte da rua é
comparado com estes desenhos. Uso interno para teste — confirmar com os mentores se
os desenhos podem ser usados/distribuídos (alternativa: SVG da Wikimedia Commons).

Entrada:  pdfs/SinalizacaoRodoviaria.pdf
Saída:    data/referencias/sinais/<código>.png
          data/referencias/sinais.csv   (codigo, nome, familia, pagina, ficheiro)

Uso:
    python scripts/extrair_referencias.py
"""

from __future__ import annotations

import csv
import re
import sys
import warnings
from pathlib import Path

import pypdfium2 as pdfium

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "pdfs" / "SinalizacaoRodoviaria.pdf"
OUT = ROOT / "data" / "referencias"
SCALE = 4  # render a 4x (288 dpi)
# código RST no início da legenda: A1a, B2, C13, D4, H7, C11a, E1, ...
CODE = re.compile(r"^\s*([A-Z]{1,2}\d{1,2}[a-z]?)\s*[–—-]\s*(.+)$")
FAMILIAS = {"A": "perigo", "B": "cedencia", "C": "proibicao", "D": "obrigacao", "E": "prescricao",
            "F": "prescricao", "G": "zona", "H": "informacao", "I": "pre_sinalizacao",
            "J": "direcao", "L": "confirmacao", "N": "localidade", "O": "outros", "M": "adicional"}


def caption(tp, box, rects):
    """Texto das linhas por baixo do desenho, na mesma coluna."""
    l, b, r, t = box
    cx = (l + r) / 2
    sel = [q for q in rects if b - 60 <= q[3] <= b + 2 and q[0] - 15 <= cx <= q[2] + 15]
    if not sel:
        return ""
    # só as linhas seguidas (sem salto grande) por baixo do desenho
    sel.sort(key=lambda q: -q[3])
    keep = [sel[0]]
    for q in sel[1:]:
        if keep[-1][1] - q[3] > 8:
            break
        keep.append(q)
    # texto do bloco inteiro de uma vez (evita palavras partidas entre retângulos)
    box = (min(q[0] for q in keep) - 1, min(q[1] for q in keep) - 1,
           max(q[2] for q in keep) + 1, max(q[3] for q in keep) + 1)
    return re.sub(r"\s+", " ", tp.get_text_bounded(*box)).strip()


def main():
    if not PDF.exists():
        sys.exit(f"Falta {PDF}")
    (OUT / "sinais").mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(str(PDF))
    rows, seen = [], set()
    for pi in range(len(pdf)):
        page = pdf[pi]
        tp = page.get_textpage()
        rects = [tp.get_rect(k) for k in range(tp.count_rects())]
        H = page.get_size()[1]
        bitmap = None
        for o in page.get_objects():
            if o.type != 3:  # 3 = imagem
                continue
            box = o.get_pos()
            m = CODE.match(caption(tp, box, rects))
            if not m:
                continue
            code, name = m.group(1), re.sub(r"\s+", " ", m.group(2)).strip()
            if code in seen:
                continue
            seen.add(code)
            if bitmap is None:
                bitmap = page.render(scale=SCALE).to_pil().convert("RGB")
            l, b, r, t = box
            crop = bitmap.crop((int(l * SCALE), int((H - t) * SCALE), int(r * SCALE), int((H - b) * SCALE)))
            fname = f"{code}.png"
            crop.save(OUT / "sinais" / fname)
            rows.append({"codigo": code, "nome": name, "familia": FAMILIAS.get(code[0], "outros"),
                         "pagina": pi + 1, "ficheiro": f"sinais/{fname}"})
    with open(OUT / "sinais.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print(f"{len(rows)} sinais extraídos -> {OUT / 'sinais.csv'}")
    print("Por família: " + ", ".join(f"{k}: {v}" for k, v in Counter(r["familia"] for r in rows).items()))


if __name__ == "__main__":
    main()
