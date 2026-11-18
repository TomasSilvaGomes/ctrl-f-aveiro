"""
Revisão a olho do catálogo, em 2 passos:

  folha     folha de imagens de um tipo: exemplos numerados, cada um com a caixa a vermelho
            e a foto à volta -> data/relatorios/folhas/<subtipo>.jpg (+ .csv: número -> id)
  guardar   guarda os números escolhidos numa folha como FOTOS DE REFERÊNCIA (recorte com
            margem) -> data/referencias/aveiro/<categoria>/<classe>/<id>.jpg
            ou, para "parece mas não é", data/referencias/aveiro/nao/<tipo>/<id>.jpg
            (créditos em data/referencias/aveiro/fontes.csv)

Uso:
    python scripts/folha_catalogo.py folha --subtipo candeeiro subterraneo --n 24
    python scripts/folha_catalogo.py guardar --folha candeeiro --numeros 3,4,10 --classe iluminacao/candeeiro
    python scripts/folha_catalogo.py guardar --folha indiferenciado --numeros 7,13 --classe nao/papeleira
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
from comum import DATA_DIR, font  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
FOLHAS = DATA_DIR / "relatorios" / "folhas"
REFS = DATA_DIR / "referencias" / "aveiro"


def catalog():
    with open(DATA_DIR / "catalogo" / "catalogo.geojson", encoding="utf-8") as f:
        return [ft["properties"] for ft in json.load(f)["features"]]


def files():
    with open(DATA_DIR / "meta" / "sample_images.csv", encoding="utf-8") as f:
        return {r["id"]: r["file"] for r in csv.DictReader(f)}


def cmd_folha(args):
    feats, fs = catalog(), files()
    FOLHAS.mkdir(parents=True, exist_ok=True)
    f15 = font(15)
    for sub in args.subtipo:
        es = [e for e in feats if e["subtipo"] == sub]
        if not es:
            print(f"{sub}: nenhum elemento no catálogo")
            continue
        sample = random.Random(args.seed).sample(es, min(args.n, len(es)))
        cols, size = 8, 200
        sheet = Image.new("RGB", (cols * size, ((len(sample) + cols - 1) // cols) * size), "white")
        dr = ImageDraw.Draw(sheet)
        rows = []
        for i, e in enumerate(sample):
            im = Image.open(DATA_DIR / "images" / fs[e["imagem_id"]]).convert("RGB")
            x0, y0, x1, y1 = e["imagem_bbox"]
            cx, cy, s = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * 0.8 + 20
            ox, oy = int(cx - s), int(cy - s)
            v = im.crop((ox, oy, int(cx + s), int(cy + s)))
            ImageDraw.Draw(v).rectangle([x0 - ox, y0 - oy, x1 - ox, y1 - oy], outline=(255, 0, 0),
                                        width=max(2, int(s / 50)))
            x, y = (i % cols) * size, (i // cols) * size
            sheet.paste(v.resize((size - 4, size - 4)), (x + 2, y + 2))
            dr.rectangle([x + 2, y + 2, x + 34, y + 21], fill=(0, 0, 0))
            dr.text((x + 5, y + 3), str(i + 1), fill=(255, 255, 0), font=f15)
            rows.append({"n": i + 1, "id": e["id"], "label": e["label"], "confianca": e["confianca"],
                         "n_vistas": e["n_vistas"], "imagem_id": e["imagem_id"]})
        sheet.save(FOLHAS / f"{sub}.jpg", quality=88)
        with open(FOLHAS / f"{sub}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"{sub}: {len(sample)} de {len(es)} -> {FOLHAS / f'{sub}.jpg'}")


def cmd_guardar(args):
    with open(FOLHAS / f"{args.folha}.csv", encoding="utf-8-sig") as f:
        rows = {int(r["n"]): r for r in csv.DictReader(f)}
    cat, fs = {e["id"]: e for e in catalog()}, files()
    out = REFS / args.classe
    out.mkdir(parents=True, exist_ok=True)
    log = REFS / "fontes.csv"
    new = not log.exists()
    nums = [int(x) for x in args.numeros.split(",")]
    with open(log, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ficheiro", "classe", "id_catalogo", "imagem_id", "fonte", "licenca"])
        for n in nums:
            e = cat[rows[n]["id"]]
            im = Image.open(DATA_DIR / "images" / fs[e["imagem_id"]]).convert("RGB")
            x0, y0, x1, y1 = e["imagem_bbox"]
            m = 0.15 * max(x1 - x0, y1 - y0)
            p = out / f"{e['id']}.jpg"
            im.crop((int(x0 - m), int(y0 - m), int(x1 + m), int(y1 + m))).save(p, quality=92)
            w.writerow([p.relative_to(REFS).as_posix(), args.classe, e["id"], e["imagem_id"],
                        e.get("imagem_url"), "CC BY-SA 4.0 (Mapillary)"])
    print(f"{args.classe}: +{len(nums)} -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("folha", help="folha de imagens numeradas de um ou mais tipos")
    p.add_argument("--subtipo", nargs="+", required=True)
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("guardar", help="guarda números de uma folha como fotos de referência")
    p.add_argument("--folha", required=True, help="nome da folha (subtipo) em data/relatorios/folhas")
    p.add_argument("--numeros", required=True, help="números na folha, separados por vírgula")
    p.add_argument("--classe", required=True, help="<categoria>/<classe> ou nao/<tipo>")
    args = ap.parse_args()
    {"folha": cmd_folha, "guardar": cmd_guardar}[args.cmd](args)


if __name__ == "__main__":
    main()
