"""
Guarda como FOTOS DE REFERÊNCIA os exemplos escolhidos (a olho) numa folha do catálogo
(folha_catalogo.py). Cada referência é o recorte do objeto com uma pequena margem.

  data/referencias/aveiro/<categoria>/<classe>/<id>.jpg      classes do catálogo
  data/referencias/aveiro/nao/<tipo>/<id>.jpg                "parece, mas não é"
                                                             (papeleira, poste elétrico, ...)

Uso:
    python scripts/guarda_referencias.py --folha candeeiro --numeros 3,4,10 --classe iluminacao/candeeiro
    python scripts/guarda_referencias.py --folha indiferenciado --numeros 7,13 --classe nao/papeleira
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from estimate_mapillary import DATA_DIR  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folha", required=True, help="nome da folha (subtipo) em data/relatorios/folhas")
    ap.add_argument("--numeros", required=True, help="números na folha, separados por vírgula")
    ap.add_argument("--classe", required=True, help="<categoria>/<classe> ou nao/<tipo>")
    args = ap.parse_args()

    rows = {int(r["n"]): r for r in csv.DictReader(
        open(DATA_DIR / "relatorios" / "folhas" / f"{args.folha}.csv", encoding="utf-8-sig"))}
    with open(DATA_DIR / "catalogo" / "catalogo.geojson", encoding="utf-8") as f:
        cat = {ft["properties"]["id"]: ft["properties"] for ft in json.load(f)["features"]}
    files = {r["id"]: r["file"] for r in csv.DictReader(open(DATA_DIR / "meta" / "sample_images.csv", encoding="utf-8"))}
    out = DATA_DIR / "referencias" / "aveiro" / args.classe
    out.mkdir(parents=True, exist_ok=True)
    log = DATA_DIR / "referencias" / "aveiro" / "fontes.csv"
    new = not log.exists()
    with open(log, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ficheiro", "classe", "id_catalogo", "imagem_id", "fonte", "licenca"])
        for n in (int(x) for x in args.numeros.split(",")):
            e = cat[rows[n]["id"]]
            im = Image.open(DATA_DIR / "images" / files[e["imagem_id"]]).convert("RGB")
            x0, y0, x1, y1 = e["imagem_bbox"]
            m = 0.15 * max(x1 - x0, y1 - y0)
            p = out / f"{e['id']}.jpg"
            im.crop((int(x0 - m), int(y0 - m), int(x1 + m), int(y1 + m))).save(p, quality=92)
            w.writerow([p.relative_to(DATA_DIR / "referencias" / "aveiro").as_posix(), args.classe, e["id"],
                        e["imagem_id"], e.get("imagem_url"), "CC BY-SA 4.0 (Mapillary)"])
    print(f"{args.classe}: +{len(args.numeros.split(','))} -> {out}")


if __name__ == "__main__":
    main()
