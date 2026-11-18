"""
Lista TODOS os tipos de "points" (map features) que a Mapillary tem num município, com
quantos há de cada, e marca os que interessam ao desafio (sinais, iluminação, resíduos).

Os "points" são objetos que a Mapillary posicionou no mapa juntando deteções de várias
fotos (ver "Show points" no visualizador). Servem só para MEDIR (nunca entram no catálogo).

Saída: data/relatorios/pontos_mapillary.md  (+ ecrã)

Uso:
    python scripts/lista_pontos_mapillary.py
    python scripts/lista_pontos_mapillary.py --municipio Ílhavo
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter

import mercantile
import requests
from shapely.geometry import Point
from shapely.prepared import prep

sys.path.insert(0, os.path.dirname(__file__))
from estimate_mapillary import DATA_DIR, UA, get_boundary, tile_points  # noqa: E402
from fetch_ground_truth import MLY_TILES, MLY_URL  # noqa: E402

# tipos Mapillary -> classe do desafio (ou None = não interessa)
INTERESSA = {
    "object--street-light": "Iluminação (luminária: poste de luz ou candeeiro)",
    "object--support--utility-pole": "Iluminação? (poste: pode ter luminária ou ser elétrico)",
    "object--support--pole": "Suporte (poste de sinal/iluminação, sem tipo)",
    "object--trash-can": "Resíduos (caixote: contentor, ecoponto ou papeleira, sem tipo)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--municipio", default="Aveiro")
    args = ap.parse_args()
    token = os.getenv("MAPILLARY_TOKEN", "").strip()
    if not token or "COLA_AQUI" in token:
        sys.exit("Falta o MAPILLARY_TOKEN no ficheiro .env")
    boundary, label = get_boundary(args.municipio)
    poly = prep(boundary)
    tiles = list(mercantile.tiles(*boundary.bounds, 14))
    s = requests.Session()
    s.headers.update(UA)
    counts = {tileset: Counter() for tileset in MLY_TILES}
    seen = set()
    for tileset, layer in MLY_TILES.items():
        print(f"{tileset}: {len(tiles)} mosaicos")
        for t in tiles:
            r = None
            for attempt in range(3):
                r = s.get(MLY_URL.format(tileset=tileset, z=t.z, x=t.x, y=t.y, token=token), timeout=30)
                if r.status_code == 200:
                    break
                time.sleep(2 * (attempt + 1))
            if r is None or r.status_code != 200 or not r.content:
                continue
            for lon, lat, p in tile_points(t, r.content, layer=layer):
                if p.get("id") in seen or not poly.contains(Point(lon, lat)):
                    continue
                seen.add(p.get("id"))
                counts[tileset][p.get("value", "?")] += 1

    lines = [f"# Tipos de points da Mapillary em {label}", "",
             "Points = objetos posicionados pela Mapillary (só para medir; nunca entram no catálogo).", ""]
    for tileset, c in counts.items():
        signs = sum(v for k, v in c.items() if k.startswith(("regulatory--", "warning--", "information--",
                                                              "complementary--")))
        others = [(k, v) for k, v in c.most_common() if not k.startswith(("regulatory--", "warning--",
                                                                            "information--", "complementary--"))]
        print(f"\n=== {tileset}: {sum(c.values())} points, {len(c)} tipos (sinais: {signs})")
        lines += [f"## {tileset} — {sum(c.values())} points ({len(c)} tipos)", "",
                  f"Sinais de trânsito (regulatory/warning/information/complementary): **{signs}**", "",
                  "| Tipo Mapillary | Quantos | Interessa? |", "|---|---|---|"]
        for k, v in others:
            tag = INTERESSA.get(k, "")
            print(f"  {v:>6}  {k}  {('<- ' + tag) if tag else ''}")
            lines.append(f"| {k} | {v} | {tag} |")
        lines.append("")
    out = DATA_DIR / "relatorios"
    out.mkdir(parents=True, exist_ok=True)
    (out / "pontos_mapillary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n-> {out / 'pontos_mapillary.md'}")


if __name__ == "__main__":
    main()
