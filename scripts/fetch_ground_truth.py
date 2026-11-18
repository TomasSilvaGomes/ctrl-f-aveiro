"""
Pontos de referência do concelho, SÓ para medir (nunca entram no catálogo; ODbL/Mapillary):
  1. OpenStreetMap (Overpass): candeeiros, ecopontos, contentores, sinais.
  2. Mapillary "points" (map features): objetos já posicionados pela Mapillary
     (candeeiros, contentores, postes, sinais). Poste com luminária a <= 4 m = poste de luz.

  --listar-tipos   em vez de descarregar, lista TODOS os tipos de points que a Mapillary tem
                   no município, com quantos há de cada (-> data/relatorios/pontos_mapillary.md)

Uso:
    python scripts/fetch_ground_truth.py
    python scripts/fetch_ground_truth.py --listar-tipos --municipio Ílhavo
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter

import mercantile
from shapely.geometry import Point
from shapely.prepared import prep

sys.path.insert(0, os.path.dirname(__file__))
from comum import DATA_DIR  # noqa: E402
from mapillary import MLY_TILES, MLY_URL, UA, categorize_mly, get_boundary, session, tile_points, token  # noqa: E402

OUT_DIR = DATA_DIR / "ground_truth"
OVERPASS = "https://overpass-api.de/api/interpreter"
# tipos Mapillary que interessam ao desafio (para a listagem)
INTERESSA = {
    "object--street-light": "Iluminação (luminária: poste de luz ou candeeiro)",
    "object--support--utility-pole": "Iluminação? (poste: pode ter luminária ou ser elétrico)",
    "object--support--pole": "Suporte (poste de sinal/iluminação, sem tipo)",
    "object--trash-can": "Resíduos (caixote: contentor, ecoponto ou papeleira, sem tipo)",
}
SIGN_PREFIXES = ("regulatory--", "warning--", "information--", "complementary--")


# ---------------------------------------------------------------- categorias
def categorize_osm(tags: dict):
    """Mapeia tags OSM para (categoria, subtipo) do desafio, ou None."""
    hw, am = tags.get("highway"), tags.get("amenity")
    if hw == "street_lamp":
        return "iluminacao", "candeeiro"
    if am == "recycling" and tags.get("recycling_type", "container") == "container":
        sub = "subterraneo" if tags.get("location") == "underground" else "ecoponto"
        return "residuos", sub
    if am == "waste_disposal":
        sub = "subterraneo" if tags.get("location") == "underground" else "indiferenciado"
        return "residuos", sub
    if hw == "stop":
        return "sinalizacao", "stop"
    if hw == "give_way":
        return "sinalizacao", "cedencia"
    if "traffic_sign" in tags:
        return "sinalizacao", tags["traffic_sign"]
    return None


# ---------------------------------------------------------------- fontes
def fetch_osm(poly):
    query = """
    [out:json][timeout:180];
    area["boundary"="administrative"]["admin_level"="7"]["name"="Aveiro"]->.a;
    (
      node["highway"="street_lamp"](area.a);
      nwr["amenity"="recycling"](area.a);
      nwr["amenity"="waste_disposal"](area.a);
      node["highway"="stop"](area.a);
      node["highway"="give_way"](area.a);
      node["traffic_sign"](area.a);
    );
    out center tags;
    """
    r = session().post(OVERPASS, data={"data": query}, headers=UA, timeout=240)
    r.raise_for_status()
    feats = []
    for el in r.json()["elements"]:
        lon = el.get("lon") or el.get("center", {}).get("lon")
        lat = el.get("lat") or el.get("center", {}).get("lat")
        if lon is None or not poly.contains(Point(lon, lat)):
            continue
        cat = categorize_osm(el.get("tags", {}))
        if not cat:
            continue
        feats.append(feature(lon, lat, {
            "source": "osm", "osm_id": f"{el['type']}/{el['id']}",
            "category": cat[0], "subtype": cat[1], "tags": el.get("tags", {}),
        }))
    return feats


def mly_points(boundary, poly, tok):
    """Todos os points da Mapillary dentro do município: (tileset, lon, lat, props), sem repetidos."""
    tiles = list(mercantile.tiles(*boundary.bounds, 14))
    seen, s = set(), session()
    for tileset, layer in MLY_TILES.items():
        print(f"  {tileset}: {len(tiles)} mosaicos")
        for t in tiles:
            url = MLY_URL.format(tileset=tileset, z=t.z, x=t.x, y=t.y, token=tok)
            for attempt in range(3):
                r = s.get(url, timeout=30)
                if r.status_code == 200:
                    break
                time.sleep(2 * (attempt + 1))
            else:
                continue
            if not r.content:
                continue
            for lon, lat, p in tile_points(t, r.content, layer=layer):
                fid = p.get("id")
                if fid in seen or not poly.contains(Point(lon, lat)):
                    continue
                seen.add(fid)
                yield tileset, lon, lat, p


def fetch_mapillary(boundary, poly, tok):
    feats = []
    for _, lon, lat, p in mly_points(boundary, poly, tok):
        cat = categorize_mly(p.get("value", ""))
        if cat:
            feats.append(feature(lon, lat, {
                "source": "mapillary", "mly_id": p.get("id"), "value": p.get("value"),
                "category": cat[0], "subtype": cat[1],
                "first_seen_at": p.get("first_seen_at"), "last_seen_at": p.get("last_seen_at"),
            }))
    return feats


def listar_tipos(boundary, poly, tok, label):
    counts = {ts: Counter() for ts in MLY_TILES}
    for ts, _, _, p in mly_points(boundary, poly, tok):
        counts[ts][p.get("value", "?")] += 1
    lines = [f"# Tipos de points da Mapillary em {label}", "",
             "Points = objetos posicionados pela Mapillary (só para medir; nunca entram no catálogo).", ""]
    for ts, c in counts.items():
        signs = sum(v for k, v in c.items() if k.startswith(SIGN_PREFIXES))
        others = [(k, v) for k, v in c.most_common() if not k.startswith(SIGN_PREFIXES)]
        print(f"\n=== {ts}: {sum(c.values())} points, {len(c)} tipos (sinais: {signs})")
        lines += [f"## {ts} — {sum(c.values())} points ({len(c)} tipos)", "",
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


def mark_light_poles(feats, radius=4.0):
    """Poste da Mapillary com uma luminária (street-light) a <= radius m -> "poste_luz"
    (a posição do poste é a BASE, como a que o sistema calcula); sem luminária perto ->
    "poste_eletrico" (não conta). A luminária fica marcada com o poste que a segura."""
    kx, ky = 111320 * math.cos(math.radians(40.64)), 110540
    lights = [f for f in feats if f["properties"]["value"] == "object--street-light"]
    grid = {}
    for i, f in enumerate(lights):
        x, y = f["geometry"]["coordinates"][0] * kx, f["geometry"]["coordinates"][1] * ky
        grid.setdefault((int(x // radius), int(y // radius)), []).append((x, y, i))
    has_pole = set()
    n = {"poste_luz": 0, "poste_eletrico": 0}
    for f in feats:
        if f["properties"]["subtype"] != "poste":
            continue
        x, y = f["geometry"]["coordinates"][0] * kx, f["geometry"]["coordinates"][1] * ky
        near = [i for a in (-1, 0, 1) for b in (-1, 0, 1)
                for lx, ly, i in grid.get((int(x // radius) + a, int(y // radius) + b), [])
                if math.hypot(lx - x, ly - y) <= radius]
        f["properties"]["subtype"] = "poste_luz" if near else "poste_eletrico"
        n[f["properties"]["subtype"]] += 1
        has_pole.update(near)
    for i, f in enumerate(lights):
        f["properties"]["em_poste"] = i in has_pole
    print(f"  postes: {n['poste_luz']} com luminária perto (poste de luz), "
          f"{n['poste_eletrico']} sem (elétrico, não contam); luminárias sem poste perto: "
          f"{len(lights) - len(has_pole)} (candeeiros de parede?)")
    return feats


# ---------------------------------------------------------------- util
def feature(lon, lat, props):
    return {"type": "Feature",
            "geometry": {"type": "Point", "coordinates": [round(lon, 7), round(lat, 7)]},
            "properties": props}


def save(feats, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    with path.open("w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f, ensure_ascii=False)
    return path


def summary(feats, label):
    c = Counter((f["properties"]["category"], f["properties"]["subtype"]) for f in feats)
    by_cat = Counter(f["properties"]["category"] for f in feats)
    print(f"\n=== {label}: {len(feats):,} pontos ===")
    for cat, n in by_cat.most_common():
        subs = [(s, k) for (c2, s), k in c.items() if c2 == cat]
        subs.sort(key=lambda x: -x[1])
        top = ", ".join(f"{s}: {k}" for s, k in subs[:8])
        print(f"  {cat:<12} {n:>6,}   ({top}{', ...' if len(subs) > 8 else ''})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listar-tipos", action="store_true", help="só lista os tipos de points da Mapillary")
    ap.add_argument("--municipio", default="Aveiro", help="(com --listar-tipos) outro município")
    args = ap.parse_args()
    tok = token()
    boundary, label = get_boundary(args.municipio if args.listar_tipos else "Aveiro")
    poly = prep(boundary)
    print(f"Limite: {label}")
    if args.listar_tipos:
        return listar_tipos(boundary, poly, tok, label)

    print("OSM (Overpass)...")
    osm = fetch_osm(poly)
    print(f"  guardado em {save(osm, 'osm_aveiro.geojson')}")
    print("Mapillary map features...")
    mly = mark_light_poles(fetch_mapillary(boundary, poly, tok))
    print(f"  guardado em {save(mly, 'mapillary_features_aveiro.geojson')}")
    summary(osm, "OpenStreetMap")
    summary(mly, "Mapillary map features")


if __name__ == "__main__":
    main()
