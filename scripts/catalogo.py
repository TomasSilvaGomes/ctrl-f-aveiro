"""
4.ª fase: CATÁLOGO — junta as várias vistas do mesmo objeto (elimina duplicados),
forma as "Ilhas de reciclagem" e exporta GeoJSON + CSV.

  1. Agrupar: deteções da mesma chave a <= raio metros juntam-se num elemento
       sinais/contentores: chave = (categoria, subtipo)   (sinais diferentes no mesmo poste
                                                           e iglu ao lado de contentor preto
                                                           ficam separados)
       iluminação:        chave = categoria (subtipo por voto)
     posição = mediana; subtipo/valor = voto pesado pela confiança;
     imagem mostrada = vista com maior confiança.
  2. Ilhas: contentores a <= raio_m entre si (config/taxonomia.yaml, agrupamentos) -> 1 elemento
     "ilha" com o n.º de contentores (membros), a composição e cada contentor em "itens" (a app
     abre-os ao carregar na ilha). Nome: só subterrâneos -> ilha de subterrâneos; só
     indiferenciados -> ilha de indiferenciados; tipos diferentes juntos -> ilha de reciclagem.
  3. Exportar data/catalogo/catalogo.geojson e .csv, com atribuição por imagem
     (CC BY-SA 4.0: autor + link).
  4. Validar: duplicados por objeto real e cobertura vs pontos Mapillary/OSM.

Uso:
    python scripts/catalogo.py                     # limiares de config/taxonomia.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(__file__))
from comum import DATA_DIR, limiares  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LAT0 = 40.64
KX, KY = 111320 * math.cos(math.radians(LAT0)), 110540
RADIUS = {"sinalizacao": 4.0, "iluminacao": 5.0, "residuos": 3.0}
KEY_BY_SUBTYPE = {"sinalizacao", "residuos"}


def xy(lon, lat):
    return lon * KX, lat * KY


def dist_m(a, b):
    return math.hypot((a[0] - b[0]) * KX, (a[1] - b[1]) * KY)


def cluster(points, radius):
    """Agrupamento guloso por grelha: pontos ordenados por confiança; cada ponto junta-se
    ao grupo cujo centro (média) esteja a <= radius, senão abre um grupo novo."""
    grid = defaultdict(list)  # célula -> índices de grupos
    groups = []               # [soma_x, soma_y, n, [itens]]
    for p in sorted(points, key=lambda d: -d["confidence"]):
        x, y = xy(p["lon"], p["lat"])
        cx, cy = int(x // radius), int(y // radius)
        best, bd = None, radius
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                for g in grid[(cx + i, cy + j)]:
                    gx, gy = groups[g][0] / groups[g][2], groups[g][1] / groups[g][2]
                    d = math.hypot(gx - x, gy - y)
                    if d <= bd:
                        best, bd = g, d
        if best is None:
            groups.append([x, y, 1, [p]])
            grid[(cx, cy)].append(len(groups) - 1)
        else:
            g = groups[best]
            g[0] += x; g[1] += y; g[2] += 1; g[3].append(p)
    return [g[3] for g in groups]


def element(items, tax, imgs):
    cat = items[0]["category"]
    votes = Counter()
    for d in items:
        votes[d["subtype"]] += d["confidence"]
    sub = votes.most_common(1)[0][0]
    best = max((d for d in items if d["subtype"] == sub), key=lambda d: d["confidence"])
    info = next(s for s in tax["categorias"][cat]["subtipos"] if s["id"] == sub)
    label, valor, codigo = info["label"], None, info.get("codigo")
    rst = Counter()   # código oficial dado por sinais.py codigo (voto entre vistas, pela confiança)
    for d in items:
        if d["subtype"] == sub and d.get("codigo_rst"):
            rst[(d["codigo_rst"], d["nome_rst"])] += d["confidence"]
    if rst:
        (c, nome), _ = rst.most_common(1)[0]
        label, codigo = f"Sinal {nome}", f"PT:{c}"
    if sub == "limite_velocidade":
        vals = Counter()  # voto entre vistas, pesado pela confiança da leitura
        for d in items:
            if d.get("valor"):
                vals[d["valor"]] += d.get("valor_conf", 1.0)
        valor = vals.most_common(1)[0][0] if vals else None
        label = label.replace("{n}", str(valor)) if valor else label.replace(" {n} km/h", "")
    meta = imgs.get(best["image_id"], {})
    image_ids = sorted({d["image_id"] for d in items})
    return {
        "categoria": cat, "categoria_label": tax["categorias"][cat]["label"],
        "subtipo": sub, "label": label, "codigo": codigo, "valor": valor,
        "lon": round(statistics.median(d["lon"] for d in items), 7),
        "lat": round(statistics.median(d["lat"] for d in items), 7),
        "confianca": round(best["confidence"], 3),
        "concordancia": round(votes[sub] / sum(votes.values()), 2),  # fração do voto no subtipo
        "n_vistas": len(items), "n_imagens": len(image_ids),
        "imagem_id": best["image_id"], "imagem_bbox": [round(v) for v in best["bbox"]],
        "imagem_url": meta.get("source_url"), "autor": meta.get("creator_username"),
        "licenca": meta.get("license") or "CC BY-SA 4.0", "fonte": "Mapillary",
        "imagens": image_ids, "estado": "automatico", "membros": None, "composicao": None,
    }


def island_kind(es, rule):
    """Tipo de ilha pela composição: só subterrâneos; só indiferenciados; tipos diferentes juntos
    (ou ecopontos) = ilha de reciclagem. None se não chega ao mínimo de contentores desse tipo."""
    subs = {e["subtipo"] for e in es}
    if subs == {"subterraneo"}:
        kind = "subterraneos"
    elif subs == {"indiferenciado"}:
        kind = "indiferenciados"
    else:   # tipos diferentes juntos (ou só ecopontos) = ilha de reciclagem
        kind = "reciclagem"
    t = rule["tipos"][kind]
    return (kind, t) if len(es) >= t.get("min_membros", 2) else (None, None)


def make_islands(elems, rule):
    """Contentores próximos (ligação simples a <= raio_m) formam UMA ilha: um só elemento com o
    n.º de contentores (membros) e a composição. O nome vem do que a ilha tem (island_kind)."""
    eco = [i for i, e in enumerate(elems) if e["categoria"] == rule["categoria"]]
    parent = {i: i for i in eco}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a in range(len(eco)):
        for b in range(a + 1, len(eco)):
            ea, eb = elems[eco[a]], elems[eco[b]]
            if dist_m((ea["lon"], ea["lat"]), (eb["lon"], eb["lat"])) <= rule["raio_m"]:
                parent[find(eco[a])] = find(eco[b])
    groups = defaultdict(list)
    for i in eco:
        groups[find(i)].append(i)
    islands = []
    for members in groups.values():
        es = [elems[i] for i in members]
        kind, t = island_kind(es, rule)
        if kind is None:
            continue
        best = max(es, key=lambda e: e["confianca"])
        isl = dict(best)
        comp = Counter(e["subtipo"] for e in es)
        isl.update({
            "subtipo": t["subtipo"], "label": t["label"], "codigo": t["codigo"],
            "composicao": " + ".join(f"{n} {s}" for s, n in comp.most_common()),
            # cada contentor da ilha, para a app abrir ao carregar na ilha
            "itens": [{k: e[k] for k in ("subtipo", "label", "lon", "lat", "confianca", "imagem_id",
                                          "imagem_bbox", "imagem_url", "autor")} for e in es],
            "lon": round(statistics.mean(e["lon"] for e in es), 7),
            "lat": round(statistics.mean(e["lat"] for e in es), 7),
            "n_vistas": sum(e["n_vistas"] for e in es),
            "imagens": sorted({i for e in es for i in e["imagens"]}),
            "n_imagens": len({i for e in es for i in e["imagens"]}),
            "membros": len(es),
        })
        islands.append((isl, members))
    return islands


def load_refs():
    """Pontos de referência (Mapillary map features + OSM, só para medir) na zona coberta:
    a <= 25 m de alguma câmara da amostra."""
    with open(DATA_DIR / "meta" / "sample_images.csv", encoding="utf-8") as f:
        cams = [(float(r["lon"]), float(r["lat"])) for r in csv.DictReader(f) if r["lon"]]
    cam_cells = {(int(xy(*c)[0] // 25), int(xy(*c)[1] // 25)) for c in cams}

    def covered(p):
        x, y = xy(*p)
        return any((int(x // 25) + i, int(y // 25) + j) in cam_cells for i in (-1, 0, 1) for j in (-1, 0, 1))

    refs = defaultdict(list)
    for name in ("mapillary_features_aveiro.geojson", "osm_aveiro.geojson"):
        with open(DATA_DIR / "ground_truth" / name, encoding="utf-8") as f:
            for ft in json.load(f)["features"]:
                p = ft["properties"]
                if p["subtype"] in ("poste", "poste_eletrico"):
                    continue
                c = tuple(ft["geometry"]["coordinates"])
                if covered(c):
                    refs[p["category"]].append(c)
    return refs


def points(es):
    """Pontos a comparar com as referências: nas ilhas, cada contentor (itens) conta por si."""
    return [(i["lon"], i["lat"]) for e in es for i in (e.get("itens") or [e])]


def validate(catalog, min_views, quiet=False, refs=None):
    """Duplicados e cobertura vs pontos de referência. Devolve {categoria: números}."""
    refs = load_refs() if refs is None else refs
    out = {}
    if not quiet:
        print(f"\nValidação do catálogo (elementos com >= {min_views} vistas; referência a <= 8 m)")
        print(f"{'categoria':<13}{'elementos':>10}{'refs zona':>10}{'elem. c/ ref':>13}{'refs encontradas':>18}"
              f"{'elem./ref':>11}")
    for cat in ("sinalizacao", "iluminacao", "residuos"):
        es = [e for e in catalog if e["categoria"] == cat and e["n_vistas"] >= min_views]
        rs = refs.get(cat, [])
        if not es or not rs:
            continue
        grid = defaultdict(list)
        for i, r in enumerate(rs):
            x, y = xy(*r)
            grid[(int(x // 8), int(y // 8))].append(i)
        hit_ref = Counter()
        n_match = 0
        for p in points(es):
            x, y = xy(*p)
            cand = [i for a in (-1, 0, 1) for b in (-1, 0, 1) for i in grid[(int(x // 8) + a, int(y // 8) + b)]]
            near = [i for i in cand if dist_m(p, rs[i]) <= 8]
            if near:
                n_match += 1
                hit_ref[min(near, key=lambda i: dist_m(p, rs[i]))] += 1
        dup = (sum(hit_ref.values()) / len(hit_ref)) if hit_ref else 0
        es = points(es)
        out[cat] = {"elementos": len(es), "refs": len(rs), "precisao": n_match / len(es),
                    "encontra": len(hit_ref) / len(rs), "por_ref": dup}
        if not quiet:
            print(f"{cat:<13}{len(es):>10,}{len(rs):>10,}{n_match / len(es):>13.0%}"
                  f"{len(hit_ref) / len(rs):>18.0%}{dup:>11.2f}")
    if not quiet:
        print("elem./ref = elementos do catálogo por objeto de referência encontrado (1.00 = sem duplicados)")
    return out


def montar(dets, tax, imgs, min_views=1):
    """Deteções com posição -> elementos do catálogo (repetidos juntos, ilhas de reciclagem)."""
    by_key = defaultdict(list)
    for d in dets:
        key = (d["category"], d["subtype"]) if d["category"] in KEY_BY_SUBTYPE else (d["category"],)
        by_key[key].append(d)
    elems = []
    for key, pts in by_key.items():
        for items in cluster(pts, RADIUS[key[0]]):
            elems.append(element(items, tax, imgs))
    elems = [e for e in elems if e["n_vistas"] >= min_views]

    rule = tax["agrupamentos"]["ilha"]
    islands = make_islands(elems, rule)
    in_island = {i for _, m in islands for i in m}
    catalog = [e for i, e in enumerate(elems) if i not in in_island] + [isl for isl, _ in islands]
    catalog.sort(key=lambda e: (e["categoria"], e["subtipo"], -e["confianca"]))
    for n, e in enumerate(catalog, 1):
        e["id"] = f"AVR-{n:05d}"
    return catalog, islands, in_island


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="owlv2")
    ap.add_argument("--min-conf", type=float, default=None,
                    help="igual para todas as categorias (omissão: confianca_minima da taxonomia)")
    ap.add_argument("--limiares", default="", help="por categoria, ex.: iluminacao=0.7,residuos=0.5")
    ap.add_argument("--min-views", type=int, default=1, help="só exporta elementos vistos em >= N deteções")
    args = ap.parse_args()
    thr = limiares(args.limiares, args.min_conf)

    with open(ROOT / "config" / "taxonomia.yaml", encoding="utf-8") as f:
        tax = yaml.safe_load(f)
    with open(DATA_DIR / "meta" / "sample_images.csv", encoding="utf-8") as f:
        imgs = {r["id"]: r for r in csv.DictReader(f)}
    dets = []
    with open(DATA_DIR / "preds" / f"{args.model}_geo.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["confidence"] >= thr[d["category"]]:
                dets.append(d)

    catalog, islands, in_island = montar(dets, tax, imgs, args.min_views)

    out = DATA_DIR / "catalogo"
    out.mkdir(parents=True, exist_ok=True)
    feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [e["lon"], e["lat"]]},
              "properties": {k: v for k, v in e.items() if k not in ("lon", "lat")}} for e in catalog]
    with open(out / "catalogo.geojson", "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f, ensure_ascii=False)
    cols = ["id", "categoria", "categoria_label", "subtipo", "label", "codigo", "valor",
            "lon", "lat", "confianca", "concordancia", "n_vistas", "n_imagens", "membros", "composicao",
            "imagem_id", "imagem_url", "autor", "licenca", "fonte", "estado"]
    with open(out / "catalogo.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for e in catalog:
            w.writerow(e)

    print(f"{len(dets):,} deteções (confiança >= " + ", ".join(f"{k} {v}" for k, v in thr.items()) + f") -> {len(catalog):,} elementos no catálogo")
    print(f"-> {out / 'catalogo.geojson'}\n-> {out / 'catalogo.csv'}\n")
    cnt = Counter((e["categoria"], e["label"].split(" {")[0]) for e in catalog)
    for cat in ("sinalizacao", "iluminacao", "residuos"):
        items = sorted(((n, l) for (c, l), n in cnt.items() if c == cat), reverse=True)
        print(f"  {cat}: " + ", ".join(f"{l} {n}" for n, l in items[:12]))
    kinds = Counter(isl["label"] for isl, _ in islands)
    print(f"  Ilhas: {len(islands)} (juntam {len(in_island)} contentores próximos): "
          + ", ".join(f"{k} {n}" for k, n in kinds.most_common()))
    validate(catalog, 1)
    validate(catalog, 2)


if __name__ == "__main__":
    main()
