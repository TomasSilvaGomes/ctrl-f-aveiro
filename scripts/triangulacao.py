"""
Triangulação multivista: posição de cada objeto a partir de VÁRIAS fotos seguidas.

Numa só foto sabemos bem a DIREÇÃO do objeto (erro ~1°), mas a DISTÂNCIA é um
palpite (altura da câmara ou tamanho do sinal). Com várias fotos tiradas de sítios
diferentes, as linhas de direção cruzam-se no objeto: a distância passa a ser medida
pela geometria, sem palpite.

  1. Juntar as deteções do mesmo objeto (mesma passagem, mesmo tipo, posições de
     uma só foto a <= --raio m). Fica 1 linha por foto (a de maior confiança).
  2. Ponto que fica mais perto de todas as linhas (mínimos quadrados, no chão, 2D).
     Linhas que passam a mais de --residuo m do ponto saem e formam outro objeto.
  3. Só conta como "triangulado" se as fotos estiverem afastadas (>= --base m) e as
     linhas se cruzarem com ângulo suficiente (>= --angulo graus). Senão fica a
     posição de uma só foto (mediana).
  4. Medições (sem referência externa):
       - erro da distância de uma só foto = distância (1 foto) - distância triangulada;
       - repetibilidade: o mesmo objeto triangulado em passagens de dias diferentes.
     E, como antes, distância ao ponto Mapillary/OSM mais próximo (só para comparar).

A posição de cada câmara é a "computed_geometry" da Mapillary (GPS corrigido pela
reconstrução 3D da própria Mapillary); a escala em metros vem do GPS.

Entrada:  <DATA_DIR>/preds/<modelo>_geo.jsonl, <DATA_DIR>/meta/sample_images.csv
Saída:    <DATA_DIR>/preds/<modelo>_tri.jsonl, <DATA_DIR>/relatorios/triangulacao.md

Uso (PowerShell, no piloto da rua):
    $env:DATA_DIR = "data\\piloto_rua"
    python scripts/triangulacao.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict

import numpy as np
from shapely.geometry import Point

sys.path.insert(0, os.path.dirname(__file__))
from build_catalog import cluster  # noqa: E402
from comum import DATA_DIR  # noqa: E402
from geolocate import reference_index, xy  # noqa: E402

NOMES = {"sinalizacao": "Sinais", "iluminacao": "Iluminação", "residuos": "Contentores"}
KX, KY = xy(1, 0)[0], xy(0, 1)[1]


def to_lonlat(x, y):
    return x / KX, y / KY


def solve(rays):
    """Ponto mais perto de todas as linhas. rays = [(cx, cy, ux, uy, peso)].
    Devolve (x, y, menor valor próprio) — valor baixo = linhas quase paralelas."""
    A, b = np.zeros((2, 2)), np.zeros(2)
    for cx, cy, ux, uy, w in rays:
        n = np.array([-uy, ux])            # normal à linha
        M = w * np.outer(n, n)
        A += M
        b += M @ np.array([cx, cy])
    ev = np.linalg.eigvalsh(A)
    if ev[0] < 1e-9:
        return None
    p = np.linalg.solve(A, b)
    return float(p[0]), float(p[1]), float(ev[0] / A.trace())


def residual(p, ray):
    cx, cy, ux, uy, _ = ray
    dx, dy = p[0] - cx, p[1] - cy
    return abs(-uy * dx + ux * dy), ux * dx + uy * dy   # (afastamento da linha, distância ao longo)


def spread_deg(rays):
    """Maior ângulo entre duas linhas do grupo."""
    angs = [math.atan2(r[2], r[3]) for r in rays]
    return max(abs((a - b + math.pi) % (2 * math.pi) - math.pi) for a in angs for b in angs) * 180 / math.pi


def baseline(rays):
    return max(math.hypot(a[0] - b[0], a[1] - b[1]) for a in rays for b in rays)


def triangulate(dets, args):
    """Triangula um grupo de deteções; devolve lista de objetos (pode partir o grupo)."""
    # 1 linha por foto (a de maior confiança)
    best = {}
    for d in dets:
        if d["image_id"] not in best or d["confidence"] > best[d["image_id"]]["confidence"]:
            best[d["image_id"]] = d
    pool = list(best.values())
    objs = []
    while pool:
        rays = []
        for d in pool:
            cx, cy = xy(d["cam_lon"], d["cam_lat"])
            b = math.radians(d["bearing"])
            rays.append((cx, cy, math.sin(b), math.cos(b), d["confidence"]))
        keep = list(range(len(pool)))
        sol = None
        while len(keep) >= 2:
            sol = solve([rays[i] for i in keep])
            if not sol:
                break
            res = [residual(sol, rays[i]) for i in keep]
            worst = max(range(len(keep)), key=lambda k: res[k][0] if res[k][1] > 0 else 1e9)
            r_perp, along = res[worst]
            if r_perp <= args.residuo and along > 0:
                break
            keep.pop(worst)
            sol = None
        ok = (sol is not None and len(keep) >= 2
              and baseline([rays[i] for i in keep]) >= args.base
              and spread_deg([rays[i] for i in keep]) >= args.angulo)
        if not ok:   # sem geometria: cada deteção restante fica como "1 vista"
            for d in pool:
                objs.append({"metodo": "1 vista", "dets": [d], "x": None})
            break
        members = [pool[i] for i in keep]
        rr = [residual(sol, rays[i]) for i in keep]
        objs.append({"metodo": "triangulado", "dets": members, "x": sol[0], "y": sol[1],
                     "rms": math.sqrt(st.mean(r[0] ** 2 for r in rr)),
                     "angulo": spread_deg([rays[i] for i in keep]),
                     "base": baseline([rays[i] for i in keep]),
                     "dist_tri": [r[1] for r in rr]})
        pool = [d for i, d in enumerate(pool) if i not in keep]
        if len(pool) < 2:
            for d in pool:
                objs.append({"metodo": "1 vista", "dets": [d], "x": None})
            break
    return objs


def table(lines, head, rows, legenda):
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    lines += ["", f"*{legenda}*", ""]
    w = [max(len(str(x)) for x in col) for col in zip(head, *rows)]
    for r in [head] + rows:
        print("  ".join(str(c).ljust(k) for c, k in zip(r, w)))
    print(f"({legenda})\n")


def m_(v):
    return f"{v:.1f} m".replace(".", ",")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="owlv2")
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--raio", type=float, default=8, help="junta posições de 1 foto a <= raio (m)")
    ap.add_argument("--residuo", type=float, default=1.5, help="linha a mais disto do ponto sai (m)")
    ap.add_argument("--base", type=float, default=3, help="distância mínima entre fotos (m)")
    ap.add_argument("--angulo", type=float, default=8, help="ângulo mínimo entre linhas (graus)")
    args = ap.parse_args()

    with open(DATA_DIR / "meta" / "sample_images.csv", encoding="utf-8") as f:
        seq = {r["id"]: r["sequence"] for r in csv.DictReader(f)}
    groups = defaultdict(list)
    with open(DATA_DIR / "preds" / f"{args.model}_geo.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["confidence"] < args.min_conf:
                continue
            key = (seq.get(d["image_id"]), d["category"],
                   d["subtype"] if d["category"] != "iluminacao" else "")
            groups[key].append(d)

    objs = []
    for (s, cat, sub), dets in groups.items():
        for items in cluster(dets, args.raio):
            for o in triangulate(items, args):
                o.update({"sequencia": s, "categoria": cat, "subtipo": sub})
                objs.append(o)

    # posição final e escrita
    out = DATA_DIR / "preds" / f"{args.model}_tri.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for o in objs:
            if o["x"] is None:
                d = o["dets"][0]
                o["lon"], o["lat"] = d["lon"], d["lat"]
            else:
                o["lon"], o["lat"] = to_lonlat(o["x"], o["y"])
            f.write(json.dumps({
                "sequencia": o["sequencia"], "categoria": o["categoria"], "subtipo": o["subtipo"],
                "metodo": o["metodo"], "lon": round(o["lon"], 7), "lat": round(o["lat"], 7),
                "n_fotos": len(o["dets"]), "rms_m": round(o.get("rms", 0), 2),
                "angulo_graus": round(o.get("angulo", 0), 1), "base_m": round(o.get("base", 0), 1),
                "imagens": [d["image_id"] for d in o["dets"]]}, ensure_ascii=False) + "\n")

    lines = ["# Triangulação multivista (piloto)", "",
             f"{sum(len(v) for v in groups.values())} deteções · {len(set(seq.values()))} passagens.", ""]
    cats = ["sinalizacao", "iluminacao", "residuos"]
    tri = [o for o in objs if o["metodo"] == "triangulado"]

    # 1. quantos objetos ficaram triangulados
    print("=== 1. Objetos triangulados")
    lines += ["## 1. Quantos objetos foram triangulados?", ""]
    rows = []
    for c in cats:
        oc = [o for o in objs if o["categoria"] == c]
        tc = [o for o in oc if o["metodo"] == "triangulado"]
        if oc:
            rows.append([NOMES[c], len(oc), len(tc), f"{len(tc) / len(oc):.0%}",
                         f"{st.median(len(o['dets']) for o in tc):.0f}" if tc else "—",
                         m_(st.median(o["rms"] for o in tc)) if tc else "—"])
    table(lines, ["Categoria", "Objetos", "Triangulados", "%", "Fotos por objeto", "Desvio típico das linhas"],
          rows, "Triangulado = visto em >= 2 fotos afastadas, com linhas a cruzarem-se. "
                "Desvio = quanto as linhas falham o ponto (menor = mais coerente).")

    # 2. erro da distância de uma só foto, medido contra a triangulação
    print("=== 2. Erro da distância de uma só foto")
    lines += ["## 2. Quanto erra a distância calculada com uma só foto?", "",
              "Comparação de cada foto com a distância medida pela triangulação "
              "(só objetos triangulados com desvio <= 1 m).", ""]
    rows = []
    for c in cats:
        e = [d["dist_m"] - dt for o in tri if o["categoria"] == c and o["rms"] <= 1
             for d, dt in zip(o["dets"], o["dist_tri"])]
        if len(e) >= 5:
            rel = [abs(x) for x in e]
            rows.append([NOMES[c], len(e), m_(st.median(rel)),
                         f"{sum(x <= 2 for x in rel) / len(rel):.0%}",
                         ("mais longe" if st.median(e) > 0 else "mais perto") + f" ({m_(abs(st.median(e)))})"])
    table(lines, ["Categoria", "Fotos", "Erro típico", "Erra até 2 m", "Tendência"], rows,
          "Erro típico = metade das fotos erra menos do que isto. Tendência = se uma só foto "
          "costuma pôr o objeto mais longe ou mais perto do que ele está.")

    # 3. repetibilidade entre passagens
    print("=== 3. Repetibilidade entre passagens")
    lines += ["## 3. Passagens de dias diferentes dão o mesmo sítio?", ""]
    rows = []
    for c in cats:
        by_seq = defaultdict(list)
        for o in tri:
            if o["categoria"] == c:
                by_seq[o["sequencia"]].append(o)
        seqs = sorted(by_seq, key=lambda s: -len(by_seq[s]))
        if len(seqs) < 2:
            continue
        a, others = by_seq[seqs[0]], [o for s in seqs[1:] for o in by_seq[s]]
        dd = []
        for o in a:
            same = [q for q in others if q["subtipo"] == o["subtipo"]]
            if same:
                d = min(math.hypot(o["x"] - q["x"], o["y"] - q["y"]) for q in same)
                if d <= 5:
                    dd.append(d)
        if dd:
            rows.append([NOMES[c], len(dd), m_(st.median(dd)), f"{sum(x <= 1 for x in dd) / len(dd):.0%}"])
    table(lines, ["Categoria", "Objetos nas 2 passagens", "Diferença típica", "Diferença até 1 m"], rows,
          "O mesmo objeto, triangulado com fotos de dias diferentes. Diferença pequena = medição repetível.")

    # 4. contra Mapillary/OSM (como antes), 1 foto vs triangulado
    print("=== 4. Contra pontos Mapillary/OSM")
    lines += ["## 4. Comparação com os pontos Mapillary/OpenStreetMap", ""]
    trees = reference_index()
    rows = []
    for c in cats:
        if c not in trees:
            continue
        tree, ref = trees[c]

        def err(lon, lat):
            px, py = xy(lon, lat)
            cand = tree.query(Point(px, py).buffer(20))
            return min((math.hypot(xy(*ref[i][:2])[0] - px, xy(*ref[i][:2])[1] - py) for i in cand),
                       default=None)
        oc = [o for o in tri if o["categoria"] == c]
        single = [e for o in oc for d in o["dets"] if (e := err(d["lon"], d["lat"])) is not None]
        trid = [e for o in oc if (e := err(*to_lonlat(o["x"], o["y"]))) is not None]
        if single and trid:
            rows.append([NOMES[c], f"{sum(x <= 5 for x in single) / len(single):.0%}",
                         f"{sum(x <= 5 for x in trid) / len(trid):.0%}",
                         m_(st.median(single)), m_(st.median(trid))])
    table(lines, ["Categoria", "1 foto: até 5 m", "Triangulado: até 5 m", "1 foto: erro típico",
                  "Triangulado: erro típico"], rows,
          "Mesmos objetos. A referência também erra alguns metros: serve para comparar, não é a verdade.")

    rep = DATA_DIR / "relatorios"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "triangulacao.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {out}\n-> {rep / 'triangulacao.md'}")


if __name__ == "__main__":
    main()
