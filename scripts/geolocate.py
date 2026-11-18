"""
3.ª fase: calcula as coordenadas (lat/lon) de cada deteção classificada.

  rumo      = rumo da câmara + ângulo horizontal do objeto na imagem
  distância = plano do chão (candeeiros, contentores: a base da caixa toca no chão)
              ou tamanho conhecido (sinais: ~0,65 m de largura)
  posição   = posição da câmara projetada (rumo, distância)
  fotos normais: direções corrigidas da distorção da lente (camera_parameters da
              Mapillary); caixas encostadas às bordas laterais (--borda) são ignoradas
  largura   = distância x largura angular da caixa (metros). Se a taxonomia tiver
              "largura_min_m" num subtipo e o objeto for mais estreito, passa para o
              subtipo "senao", ou é descartado se senao = "descartar"
              (ex.: subterrâneo < 1,2 m é quase sempre uma papeleira -> descartado).

Valida contra os pontos de mapa da Mapillary/OSM (vizinho mais próximo da mesma
categoria) e verifica a convenção de rumo (erro angular médio deve ser ~0°).

Entrada:  data/preds/<modelo>_classified.jsonl
Saída:    data/preds/<modelo>_geo.jsonl

Uso:
    python scripts/geolocate.py --model owlv2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict

import yaml
from shapely.geometry import Point
from shapely.strtree import STRtree

sys.path.insert(0, os.path.dirname(__file__))
from comum import DATA_DIR  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SIGN_WIDTH_M = 0.65
LAT0 = 40.64
KX, KY = 111320 * math.cos(math.radians(LAT0)), 110540


def xy(lon, lat):
    return lon * KX, lat * KY


def project(lon, lat, bearing_deg, dist_m):
    b = math.radians(bearing_deg)
    return (lon + dist_m * math.sin(b) / (111320 * math.cos(math.radians(lat))),
            lat + dist_m * math.cos(b) / 110540)


def geometry(det, meta, h_pano, h_persp):
    """Devolve (rumo, distância, método) ou None."""
    x0, y0, x1, y1 = det["bbox"]
    W, H = float(meta["thumb_w"]), float(meta["thumb_h"])
    heading = float(meta["compass_angle"])
    xc = (x0 + x1) / 2
    is_sign = det["category"] == "sinalizacao"

    if meta["is_pano"] == "True":
        bearing = heading + (xc / W - 0.5) * 360
        if is_sign:
            ang = (x1 - x0) / W * 2 * math.pi
            return bearing % 360, SIGN_WIDTH_M / ang if ang > 0 else None, "tamanho"
        below = -(0.5 - y1 / H) * math.pi  # ângulo da base abaixo do horizonte
        if below < math.radians(0.5):
            return bearing % 360, None, "chao"
        return bearing % 360, h_pano / math.tan(below), "chao"

    xu, _ = undistort(xc, y1, meta)
    bearing = heading + math.degrees(math.atan(xu))
    if is_sign:
        ang = angular_width(det, meta)
        return bearing % 360, SIGN_WIDTH_M / ang if ang > 0 else None, "tamanho"
    xb, yb = undistort(xc, y1, meta)
    below = math.atan2(yb, math.sqrt(1 + xb * xb))   # ângulo da base abaixo do horizonte
    if below < math.radians(0.5):
        return bearing % 360, None, "chao"
    return bearing % 360, h_persp / math.tan(below), "chao"


def undistort(px, py, meta):
    """Píxel de uma foto normal -> direção sem distorção da lente (x/z, y/z), com o modelo
    de câmara da Mapillary/OpenSfM: camera_parameters = [focal, k1, k2] (focal relativa ao
    lado maior). 'perspective': r_d = r_u (1 + k1 r_u² + k2 r_u⁴); 'fisheye':
    r_d = θ (1 + k1 θ² + k2 θ⁴), com θ = atan(r_u)."""
    W, H = float(meta["thumb_w"]), float(meta["thumb_h"])
    p = json.loads(meta.get("camera_parameters") or "null") or [0.8]
    focal = float(p[0]) or 0.8
    k1, k2 = (float(p[1]), float(p[2])) if len(p) >= 3 else (0.0, 0.0)
    xd = (px - W / 2) / max(W, H) / focal
    yd = (py - H / 2) / max(W, H) / focal
    rd = math.hypot(xd, yd)
    if rd < 1e-9 or (k1 == 0 and k2 == 0 and meta.get("camera_type") != "fisheye"):
        return xd, yd
    if meta.get("camera_type") == "fisheye":
        th = rd
        for _ in range(20):
            th = rd / (1 + k1 * th ** 2 + k2 * th ** 4)
        ru = math.tan(min(th, 1.5))
    else:
        ru = rd
        for _ in range(20):
            ru = rd / (1 + k1 * ru ** 2 + k2 * ru ** 4)
    return xd * ru / rd, yd * ru / rd


def at_border(det, meta, frac):
    """Caixa encostada às bordas laterais de uma foto normal (onde a lente mais distorce)."""
    if meta["is_pano"] == "True" or frac <= 0:
        return False
    W = float(meta["thumb_w"])
    return det["bbox"][0] < frac * W or det["bbox"][2] > (1 - frac) * W


def angular_width(det, meta):
    """Largura angular da caixa (radianos)."""
    x0, y0, x1, y1 = det["bbox"]
    W = float(meta["thumb_w"])
    if meta["is_pano"] == "True":
        return (x1 - x0) / W * 2 * math.pi
    yc = (y0 + y1) / 2
    return math.atan(undistort(x1, yc, meta)[0]) - math.atan(undistort(x0, yc, meta)[0])


def box_width_m(bbox, meta, h_pano=2.2, h_persp=1.66):
    """Largura real (m) de um objeto pousado no chão, a partir da caixa. None se não der."""
    g = geometry({"bbox": bbox, "category": "residuos"}, meta, h_pano, h_persp)
    if not g or g[1] is None:
        return None
    return g[1] * angular_width({"bbox": bbox}, meta)


# Referência dos contentores: a Mapillary marca papeleiras (~0,4–0,6 m) como "trash-can";
# a ficha não as pede. Nas medições só contam caixotes com pelo menos esta largura.
MIN_CONTENTOR_M = 0.9


def size_rules():
    """{(categoria, subtipo): (largura mínima, subtipo alternativo ou "descartar", info)}."""
    with open(os.path.join(ROOT, "config", "taxonomia.yaml"), encoding="utf-8") as f:
        tax = yaml.safe_load(f)
    rules = {}
    for cat, spec in tax["categorias"].items():
        subs = {s["id"]: s for s in spec["subtipos"]}
        for s in spec["subtipos"]:
            if s.get("largura_min_m") and s.get("senao") == "descartar":
                rules[(cat, s["id"])] = (s["largura_min_m"], "descartar", None)
            elif s.get("largura_min_m") and s.get("senao") in subs:
                rules[(cat, s["id"])] = (s["largura_min_m"], s["senao"], subs[s["senao"]])
    return rules


def reference_index():
    """Pontos de referência por categoria (Mapillary + OSM), sem postes de eletricidade."""
    pts = defaultdict(list)
    for name in ("mapillary_features_aveiro.geojson", "osm_aveiro.geojson"):
        with open(DATA_DIR / "ground_truth" / name, encoding="utf-8") as f:
            for feat in json.load(f)["features"]:
                p = feat["properties"]
                if p["subtype"] in ("poste", "poste_eletrico"):  # postes sem luminária não contam
                    continue
                pts[p["category"]].append((*feat["geometry"]["coordinates"], p))
    trees = {c: (STRtree([Point(*xy(lon, lat)) for lon, lat, _ in v]), v) for c, v in pts.items()}
    return trees


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="owlv2")
    ap.add_argument("--h-pano", type=float, default=2.2,
                    help="altura da câmara 360° (m) — 2.2 calibrado contra pontos Mapillary")
    ap.add_argument("--h-persp", type=float, default=1.95,
                    help="altura da câmara normal/frontal (m) — 1.95 sugerido pela calibração (17/11)")
    ap.add_argument("--max-dist", type=float, default=35)
    ap.add_argument("--borda", type=float, default=0.03,
                    help="fotos normais: ignora caixas a menos desta fração das bordas laterais (0 = não)")
    ap.add_argument("--match-m", type=float, default=20)
    args = ap.parse_args()

    with open(DATA_DIR / "meta" / "sample_images.csv", encoding="utf-8") as f:
        imgs = {r["id"]: r for r in csv.DictReader(f)}
    trees = reference_index()
    rules = size_rules()
    n_resized = defaultdict(int)
    n_border = 0

    out_path = DATA_DIR / "preds" / f"{args.model}_geo.jsonl"
    n_in = n_out = 0
    errs = defaultdict(list)       # categoria -> erros (m) até à referência mais próxima
    bearing_err = defaultdict(list)
    ratios = defaultdict(list)     # (tipo câmara) -> distância real / estimada (método chão)
    unmatched = defaultdict(int)

    with open(DATA_DIR / "preds" / f"{args.model}_classified.jsonl", encoding="utf-8") as fin, \
            out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            det = json.loads(line)
            n_in += 1
            meta = imgs.get(det["image_id"])
            if not meta or not meta["lon"]:
                continue
            if at_border(det, meta, args.borda):
                n_border += 1
                continue
            g = geometry(det, meta, args.h_pano, args.h_persp)
            if not g or g[1] is None or not (1 <= g[1] <= args.max_dist):
                continue
            bearing, dist, method = g
            cam_lon, cam_lat = float(meta["lon"]), float(meta["lat"])
            lon, lat = project(cam_lon, cam_lat, bearing, dist)
            width = dist * angular_width(det, meta)
            rule = rules.get((det["category"], det["subtype"]))
            if rule and width < rule[0]:
                n_resized[f"{det['subtype']} -> {rule[1]}"] += 1
                if rule[1] == "descartar":
                    continue
                det.update({"subtype": rule[1], "label": rule[2]["label"],
                            "codigo": rule[2].get("codigo")})
            det.update({"lon": round(lon, 7), "lat": round(lat, 7), "dist_m": round(dist, 1),
                        "largura_m": round(width, 2),
                        "bearing": round(bearing, 1), "geo_method": method,
                        "cam_lon": cam_lon, "cam_lat": cam_lat,
                        "pano": meta["is_pano"] == "True"})
            fout.write(json.dumps(det, ensure_ascii=False) + "\n")
            n_out += 1

            # validação
            cat = det["category"]
            if cat not in trees:
                continue
            tree, ref = trees[cat]
            px, py = xy(lon, lat)
            cand = tree.query(Point(px, py).buffer(args.match_m))
            if not len(cand):
                unmatched[cat] += 1
                continue
            j = min(cand, key=lambda i: math.hypot(xy(*ref[i][:2])[0] - px, xy(*ref[i][:2])[1] - py))
            rx, ry = xy(*ref[j][:2])
            errs[cat].append(math.hypot(rx - px, ry - py))
            cx, cy = xy(cam_lon, cam_lat)
            true_bearing = math.degrees(math.atan2(rx - cx, ry - cy)) % 360
            bearing_err[cat].append((bearing - true_bearing + 180) % 360 - 180)
            if method == "chao":
                true_d = math.hypot(rx - cx, ry - cy)
                ratios["360" if det["pano"] else "normal"].append(true_d / dist)

    print(f"Deteções classificadas: {n_in:,}  |  com coordenadas (1–{args.max_dist:.0f} m): {n_out:,}")
    print(f"Ignoradas por estarem encostadas às bordas (fotos normais, {args.borda:.0%}): {n_border}")
    for k, n in n_resized.items():
        print(f"Regra de tamanho: {k}: {n}")
    print(f"-> {out_path}\n")
    print(f"Validação: referência mais próxima da mesma categoria a <= {args.match_m:.0f} m "
          "(Mapillary + OSM)")
    print(f"{'categoria':<13}{'c/ ref':>8}{'s/ ref':>8}{'erro mediano':>14}{'<5 m':>7}{'<10 m':>7}"
          f"{'erro rumo med.':>16}")
    for cat in ("sinalizacao", "iluminacao", "residuos"):
        e = errs.get(cat, [])
        if not e:
            continue
        b = bearing_err[cat]
        print(f"{cat:<13}{len(e):>8,}{unmatched[cat]:>8,}{statistics.median(e):>12.1f} m"
              f"{sum(x < 5 for x in e) / len(e):>7.0%}{sum(x < 10 for x in e) / len(e):>7.0%}"
              f"{statistics.median(b):>14.1f}°")
    for k, r in ratios.items():
        if len(r) > 20:
            m = statistics.median(r)
            h = args.h_pano if k == "360" else args.h_persp
            print(f"\nCalibração ({k}): distância real/estimada mediana = {m:.2f} "
                  f"-> altura de câmara sugerida ≈ {h * m:.2f} m (atual {h} m)")
    print("\nNota: o vizinho mais próximo favorece erros baixos onde há muitos pontos "
          "(ex.: candeeiros); a referência também tem erro próprio (alguns metros).")


if __name__ == "__main__":
    main()
