"""
Amostra de fotos Mapillary de um município, em 3 passos (cada um é um comando):

  indice       lista TODAS as fotos do concelho (mosaicos de cobertura, zoom 14) e estima
               o espaço em disco          -> data/meta/<concelho>_index.csv
  selecionar   escolhe fotos URBANAS onde, à partida, aparecem as 3 categorias (pontos
               Mapillary/OSM à frente da câmara; longe de autoestradas; sem pedidos de imagem)
                                          -> data/meta/urban_candidates.csv
  descarregar  descarrega as fotos + metadados (pose, câmara, autor) + marcações Mapillary
               (SÓ para medir)            -> data/images, data/meta/sample_images.csv,
                                             data/ground_truth/mly_detections.jsonl

Uso:
    python scripts/amostra.py indice --concelho Aveiro
    python scripts/amostra.py selecionar
    python scripts/amostra.py descarregar --ids-from data/meta/sample_images.csv --n 100000 --size original

Para outra pasta de dados (ex.: ruas piloto): $env:DATA_DIR = "data\\piloto_rua" antes.
A Mapillary já desfoca rostos/matrículas; as fotos ficam só em disco local (no .gitignore).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import mercantile
import requests
from shapely.geometry import LineString, Point
from shapely.prepared import prep
from shapely.strtree import STRtree

sys.path.insert(0, os.path.dirname(__file__))
from comum import DATA_DIR  # noqa: E402
from mapillary import GRAPH, TILE_URL, UA, fetch_one, get_boundary, session, tile_points, token  # noqa: E402

META_DIR = DATA_DIR / "meta"


def year_of(ms):
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).year if ms else None


# ================================================================== indice
def fetch_index(boundary, tok):
    poly = prep(boundary)
    tiles = list(mercantile.tiles(*boundary.bounds, 14))   # camada "image" só existe no zoom 14
    print(f"A ler {len(tiles)} mosaicos (zoom 14)...")
    images: dict[int, dict] = {}
    s = session()
    for i, t in enumerate(tiles, 1):
        url = TILE_URL.format(z=t.z, x=t.x, y=t.y, token=tok)
        for attempt in range(3):
            r = s.get(url, timeout=30)
            if r.status_code == 200:
                break
            if r.status_code in (401, 403):
                sys.exit(f"Token rejeitado ({r.status_code}). Confirma o Client Token no .env.")
            time.sleep(2 * (attempt + 1))
        else:
            print(f"  mosaico {t} falhou ({r.status_code}), a saltar")
            continue
        if r.content:
            for lon, lat, p in tile_points(t, r.content):
                if p.get("id") in images or not poly.contains(Point(lon, lat)):
                    continue
                images[p["id"]] = {"id": p["id"], "lon": round(lon, 7), "lat": round(lat, 7),
                                   "captured_at": p.get("captured_at"), "compass_angle": p.get("compass_angle"),
                                   "is_pano": bool(p.get("is_pano")), "sequence_id": p.get("sequence_id"),
                                   "creator_id": p.get("creator_id")}
        if i % 25 == 0 or i == len(tiles):
            print(f"  {i}/{len(tiles)} mosaicos, {len(images)} imagens até agora")
    return list(images.values())


def sample_sizes(images, tok, n):
    """Mede o tamanho real das miniaturas 1024/2048 numa amostra aleatória."""
    sizes = {f: {"pano": [], "persp": []} for f in ("thumb_1024_url", "thumb_2048_url")}
    s = session()
    for img in random.Random(42).sample(images, min(n, len(images))):
        r = s.get(f"{GRAPH}/{img['id']}", timeout=30,
                  params={"access_token": tok, "fields": "thumb_1024_url,thumb_2048_url"})
        if r.status_code != 200:
            continue
        kind = "pano" if img["is_pano"] else "persp"
        for field, url in r.json().items():
            if field in sizes and url:
                t = s.get(url, timeout=60)
                if t.status_code == 200:
                    sizes[field][kind].append(len(t.content))
    return sizes


def cmd_indice(args):
    tok = token()
    boundary, label = get_boundary(args.concelho)
    print(f"Limite: {label}")
    images = fetch_index(boundary, tok)
    if not images:
        sys.exit("Nenhuma imagem encontrada.")
    META_DIR.mkdir(parents=True, exist_ok=True)
    out = META_DIR / f"{args.concelho.lower()}_index.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(images[0].keys()))
        w.writeheader()
        w.writerows(images)
    print(f"Índice guardado em {out}")

    years = Counter(year_of(i["captured_at"]) for i in images)
    print(f"\nImagens: {len(images):,} | panorâmicas 360°: {sum(i['is_pano'] for i in images):,} | "
          f"sequências: {len({i['sequence_id'] for i in images}):,}")
    print("Por ano: " + ", ".join(f"{y}: {c:,}" for y, c in sorted(years.items(), key=lambda x: (x[0] is None, x[0]))))
    if not args.sample:
        return
    print(f"\nA medir tamanhos numa amostra de {args.sample} imagens...")
    sizes = sample_sizes(images, tok, args.sample)
    now = datetime.now(timezone.utc).year
    subsets = {"todas": images, f"desde {now - 3}": [i for i in images if (year_of(i["captured_at"]) or 0) >= now - 3]}
    for field, lab in (("thumb_1024_url", "1024 px"), ("thumb_2048_url", "2048 px")):
        avg = {k: (statistics.mean(v) if v else None) for k, v in sizes[field].items()}
        fb = next((a for a in avg.values() if a), None)
        if not fb:
            continue
        for name, sub in subsets.items():
            p = sum(i["is_pano"] for i in sub)
            total = p * (avg["pano"] or fb) + (len(sub) - p) * (avg["persp"] or fb)
            print(f"  {lab} {name:>12}: {len(sub):>9,} imagens -> {total / 1e9:6.1f} GB")


# ================================================================== selecionar
LAT0 = 40.64  # projeção local equiretangular (erro desprezável à escala do concelho)
KX, KY = 111320 * math.cos(math.radians(LAT0)), 110540


def xy(lon, lat):
    return lon * KX, lat * KY


def motorways(concelho):
    q = f"""
    [out:json][timeout:120];
    area["boundary"="administrative"]["admin_level"="7"]["name"="{concelho}"]->.a;
    way["highway"~"^(motorway|motorway_link|trunk|trunk_link)$"](area.a);
    out geom;
    """
    r = requests.post("https://overpass-api.de/api/interpreter", data={"data": q}, headers=UA, timeout=180)
    r.raise_for_status()
    return [LineString([xy(p["lon"], p["lat"]) for p in w["geometry"]])
            for w in r.json()["elements"] if len(w.get("geometry", [])) > 1]


def reference_points():
    pts = []  # (x, y, categoria, subtipo, fonte)
    for name in ("mapillary_features_aveiro.geojson", "osm_aveiro.geojson"):
        with open(DATA_DIR / "ground_truth" / name, encoding="utf-8") as f:
            for feat in json.load(f)["features"]:
                p = feat["properties"]
                if p["subtype"] in ("poste", "poste_eletrico"):  # postes de eletricidade não contam
                    continue
                lon, lat = feat["geometry"]["coordinates"]
                pts.append((*xy(lon, lat), p["category"], p["subtype"], p["source"]))
    return pts


def angle_diff(a, b):
    return abs((a - b + 180) % 360 - 180)


def cmd_selecionar(args):
    print("Autoestradas/vias rápidas (OSM)...")
    mw = motorways(args.concelho)
    mw_tree = STRtree(mw)
    refs = reference_points()
    ref_tree = STRtree([Point(x, y) for x, y, *_ in refs])
    print(f"  {len(mw)} troços · pontos de referência: {len(refs):,}")

    W, S, E, N = map(float, args.bbox.split(","))
    cands, stats = [], defaultdict(int)
    with open(META_DIR / f"{args.concelho.lower()}_index.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stats["total"] += 1
            if not row["captured_at"] or not row["compass_angle"]:
                continue
            if not (W <= float(row["lon"]) <= E and S <= float(row["lat"]) <= N):
                continue
            year = year_of(row["captured_at"])
            if year < args.since:
                continue
            is_pano = row["is_pano"] == "True"
            stats["recentes_cidade"] += 1
            x, y = xy(float(row["lon"]), float(row["lat"]))
            cam = Point(x, y)
            if any(mw[i].distance(cam) < args.motorway_m for i in mw_tree.query(cam.buffer(args.motorway_m))):
                stats["autoestrada"] += 1
                continue
            heading = float(row["compass_angle"])
            counts, subs = defaultdict(int), set()
            for i in ref_tree.query(cam.buffer(args.radius)):
                rx, ry, cat, sub, src = refs[i]
                d = math.hypot(rx - x, ry - y)
                if d > args.radius or d < 2:
                    continue
                bearing = math.degrees(math.atan2(rx - x, ry - y)) % 360
                if is_pano or angle_diff(bearing, heading) <= args.fov / 2:   # 360° vê tudo
                    counts[cat] += 1
                    if cat == "residuos" and src == "osm":
                        subs.add(sub)
            if all(counts[c] for c in ("sinalizacao", "iluminacao", "residuos")):
                stats["com_3_categorias"] += 1
                cands.append({**row, "x": x, "y": y, "year": year, "n_sinal": counts["sinalizacao"],
                              "n_ilum": counts["iluminacao"], "n_resid": counts["residuos"],
                              "subtipos_osm": "|".join(sorted(subs))})

    # desbaste por sequência (frames seguidos são quase iguais)
    cands.sort(key=lambda c: -(c["n_resid"] * 3 + c["n_sinal"] + c["n_ilum"]))
    kept, by_seq = [], defaultdict(list)
    for c in cands:
        if all(math.hypot(c["x"] - k["x"], c["y"] - k["y"]) >= args.spacing for k in by_seq[c["sequence_id"]]):
            by_seq[c["sequence_id"]].append(c)
            kept.append(c)
    out = META_DIR / "urban_candidates.csv"
    cols = ["id", "lon", "lat", "captured_at", "year", "is_pano", "compass_angle", "sequence_id",
            "n_sinal", "n_ilum", "n_resid", "subtipos_osm"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(kept)
    print(f"Índice {stats['total']:,} · cidade desde {args.since}: {stats['recentes_cidade']:,} · "
          f"autoestrada: {stats['autoestrada']:,} · com as 3 categorias: {stats['com_3_categorias']:,} · "
          f"após desbaste ({args.spacing:.0f} m): {len(kept):,} em {len(by_seq)} sequências\n-> {out}")


# ================================================================== descarregar
def spread(index_csv, n, since, pano, step, seed):
    """Amostra espalhada: 1 foto em cada `step` de cada sequência, depois sorteio de n."""
    by_seq = defaultdict(list)
    with open(index_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row["is_pano"] == "True") != pano or not row["captured_at"]:
                continue
            if year_of(row["captured_at"]) < since:
                continue
            by_seq[row["sequence_id"]].append((int(row["captured_at"]), row["id"]))
    pool = []
    for frames in by_seq.values():
        frames.sort()
        pool.extend(fid for _, fid in frames[::step])
    random.Random(seed).shuffle(pool)
    print(f"Candidatas: {len(pool):,} em {len(by_seq)} sequências (1 em cada {step}) -> {min(n, len(pool))}")
    return pool[:n]


def cmd_descarregar(args):
    tok = token()
    if args.ids_from:
        with open(args.ids_from, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if args.only_pano:
            rows = [r for r in rows if (r.get("is_pano") == "True") == (args.only_pano == "yes")]
        ids = [r["id"] for r in rows][: args.n]
        print(f"{len(ids)} imagens de {args.ids_from}")
    else:
        ids = spread(META_DIR / "aveiro_index.csv", args.n, args.since, args.pano, args.step, args.seed)
    img_dir, det_path = DATA_DIR / "images", DATA_DIR / "ground_truth" / "mly_detections.jsonl"
    img_dir.mkdir(parents=True, exist_ok=True)
    det_path.parent.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    metas, n_det, errors = [], 0, 0
    thumb = f"thumb_{args.size}_url"
    with ThreadPoolExecutor(args.workers) as ex, det_path.open("w", encoding="utf-8") as fdet:
        futs = {ex.submit(fetch_one, i, tok, thumb, img_dir): i for i in ids}
        for k, fut in enumerate(as_completed(futs), 1):
            try:
                meta, dets = fut.result()
            except Exception as e:  # noqa: BLE001
                errors += 1
                print(f"  erro {futs[fut]}: {e}")
                continue
            metas.append(meta)
            for det in dets:
                fdet.write(json.dumps({"image_id": meta["id"], **det}) + "\n")
            n_det += len(dets)
            if k % 100 == 0 or k == len(ids):
                print(f"  {k}/{len(ids)} imagens, {n_det} marcações Mapillary")
    with (META_DIR / "sample_images.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(metas[0].keys()))
        w.writeheader()
        w.writerows(metas)
    mb = sum((img_dir / m["file"]).stat().st_size for m in metas) / 1e6
    print(f"\n{len(metas)} imagens ({mb:.0f} MB) em {img_dir} · {n_det} marcações Mapillary"
          + (f" · erros: {errors}" if errors else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("indice", help="lista todas as fotos do concelho + espaço em disco")
    p.add_argument("--concelho", default="Aveiro")
    p.add_argument("--sample", type=int, default=40, help="fotos para medir o tamanho (0 = não medir)")
    p = sub.add_parser("selecionar", help="fotos urbanas com as 3 categorias à vista")
    p.add_argument("--concelho", default="Aveiro")
    p.add_argument("--since", type=int, default=2019)
    p.add_argument("--bbox", default="-8.680,40.615,-8.615,40.660", help="W,S,E,N (por omissão a cidade)")
    p.add_argument("--radius", type=float, default=30)
    p.add_argument("--fov", type=float, default=90)
    p.add_argument("--motorway-m", type=float, default=40)
    p.add_argument("--spacing", type=float, default=15)
    p = sub.add_parser("descarregar", help="fotos + metadados + marcações Mapillary")
    p.add_argument("--ids-from", help="CSV com coluna 'id' (ex.: data/meta/sample_images.csv)")
    p.add_argument("--only-pano", choices=["yes", "no"], help="filtra o --ids-from por tipo")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--since", type=int, default=2024)
    p.add_argument("--pano", action="store_true", help="(sem --ids-from) só panorâmicas 360°")
    p.add_argument("--step", type=int, default=10)
    p.add_argument("--size", choices=["1024", "2048", "original"], default="2048")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    {"indice": cmd_indice, "selecionar": cmd_selecionar, "descarregar": cmd_descarregar}[args.cmd](args)


if __name__ == "__main__":
    main()
