"""
Tudo o que DESCARREGA dados (Mapillary, OpenStreetMap, modelos). Um comando por tarefa:

  amostra      índice das fotos do município, escolha da amostra espalhada e descarga
               (subcomandos: indice | selecionar | descarregar)
  sequencia    fotos seguidas à volta de um ponto ou ao longo de uma rua (ruas piloto)
  referencias  pontos Mapillary/OSM e marcações nas fotos — SÓ para medir, nunca no catálogo
  poses        rotação real da câmara e pontos 3D da Mapillary (meta/poses.csv, sfm/<id>.npz)
  modelos      descarrega os modelos de IA para a cache (uma vez, numa instalação nova)
  video        vídeo gravado na rua + trilho GPX -> uma foto a cada N metros (como a Mapillary)
  fotos        fotos tiradas com o telemóvel (GPS e bússola no EXIF) -> formato das amostras

Uso:
    python scripts/dados.py amostra indice
    python scripts/dados.py amostra selecionar
    python scripts/dados.py amostra descarregar
    python scripts/dados.py sequencia --lat 40.64 --lon -8.65 --saida data/piloto_x
    python scripts/dados.py referencias
    python scripts/dados.py poses
    python scripts/dados.py modelos
    python scripts/dados.py video gravacao.mp4 --gpx trilho.gpx --saida data/video_rua
    python scripts/dados.py fotos C:\pasta_das_fotos --saida data/fotos_rua
    python scripts/dados.py <comando> -h      # opções de cada comando
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import statistics
import sys
import time
import zlib
from collections import Counter, defaultdict
from concurrent.futures import as_completed, ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import mercantile
import numpy as np
import requests
from shapely.geometry import LineString, Point
from shapely.prepared import prep
from shapely.strtree import STRtree

sys.path.insert(0, os.path.dirname(__file__))
from comum import DATA_DIR, ROOT, rodrigues  # noqa: E402
from mapillary import GRAPH, MLY_TILES, MLY_URL, TILE_URL, UA, categorize_mly, fetch_one, get_boundary, session, tile_points, token  # noqa: E402


####################################################################################################
# AMOSTRA  (antes: scripts/amostra.py)
####################################################################################################
# Amostra de fotos Mapillary de um município, em 3 passos (cada um é um comando):
#
#   indice       lista TODAS as fotos do concelho (mosaicos de cobertura, zoom 14) e estima
#                o espaço em disco          -> data/meta/<concelho>_index.csv
#   selecionar   escolhe fotos URBANAS onde, à partida, aparecem as 3 categorias (pontos
#                Mapillary/OSM à frente da câmara; longe de autoestradas; sem pedidos de imagem)
#                                           -> data/meta/urban_candidates.csv
#   descarregar  descarrega as fotos + metadados (pose, câmara, autor) + marcações Mapillary
#                (SÓ para medir)            -> data/images, data/meta/sample_images.csv,
#                                              data/ground_truth/mly_detections.jsonl
#
# Uso:
#     python scripts/dados.py amostra indice --concelho Aveiro
#     python scripts/dados.py amostra selecionar
#     python scripts/dados.py amostra descarregar --ids-from data/meta/sample_images.csv --n 100000 --size original
#
# Para outra pasta de dados (ex.: ruas piloto): $env:DATA_DIR = "data\\piloto_rua" antes.
# A Mapillary já desfoca rostos/matrículas; as fotos ficam só em disco local (no .gitignore).


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


def main_amostra(argv=None):
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
    args = ap.parse_args(argv)
    {"indice": cmd_indice, "selecionar": cmd_selecionar, "descarregar": cmd_descarregar}[args.cmd](args)


####################################################################################################
# SEQUENCIA  (antes: scripts/captura_sequencia.py)
####################################################################################################
# Fotos SEGUIDAS (frames consecutivos) de uma rua -> pasta de dados própria, pronta para a
# pipeline e para a triangulação. Dois modos:
#
#   Por ponto (--lat --lon): procura a passagem mais perto do ponto (API Mapillary; funciona em
#       qualquer município, sem índice): 360° primeiro, depois a mais recente, depois a mais perto.
#       Ficam --n fotos seguidas a partir da mais próxima (o sentido não interessa).
#   Por rua (--inicio --direcao): a rua começa em --inicio, segue na direção de --direcao durante
#       --comprimento m. Ficam TODAS as fotos a <= --largura m do eixo, das sequências com pelo
#       menos --min-fotos fotos (usa data/meta/aveiro_index.csv; por omissão só 360°).
#
# Para cada foto: resolução original, metadados da pipeline, marcações Mapillary (SÓ para medir)
# e EXIF com GPS, direção da câmara e atribuição CC BY-SA 4.0 (autor + id).
#
#   <saida>/images/*.jpg                       fotos com EXIF
#   <saida>/meta/sample_images.csv             metadados (mesmo formato da amostra)
#   <saida>/meta/manifesto.json                ordem das fotos, posição, rumo, atribuição
#   <saida>/ground_truth/mly_detections.jsonl  marcações Mapillary nas fotos (só para medir)
#   <saida>/ground_truth/*.geojson             referências Mapillary/OSM (cópia, só para medir)
#
# Uso:
#     python scripts/dados.py sequencia --lat 40.631135 --lon -8.648327 --n 100
#     python scripts/dados.py sequencia --lat 40.643331 --lon -8.645127 --tipo frontal --saida data/piloto_frontal
#     python scripts/dados.py sequencia --inicio 40.643331,-8.645127 --direcao 40.643185,-8.645803 --saida data/piloto_rua
# Depois: python scripts/pipeline.py --dados <saida> --triangulacao


MAIN_DATA = ROOT / "data"


# ------------------------------------------------------------------ EXIF
def dms(v):
    """Graus decimais -> graus, minutos, segundos (formato EXIF)."""
    v = abs(v)
    d = int(v)
    m = int((v - d) * 60)
    s = Fraction((v - d - m / 60) * 3600).limit_denominator(10000)
    return ((d, 1), (m, 1), (s.numerator, s.denominator))


def write_exif(path, meta, fonte="Mapillary", licenca="CC BY-SA 4.0"):
    import piexif  # MIT
    lat, lon = float(meta["lat"]), float(meta["lon"])
    ang = Fraction(float(meta["compass_angle"] or 0) % 360).limit_denominator(100)
    attribution = f"{fonte} / {meta['creator_username']} ({licenca}) - ID: {meta['id']}"
    exif = {
        "0th": {piexif.ImageIFD.Artist: (meta["creator_username"] or "?").encode("utf-8"),
                piexif.ImageIFD.Copyright: attribution.encode("utf-8")},
        "GPS": {piexif.GPSIFD.GPSLatitudeRef: b"N" if lat >= 0 else b"S",
                piexif.GPSIFD.GPSLatitude: dms(lat),
                piexif.GPSIFD.GPSLongitudeRef: b"E" if lon >= 0 else b"W",
                piexif.GPSIFD.GPSLongitude: dms(lon),
                piexif.GPSIFD.GPSImgDirectionRef: b"T",
                piexif.GPSIFD.GPSImgDirection: (ang.numerator, ang.denominator)},
    }
    piexif.insert(piexif.dump(exif), str(path))
    return attribution


# ------------------------------------------------------------------ escolha das fotos
def by_point(args, tok):
    s = session()

    def get(url, **params):
        r = s.get(url, params={"access_token": tok, **params}, timeout=30)
        r.raise_for_status()
        js = r.json()
        if "error" in js:
            raise RuntimeError(js["error"].get("message"))
        return js

    kx, ky = 111320 * math.cos(math.radians(args.lat)), 110540
    d = args.raio / ky * 1.5
    near = get(f"{GRAPH}/images", bbox=f"{args.lon - d},{args.lat - d},{args.lon + d},{args.lat + d}",
               fields="id,sequence,computed_geometry,geometry,captured_at,is_pano", limit=500)["data"]

    def dist(img):
        g = (img.get("computed_geometry") or img.get("geometry") or {}).get("coordinates", [0, 0])
        return math.hypot((g[0] - args.lon) * kx, (g[1] - args.lat) * ky)
    near = [i for i in near if i.get("sequence") and dist(i) <= args.raio]
    if args.tipo != "qualquer":
        near = [i for i in near if bool(i.get("is_pano")) == (args.tipo == "360")]
    if not near:
        sys.exit(f"Nenhuma foto ({args.tipo}) a <= {args.raio:.0f} m deste ponto.")
    start = min(near, key=lambda i: (not i.get("is_pano"), -int(i.get("captured_at") or 0), dist(i)))
    seq = start["sequence"]
    print(f"Foto inicial {start['id']} a {dist(start):.0f} m · passagem {seq} · "
          f"{'360°' if start.get('is_pano') else 'normal'}")
    ids = [x["id"] for x in get(f"{GRAPH}/image_ids", sequence_id=seq)["data"]]
    if start["id"] not in ids:
        sys.exit("A foto inicial não aparece na lista da passagem (resposta inesperada da API).")
    k = ids.index(start["id"])
    lo = max(0, min(k, len(ids) - args.n))
    print(f"Passagem com {len(ids)} fotos; ficam {min(args.n, len(ids))} seguidas (posições {lo}–{lo + args.n - 1})")
    return ids[lo:lo + args.n]


def by_street(args, out):
    a = [float(v) for v in args.inicio.split(",")]
    b = [float(v) for v in args.direcao.split(",")]
    kx, ky = 111320 * math.cos(math.radians(a[0])), 110540

    def xy(lat, lon):
        return (lon - a[1]) * kx, (lat - a[0]) * ky
    bx, by = xy(*b)
    n = math.hypot(bx, by)
    ux, uy = bx / n, by / n
    print(f"Rua: rumo {math.degrees(math.atan2(ux, uy)) % 360:.0f}°, {args.comprimento:.0f} m")
    want_pano = {"360": True, "frontal": False}.get(args.tipo)
    seqs = defaultdict(list)
    with open(MAIN_DATA / "meta" / "aveiro_index.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if want_pano is not None and (r["is_pano"] == "True") != want_pano:
                continue
            x, y = xy(float(r["lat"]), float(r["lon"]))
            t, d = x * ux + y * uy, abs(-x * uy + y * ux)  # ao longo / ao lado da rua
            if -20 <= t <= args.comprimento and d <= args.largura:
                seqs[r["sequence_id"]].append((t, r))
    rows = []
    escolhidas = []
    for s, v in seqs.items():
        if len(v) < args.min_fotos:
            continue
        por_hora = sorted(v, key=lambda z: int(z[1]["captured_at"] or 0))
        avanca = por_hora[-1][0] - por_hora[0][0]           # > 0: anda do início para a direção
        if args.sentido and avanca <= 0:
            continue
        escolhidas.append((max(t for t, _ in v) - min(t for t, _ in v), len(v), s, v))
    escolhidas.sort(reverse=True)                            # as que cobrem mais rua primeiro
    if args.passagens:
        escolhidas = escolhidas[: args.passagens]
    print(f"{'sequência':<24}{'fotos':>6}{'cobre':>8}{'espaço':>9}  data")
    for span, _, s, v in escolhidas:
        v.sort(key=lambda z: z[0])
        gaps = sorted(q - p for (p, _), (q, _) in zip(v, v[1:]))
        date = datetime.fromtimestamp(int(v[0][1]["captured_at"]) / 1000, timezone.utc).date()
        print(f"{s:<24}{len(v):>6}{span:>6.0f} m{gaps[len(gaps) // 2] if gaps else 0:>7.1f} m  {date}")
        rows += [{"id": r["id"], "sequence_id": s, "ao_longo_m": round(t, 1), "is_pano": r["is_pano"]} for t, r in v]
    if not rows:
        sys.exit("Nenhuma sequência encontrada nesta rua.")
    (out / "meta").mkdir(parents=True, exist_ok=True)
    with open(out / "meta" / "rua_ids.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} fotos -> {out / 'meta' / 'rua_ids.csv'}")
    return [r["id"] for r in rows]


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception as e:  # noqa: BLE001
        print(f"  erro {a[0]}: {e}")
        return None


# ------------------------------------------------------------------ principal
def main_sequencia(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("por ponto")
    g.add_argument("--lat", type=float)
    g.add_argument("--lon", type=float)
    g.add_argument("--n", type=int, default=100, help="n.º de fotos seguidas")
    g.add_argument("--raio", type=float, default=30, help="procura fotos a <= raio m do ponto")
    g = ap.add_argument_group("por rua")
    g.add_argument("--inicio", help="lat,lon do início da rua")
    g.add_argument("--direcao", help="lat,lon de um ponto mais à frente na rua")
    g.add_argument("--comprimento", type=float, default=400, help="metros a partir do início")
    g.add_argument("--largura", type=float, default=12, help="distância máxima ao eixo (m)")
    g.add_argument("--min-fotos", type=int, default=30, help="sequências com menos fotos ficam de fora")
    g.add_argument("--so-selecionar", action="store_true", help="só escreve meta/rua_ids.csv (não descarrega)")
    g.add_argument("--sentido", action="store_true",
                   help="só passagens que andam do --inicio para o --direcao (um único sentido)")
    g.add_argument("--passagens", type=int, default=0,
                   help="fica só com as N passagens que cobrem mais rua (0 = todas)")
    ap.add_argument("--tipo", choices=["360", "frontal", "qualquer"], default="360")
    ap.add_argument("--size", choices=["1024", "2048", "original"], default="original")
    ap.add_argument("--saida", default=str(MAIN_DATA / "piloto_sequencia"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    if (args.lat is None) == (args.inicio is None):
        ap.error("indica --lat/--lon (por ponto) OU --inicio/--direcao (por rua)")
    if args.inicio and not args.direcao:
        ap.error("--inicio precisa de --direcao")

    out = Path(args.saida)
    tok = token()
    if args.inicio:
        chosen = by_street(args, out)
        if args.so_selecionar:
            return
    else:
        chosen = by_point(args, tok)

    for d in ("images", "meta", "ground_truth"):
        (out / d).mkdir(parents=True, exist_ok=True)
    thumb = f"thumb_{args.size}_url"
    metas, dets, errors = [], [], 0
    with ThreadPoolExecutor(args.workers) as ex:
        for res in ex.map(lambda x: _safe(fetch_one, x, tok, thumb, out / "images"), chosen):
            if res is None:
                errors += 1
                continue
            meta, ds = res
            meta["atribuicao"] = write_exif(out / "images" / meta["file"], meta)
            metas.append(meta)
            dets += [{"image_id": meta["id"], **x} for x in ds]
    metas.sort(key=lambda m: (str(m["sequence"]), int(m["captured_at"] or 0)))   # ordem real de captura

    with open(out / "meta" / "sample_images.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(metas[0].keys()))
        w.writeheader()
        w.writerows(metas)
    with open(out / "ground_truth" / "mly_detections.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in dets)
    lat0 = float(metas[0]["lat"])
    kx, ky = 111320 * math.cos(math.radians(lat0)), 110540
    steps = [math.hypot((float(b["lon"]) - float(a["lon"])) * kx, (float(b["lat"]) - float(a["lat"])) * ky)
             for a, b in zip(metas, metas[1:]) if a["sequence"] == b["sequence"]] or [0.0]
    manifesto = [{"ordem": n, "image_id": m["id"], "lat": m["lat"], "lon": m["lon"], "rumo": m["compass_angle"],
                  "captured_at": m["captured_at"], "is_pano": m["is_pano"], "sequencia": m["sequence"],
                  "ficheiro": f"images/{m['file']}", "atribuicao": m["atribuicao"]}
                 for n, m in enumerate(metas, 1)]
    with open(out / "meta" / "manifesto.json", "w", encoding="utf-8") as f:
        json.dump(manifesto, f, ensure_ascii=False, indent=1)
    for name in ("osm_aveiro.geojson", "mapillary_features_aveiro.geojson"):
        src = MAIN_DATA / "ground_truth" / name
        if src.exists():
            shutil.copyfile(src, out / "ground_truth" / name)

    mb = sum((out / "images" / m["file"]).stat().st_size for m in metas) / 1e6
    print(f"{len(metas)} fotos ({mb:.0f} MB) · percurso {sum(steps):.0f} m · "
          f"espaço mediano {sorted(steps)[len(steps) // 2]:.1f} m · {len(dets)} marcações Mapillary"
          + (f" · erros: {errors}" if errors else ""))
    print(f"-> {out}")


####################################################################################################
# REFERENCIAS  (antes: scripts/fetch_ground_truth.py)
####################################################################################################
# Pontos de referência do concelho, SÓ para medir (nunca entram no catálogo; ODbL/Mapillary):
#   1. OpenStreetMap (Overpass): candeeiros, ecopontos, contentores, sinais.
#   2. Mapillary "points" (map features): objetos já posicionados pela Mapillary
#      (candeeiros, contentores, postes, sinais). Poste com luminária a <= 4 m = poste de luz.
#
#   --listar-tipos   em vez de descarregar, lista TODOS os tipos de points que a Mapillary tem
#                    no município, com quantos há de cada (-> data/relatorios/pontos_mapillary.md)
#
# Uso:
#     python scripts/dados.py referencias
#     python scripts/dados.py referencias --listar-tipos --municipio Ílhavo


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


def main_referencias(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listar-tipos", action="store_true", help="só lista os tipos de points da Mapillary")
    ap.add_argument("--municipio", default="Aveiro", help="(com --listar-tipos) outro município")
    args = ap.parse_args(argv)
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


####################################################################################################
# POSES  (antes: scripts/pose_3d.py)
####################################################################################################
# Pose real da câmara e profundidade 3D a partir da reconstrução da Mapillary (OpenSfM).
#
# Para cada foto da amostra:
#   1. computed_rotation (rotação real da câmara: rumo, inclinação, rolamento)
#        -> <DATA_DIR>/meta/poses.csv          (id, rotation, cluster)
#      O posicao.py calcular usa-a para corrigir a inclinação (fotos frontais inclinadas ~5–10°).
#   2. reconstrução 3D da passagem (sfm_cluster: pontos 3D + pose de cada foto). Os pontos são
#      projetados na foto (modelo de câmara OpenSfM) e guardados os que caem dentro dela:
#        -> <DATA_DIR>/sfm/<id>.npz   u, v (píxeis da miniatura), hdist (m na horizontal
#                                      até à câmara), dz (altura relativa à câmara), lon, lat
#      O posicao.py calcular --sfm usa os pontos dentro de cada caixa para medir a distância.
#
# Os pontos 3D são da reconstrução das FOTOS (não são marcações/points de objetos da Mapillary).
#
# Uso:
#     python scripts/dados.py poses                 # amostra principal
#     $env:DATA_DIR = "data\\piloto_rua"; python scripts/dados.py poses


MAX_DIST = 60.0


def project(P, cam, W, H):
    """Pontos no referencial da câmara (x direita, y baixo, z frente) -> píxeis (u, v), máscara."""
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    proj = cam.get("projection_type", "perspective")
    if proj in ("spherical", "equirectangular"):
        lon = np.arctan2(x, z)
        lat = np.arctan2(-y, np.hypot(x, z))
        u = (lon / (2 * np.pi) + 0.5) * W
        v = (0.5 - lat / np.pi) * H
        return u, v, np.ones(len(P), bool)
    ok = z > 0.1
    xn, yn = x / np.where(ok, z, 1), y / np.where(ok, z, 1)
    r = np.hypot(xn, yn)
    f, k1, k2 = cam.get("focal", 0.8), cam.get("k1", 0.0), cam.get("k2", 0.0)
    if proj == "fisheye":
        th = np.arctan(r)
        d = np.where(r > 1e-9, th / np.maximum(r, 1e-9), 1.0) * (1 + k1 * th ** 2 + k2 * th ** 4)
    else:
        d = 1 + k1 * r ** 2 + k2 * r ** 4
    s = max(W, H)
    return W / 2 + f * d * xn * s, H / 2 + f * d * yn * s, ok


def process_cluster(url, items):
    """items: [(meta, rotation)] das fotos desta passagem -> nº de fotos com pontos guardados.
    Corre num processo à parte (com threads o numpy/zlib chegou a rebentar no Windows)."""
    rec = json.loads(zlib.decompress(session().get(url, timeout=120).content))[0]
    shots = rec["shots"]
    keys = list(shots)
    rots = np.array([shots[k]["rotation"] for k in keys], float)
    pares = []
    for meta, rot in items:
        j = int(np.argmin(np.linalg.norm(rots - np.asarray(rot), axis=1)))
        if np.linalg.norm(rots[j] - rot) <= 1e-3:
            pares.append((meta, shots[keys[j]]))
    return guardar_pontos(rec, pares)


def guardar_pontos(rec, pares):
    """Reconstrução OpenSfM (da Mapillary ou nossa) + [(meta, shot)] -> sfm/<id>.npz por foto.
    Devolve o nº de fotos com pontos guardados."""
    ref = rec["reference_lla"]
    pts = np.array([p["coordinates"] for p in rec["points"].values()], float)
    if not len(pts):
        return 0
    done = 0
    for meta, shot in pares:
        R = rodrigues(shot["rotation"])
        t = np.asarray(shot["translation"], float)
        C = -R.T @ t                                     # centro da câmara (ENU da passagem)
        Pc = pts @ R.T + t                               # pontos no referencial da câmara
        W, H = float(meta["thumb_w"]), float(meta["thumb_h"])
        cam = rec["cameras"].get(shot["camera"], {})
        u, v, ok = project(Pc, cam, W, H)
        rel = pts - C
        hd = np.hypot(rel[:, 0], rel[:, 1])
        m = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H) & (hd <= MAX_DIST) & (hd > 0.5)
        if m.sum() < 5:
            continue
        lat0 = ref["latitude"]
        lon = ref["longitude"] + pts[m, 0] / (111320 * math.cos(math.radians(lat0)))
        lat = lat0 + pts[m, 1] / 110540
        np.savez_compressed(DATA_DIR / "sfm" / f"{meta['id']}.npz",
                            u=u[m].astype(np.float32), v=v[m].astype(np.float32),
                            hdist=hd[m].astype(np.float32), dz=rel[m, 2].astype(np.float32),
                            lon=lon, lat=lat)
        done += 1
    return done


# --opensfm: as NOSSAS fotos (vídeo, ou uma rua nova) não têm reconstrução da Mapillary. Corre o
# OpenSfM original (BSD-2, Mapillary) em Docker sobre as fotos de DATA_DIR e grava o mesmo
# meta/poses.csv + sfm/<id>.npz; troca também lon/lat/rumo da câmara pelos reconstruídos (o GPS
# fica em lon_gps/lat_gps), como a Mapillary faz. Nada vem da API: a rua fica só com dados nossos.
# O Docker só LÊ a pasta de dados (o Controlled Folder Access bloqueia escritas em Documentos);
# o resultado sai pelo stdout e é o Python que o grava em DATA_DIR/opensfm/reconstruction.json.
# Imagem Docker (uma vez, com o Docker Desktop ligado):
#     git clone --recurse-submodules https://github.com/mapillary/OpenSfM
#     docker build -f OpenSfM/Dockerfile.ubuntu24 -t opensfm:ctrlf OpenSfM
#
#     $env:DATA_DIR = "data\\video_rua"; python scripts/dados.py poses --opensfm

OPENSFM_PASSOS = ("extract_metadata", "detect_features", "match_features", "create_tracks", "reconstruct")


def correr_opensfm(metas, imagem, processos, dop):
    """Fotos de DATA_DIR -> lista de reconstruções OpenSfM (reconstruction.json)."""
    import subprocess
    ds = DATA_DIR / "opensfm"
    ds.mkdir(parents=True, exist_ok=True)
    m0 = metas[0]
    W, H = int(m0["thumb_w"]), int(m0["thumb_h"])
    if m0["is_pano"] == "True":
        cam = {"projection_type": "spherical", "width": W, "height": H}
    else:
        cp = json.loads(m0["camera_parameters"] or "[]")       # [focal, k1, k2] (focal / lado maior)
        f, k1, k2 = (cp + [0.85, 0.0, 0.0][len(cp):])[:3]
        cam = {"projection_type": "perspective", "width": W, "height": H, "focal": f, "k1": k1, "k2": k2}
    gps = lambda m, k: float(m.get(f"{k}_gps") or m[k])  # noqa: E731
    exif = {m["file"]: {"gps": {"latitude": gps(m, "lat"), "longitude": gps(m, "lon"), "altitude": 0.0, "dop": dop},
                        "capture_time": int(m["captured_at"]) / 1000} for m in metas}
    (ds / "exif_overrides.json").write_text(json.dumps(exif), encoding="utf-8")
    (ds / "camera_models_overrides.json").write_text(json.dumps({"all": cam}), encoding="utf-8")
    (ds / "image_list.txt").write_text("".join(f"/in/images/{m['file']}\n" for m in metas), encoding="utf-8")
    (ds / "config.yaml").write_text(
        f"processes: {processos}\nuse_altitude_tag: false\nmatching_gps_neighbors: 10\n"
        "matching_order_neighbors: 10\nfeature_process_size: 2048\n", encoding="utf-8")
    passos = " ".join(OPENSFM_PASSOS)
    script = ("mkdir -p /ds && cp /in/cfg/config.yaml /in/cfg/exif_overrides.json /in/cfg/camera_models_overrides.json "
              "/in/cfg/image_list.txt /ds/ && "
              f"for s in {passos}; do echo \"  OpenSfM: $s\" >&2; "
              "python bin/opensfm_main.py $s /ds > /ds/log.txt 2>&1 || { tail -20 /ds/log.txt >&2; exit 1; }; done && "
              "cat /ds/reconstruction.json")
    cmd = ["docker", "run", "--rm",
           "--mount", f"type=bind,source={(DATA_DIR / 'images').resolve()},target=/in/images,readonly",
           "--mount", f"type=bind,source={ds.resolve()},target=/in/cfg,readonly",
           imagem, "bash", "-c", script]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, encoding="utf-8")
    if r.returncode != 0 or not r.stdout.strip():
        sys.exit(f"OpenSfM falhou (código {r.returncode}). A imagem Docker '{imagem}' existe e o Docker está ligado?")
    (ds / "reconstruction.json").write_text(r.stdout, encoding="utf-8")
    return json.loads(r.stdout)


def main_poses_opensfm(args):
    t0 = time.time()
    with open(DATA_DIR / "meta" / "sample_images.csv", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    print(f"OpenSfM ({args.imagem}) em {len(linhas)} fotos de {DATA_DIR}", flush=True)
    recs = correr_opensfm(linhas, args.imagem, args.processos_sfm, args.dop)
    por_ficheiro = {m["file"]: m for m in linhas}
    (DATA_DIR / "sfm").mkdir(parents=True, exist_ok=True)
    poses, desvios, n_img = {}, [], 0
    for ci, rec in enumerate(recs):
        ref = rec["reference_lla"]
        kx = 111320 * math.cos(math.radians(ref["latitude"]))
        pares = []
        for nome, shot in rec["shots"].items():
            meta = por_ficheiro.get(nome)
            if meta is None:
                continue
            pares.append((meta, shot))
            poses[meta["id"]] = (shot["rotation"], f"opensfm_{ci}")
            R = rodrigues(shot["rotation"])
            C = -R.T @ np.asarray(shot["translation"], float)
            fwd = R.T @ np.array([0.0, 0.0, 1.0])            # eixo da câmara no mundo (E, N, U)
            if not meta.get("lon_gps"):
                meta["lon_gps"], meta["lat_gps"] = meta["lon"], meta["lat"]
            lon, lat = ref["longitude"] + C[0] / kx, ref["latitude"] + C[1] / 110540
            desvios.append(math.hypot((lon - float(meta["lon_gps"])) * kx, (lat - float(meta["lat_gps"])) * 110540))
            meta["lon"], meta["lat"] = round(lon, 8), round(lat, 8)
            meta["compass_angle"] = round(math.degrees(math.atan2(fwd[0], fwd[1])) % 360, 2)
        n_img += guardar_pontos(rec, pares)
    with open(DATA_DIR / "meta" / "poses.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "rotation", "cluster"])
        for m in linhas:
            rot, cl = poses.get(m["id"], (None, ""))
            w.writerow([m["id"], json.dumps(rot) if rot else "", cl])
    with open(DATA_DIR / "meta" / "sample_images.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)
    n_pts = sum(len(r["points"]) for r in recs)
    print(f"Reconstruídas {len(poses)}/{len(linhas)} fotos em {len(recs)} reconstrução(ões), {n_pts:,} pontos 3D "
          f"({time.time() - t0:.0f} s)\n  -> meta/poses.csv (rotação) · sfm/: {n_img} fotos com pontos · "
          f"posição da câmara: desvio mediano ao GPS {statistics.median(desvios) if desvios else 0:.1f} m "
          f"(o GPS fica em lon_gps/lat_gps)")


def main_poses(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--processos", type=int, default=6, help="processos para a parte 3D")
    ap.add_argument("--sem-3d", action="store_true", help="só as rotações (rápido)")
    ap.add_argument("--opensfm", action="store_true",
                    help="fotos NOSSAS: reconstrução própria com o OpenSfM em Docker (sem API da Mapillary)")
    ap.add_argument("--imagem", default="opensfm:ctrlf", help="imagem Docker do OpenSfM")
    ap.add_argument("--processos-sfm", type=int, default=8, help="processos do OpenSfM")
    ap.add_argument("--dop", type=float, default=5.0, help="erro do GPS em metros (telemóvel ~5)")
    args = ap.parse_args(argv)
    if args.opensfm:
        return main_poses_opensfm(args)
    tok = token()
    with open(DATA_DIR / "meta" / "sample_images.csv", encoding="utf-8") as f:
        metas = {r["id"]: r for r in csv.DictReader(f)}
    (DATA_DIR / "sfm").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def info(iid):
        r = session().get(f"{GRAPH}/{iid}", timeout=30,
                          params={"access_token": tok, "fields": "computed_rotation,sfm_cluster"})
        return iid, (r.json() if r.status_code == 200 else {})

    poses, clusters = {}, defaultdict(list)
    with ThreadPoolExecutor(args.workers) as ex:
        for k, (iid, js) in enumerate(ex.map(info, metas), 1):
            rot = js.get("computed_rotation")
            sc = js.get("sfm_cluster") or {}
            poses[iid] = (rot, sc.get("id", ""))
            if rot and sc.get("url"):
                clusters[sc["url"]].append((metas[iid], rot))
            if k % 500 == 0:
                print(f"  poses {k}/{len(metas)} ({time.time() - t0:.0f} s)", flush=True)
    with open(DATA_DIR / "meta" / "poses.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "rotation", "cluster"])
        for iid, (rot, cl) in poses.items():
            w.writerow([iid, json.dumps(rot) if rot else "", cl])
    print(f"Poses: {sum(1 for r, _ in poses.values() if r)}/{len(poses)} com rotação -> meta/poses.csv "
          f"· {len(clusters)} reconstruções 3D", flush=True)
    if args.sem_3d:
        return

    done = {f.stem for f in (DATA_DIR / "sfm").glob("*.npz")}
    clusters = {u: [it for it in items if it[0]["id"] not in done] for u, items in clusters.items()}
    clusters = {u: items for u, items in clusters.items() if items}
    print(f"Já feitas: {len(done)} fotos · a fazer: {len(clusters)} reconstruções", flush=True)
    n_img, n_cl = 0, 0
    with ProcessPoolExecutor(args.processos) as ex:
        futs = {ex.submit(process_cluster, url, items): url for url, items in clusters.items()}
        for fut in as_completed(futs):
            n_cl += 1
            try:
                n_img += fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  erro numa reconstrução: {str(e)[:80]}", flush=True)
            if n_cl % 50 == 0:
                print(f"  3D {n_cl}/{len(clusters)} reconstruções, {n_img} fotos com pontos ({time.time() - t0:.0f} s)",
                      flush=True)
    print(f"3D: {n_img}/{len(metas)} fotos com pontos -> {DATA_DIR / 'sfm'} ({time.time() - t0:.0f} s)")


####################################################################################################
# MODELOS  (antes: scripts/descarregar_modelos.py)
####################################################################################################
# Descarrega os modelos de IA para a cache do Hugging Face. Corre-se uma vez, numa instalação nova.
#
#   OWLv2   google/owlv2-base-patch16-ensemble   Apache 2.0   encontra objetos
#   SigLIP  google/siglip-so400m-patch14-384     Apache 2.0   tipo, limites de velocidade, código do sinal
#   SAM 2   facebook/sam2.1-hiera-large          Apache 2.0   contorno do objeto
#   EgoBlur Gen2 (Meta, caras + matrículas)      Apache 2.0   download manual (projectaria.com) -> weights/
#
# Uso:
#     python scripts/dados.py modelos
#     python scripts/dados.py modelos --so siglip

MODELOS = {
    "owlv2": "google/owlv2-base-patch16-ensemble",
    "siglip": "google/siglip-so400m-patch14-384",
    "sam2": "facebook/sam2.1-hiera-large",
}


def descarregar(nome):
    from huggingface_hub import snapshot_download
    t = time.time()
    path = snapshot_download(MODELOS[nome], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
    size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs) / 1e6
    print(f"{nome:<7} {MODELOS[nome]:<38} {size:>7.0f} MB  ({time.time() - t:.0f} s)")


def main_modelos(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", nargs="*", choices=list(MODELOS), default=list(MODELOS))
    args = ap.parse_args(argv)
    for nome in args.so:
        descarregar(nome)
    from comum import EGOBLUR_DIR, EGOBLUR_PESOS
    faltam = [n for n, _, _ in EGOBLUR_PESOS if not (EGOBLUR_DIR / n).exists()]
    if faltam:
        print(f"EgoBlur Gen2 (caras e matrículas): descarregar à mão em https://www.projectaria.com/tools/egoblur "
              f"(aceitar os termos) e pôr {' e '.join(faltam)} em {EGOBLUR_DIR}")


####################################################################################################
# VIDEO  (novo: vídeo gravado na rua -> sequência de fotos, como a Mapillary faz)
####################################################################################################
# Um vídeo + o trilho GPS (ficheiro GPX) -> uma foto a cada --passo metros ao longo do percurso,
# no mesmo formato das amostras (images/ + meta/sample_images.csv), pronta para a pipeline.
#
#   - escolha pela DISTÂNCIA (como o mapillary_tools: 1 foto / 3 m por omissão): com o carro
#     parado não sai nenhuma foto repetida;
#   - em cada ponto escolhe a imagem mais NÍTIDA numa janela de --janela segundos;
#   - posição = GPS interpolado no instante dessa imagem; rumo = sentido do movimento;
#   - câmara frontal (perspetiva); --focal = distância focal / maior lado da imagem
#     (telemóvel ~0,8; ação/GoPro linear ~0,6).
#   - caras e matrículas desfocadas (EgoBlur) ANTES de gravar; o vídeo original não é guardado.
#
# Uso:
#     python scripts/dados.py video gravacao.mp4 --gpx trilho.gpx --saida data/video_rua
#     python scripts/dados.py video gravacao.mp4 --gpx trilho.gpx --saida data/video_rua --passo 4 --desvio-s 1.5


def read_gpx(path):
    """GPX -> lista ordenada de (t_epoch_s, lat, lon)."""
    import xml.etree.ElementTree as ET
    pts = []
    for el in ET.parse(path).getroot().iter():
        if el.tag.endswith("trkpt") or el.tag.endswith("wpt"):
            tt = next((c.text for c in el if c.tag.endswith("time")), None)
            if tt:
                ts = datetime.fromisoformat(tt.replace("Z", "+00:00")).timestamp()
                pts.append((ts, float(el.get("lat")), float(el.get("lon"))))
    pts.sort()
    if len(pts) < 2:
        sys.exit(f"O GPX {path} não tem pontos com hora.")
    return pts


def data_do_video(path):
    """Hora (epoch UTC) em que o vídeo começou, lida pelo ffprobe: no iPhone a etiqueta
    com.apple.quicktime.creationdate (hora local + fuso); senão creation_time (UTC). None se não houver."""
    import re
    import subprocess
    exe = shutil.which("ffprobe")
    if not exe:
        return None
    r = subprocess.run([exe, "-v", "error", "-show_entries", "format_tags:stream_tags", "-of", "json", str(path)],
                       capture_output=True, text=True, encoding="utf-8")
    try:
        js = json.loads(r.stdout or "{}")
    except json.JSONDecodeError:
        return None
    tags = dict(js.get("format", {}).get("tags", {}))
    for s in js.get("streams", []):
        for k, v in s.get("tags", {}).items():
            tags.setdefault(k, v)
    for k in ("com.apple.quicktime.creationdate", "creation_time"):
        v = tags.get(k)
        if v:
            v = re.sub(r"([+-]\d\d)(\d\d)$", r"\1:\2", v.strip().replace("Z", "+00:00"))
            try:
                d = datetime.fromisoformat(v)
            except ValueError:
                continue
            return (d if d.tzinfo else d.replace(tzinfo=timezone.utc)).timestamp()
    return None


def main_video(argv=None):
    import cv2
    ap = argparse.ArgumentParser(description="vídeo + GPX -> fotos a cada N metros (formato das amostras)")
    ap.add_argument("video")
    ap.add_argument("--gpx", required=True, help="trilho GPS com hora (GoPro/telemóvel -> GPX)")
    ap.add_argument("--saida", required=True, help="pasta de dados nova (ex.: data/video_rua)")
    ap.add_argument("--passo", type=float, default=4.0, help="metros entre fotos (Mapillary: 3)")
    ap.add_argument("--janela", type=float, default=0.4, help="segundos à volta de cada ponto para escolher a mais nítida")
    ap.add_argument("--desvio-s", type=float, default=None,
                    help="à mão: o vídeo começa N s depois do 1.º ponto do GPX (por omissão: data gravada no vídeo)")
    ap.add_argument("--focal", type=float, default=0.8, help="distância focal / maior lado da imagem")
    ap.add_argument("--largura", type=int, default=1920, help="largura das fotos guardadas (px)")
    ap.add_argument("--autor", default="equipa")
    ap.add_argument("--licenca", default="CC BY-SA 4.0")
    ap.add_argument("--sem-desfocar", action="store_true",
                    help="SÓ PARA TESTES com fotos já desfocadas (ex.: da Mapillary)")
    args = ap.parse_args(argv)
    t0 = time.time()
    from comum import Desfocar
    try:
        desfocar = None if args.sem_desfocar else Desfocar()
    except FileNotFoundError as e:
        sys.exit(str(e))

    gpx = read_gpx(args.gpx)
    ts = np.array([g[0] for g in gpx])
    lat = np.array([g[1] for g in gpx])
    lon = np.array([g[2] for g in gpx])
    kx, ky = 111320 * math.cos(math.radians(lat.mean())), 110540
    step = np.hypot(np.diff(lon) * kx, np.diff(lat) * ky)
    dist = np.concatenate([[0], np.cumsum(step)])          # metros percorridos em cada ponto GPS

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Não consegui abrir {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = n_frames / fps
    if args.desvio_s is not None:
        start, como = ts[0] + args.desvio_s, f"--desvio-s {args.desvio_s:g}"
    else:
        start = data_do_video(args.video)                    # hora (UTC) do 1.º frame do vídeo
        if start is None:
            sys.exit("O vídeo não tem data de gravação (foi enviado por mensagem?). Usar o ficheiro original "
                     "ou dizer à mão quando começou: --desvio-s <segundos depois do início do GPX>")
        como = "data gravada no vídeo"
    hm = lambda t: datetime.fromtimestamp(t).strftime("%H:%M:%S")  # noqa: E731
    print(f"Vídeo {hm(start)}–{hm(start + dur)} ({como}) · GPX {hm(ts[0])}–{hm(ts[-1])}", flush=True)
    if start < ts[0] - 1 or start + dur > ts[-1] + 1:
        fora = max(ts[0] - start, start + dur - ts[-1])
        if fora > dur / 2:
            sys.exit("O vídeo não está dentro do trilho GPX (horas não batem certo). O GPX tem de começar antes "
                     "do vídeo e acabar depois; ou dizer à mão --desvio-s.")
        print(f"  aviso: {fora:.0f} s do vídeo ficam fora do GPX (sem posição): o GPX deve começar antes e "
              f"acabar depois do vídeo", flush=True)
    # pontos-alvo: a cada --passo metros, dentro do troço que o vídeo cobre
    d_ini = np.interp(start, ts, dist)
    d_fim = np.interp(start + dur, ts, dist)
    alvos_d = np.arange(d_ini, d_fim, args.passo)
    # instante em que se passa em cada ponto (1.ª vez: parado não conta)
    alvos_t = np.interp(alvos_d, dist, ts) - start
    janelas = [(max(0, int((a - args.janela) * fps)), int((a + args.janela) * fps)) for a in alvos_t]
    print(f"Vídeo {dur:.0f} s a {fps:.0f} fps · percurso {d_fim - d_ini:.0f} m · {len(janelas)} pontos a cada {args.passo:g} m",
          flush=True)

    out = Path(args.saida)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    stem = "".join(c if c.isalnum() else "_" for c in Path(args.video).stem)
    best = {}            # n.º do ponto -> (nitidez, frame, imagem)
    j, f_idx = 0, -1
    while j < len(janelas):
        if not cap.grab():
            break
        f_idx += 1
        while j < len(janelas) and f_idx > janelas[j][1]:
            j += 1
        for k in range(j, min(j + 3, len(janelas))):    # janelas próximas podem sobrepor-se
            a, b = janelas[k]
            if a <= f_idx <= b:
                ok, im = cap.retrieve()
                if not ok:
                    continue
                g = cv2.cvtColor(cv2.resize(im, (480, int(480 * im.shape[0] / im.shape[1]))), cv2.COLOR_BGR2GRAY)
                sharp = cv2.Laplacian(g, cv2.CV_64F).var()
                if k not in best or sharp > best[k][0]:
                    best[k] = (sharp, f_idx, im)
    cap.release()

    metas, usados, n_blur = [], set(), 0
    for k in sorted(best):
        _, f_idx, im = best[k]
        if f_idx in usados:          # dois pontos com a mesma imagem (vídeo lento): fica uma
            continue
        usados.add(f_idx)
        tg = start + f_idx / fps     # hora GPS desta imagem
        la, lo = float(np.interp(tg, ts, lat)), float(np.interp(tg, ts, lon))
        da = np.interp(tg, ts, dist)  # rumo = direção entre 5 m antes e 5 m depois
        a_lat, a_lon = np.interp(np.interp(da - 5, dist, ts), ts, lat), np.interp(np.interp(da - 5, dist, ts), ts, lon)
        b_lat, b_lon = np.interp(np.interp(da + 5, dist, ts), ts, lat), np.interp(np.interp(da + 5, dist, ts), ts, lon)
        rumo = math.degrees(math.atan2((b_lon - a_lon) * kx, (b_lat - a_lat) * ky)) % 360
        h, w = im.shape[:2]
        if w > args.largura:
            im = cv2.resize(im, (args.largura, round(h * args.largura / w)), interpolation=cv2.INTER_AREA)
            h, w = im.shape[:2]
        if desfocar:
            im, nb = desfocar(im)
            n_blur += nb
        iid = f"{stem}_{f_idx:06d}"
        fn = f"{iid}.jpg"
        cv2.imwrite(str(out / "images" / fn), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        meta = {"id": iid, "file": fn, "thumb_w": w, "thumb_h": h, "width": w, "height": h,
                "lon": round(lo, 8), "lat": round(la, 8), "lon_gps": round(lo, 8), "lat_gps": round(la, 8),
                "compass_angle": round(rumo, 2), "camera_type": "perspective",
                "camera_parameters": json.dumps([args.focal, 0.0, 0.0]), "is_pano": "False",
                "captured_at": int(tg * 1000), "sequence": stem, "creator_username": args.autor,
                "creator_id": "", "license": args.licenca, "source_url": "", "atribuicao": ""}
        meta["atribuicao"] = write_exif(out / "images" / fn, meta, fonte="Vídeo", licenca=args.licenca)
        metas.append(meta)

    with open(out / "meta" / "sample_images.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(metas[0].keys()))
        w.writeheader()
        w.writerows(metas)
    for name in ("osm_aveiro.geojson", "mapillary_features_aveiro.geojson"):   # só para medir
        src = ROOT / "data" / "ground_truth" / name
        if src.exists():
            (out / "ground_truth").mkdir(exist_ok=True)
            shutil.copyfile(src, out / "ground_truth" / name)
    print(f"{len(metas)} fotos ({time.time() - t0:.0f} s) · "
          + (f"{n_blur} caras/matrículas desfocadas" if desfocar else "SEM desfocagem (teste)") + f" -> {out}")
    print(f"A seguir: python scripts/pipeline.py --dados {out} --triangulacao --imagens 6")


####################################################################################################
# FOTOS  (fotos tiradas com o telemóvel, sem instalar nada)
####################################################################################################
# O vídeo da app Câmara do iPhone só guarda UM ponto GPS (o do início); cada FOTO guarda GPS,
# direção da bússola (GPSImgDirection), hora e distância focal. Uma pasta de fotos tiradas ao longo
# da rua (a pé, uma a cada poucos passos, ou em modo contínuo) -> mesmo formato das amostras.
#   - JPEG (iPhone: Definições > Câmara > Formatos > "Mais compatível", ou "Transferir para Mac ou
#     PC: Automático", que converte ao passar por cabo). HEIC não: o leitor traz código GPL.
#   - fotos ao alto ou deitadas (roda pela orientação EXIF); distância focal pela equivalente 35 mm.
#   - sem direção da bússola: rumo = sentido do movimento entre fotos seguidas.
#   - caras e matrículas desfocadas (EgoBlur) ANTES de gravar.
#
#     python scripts/dados.py fotos C:\pasta_das_fotos --saida data/fotos_rua


def exif_foto(path):
    """JPEG -> (imagem PIL já rodada, {lat, lon, rumo, t, f35}) a partir do EXIF."""
    from PIL import Image, ImageOps
    im = Image.open(path)
    ex = im.getexif()
    gps = ex.get_ifd(0x8825) if ex else {}
    sub = ex.get_ifd(0x8769) if ex else {}

    def graus(v, ref):
        d = float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
        return -d if ref in ("S", "W", b"S", b"W") else d
    info = {"lat": None, "lon": None, "rumo": None, "t": None, "f35": None}
    if gps.get(2) and gps.get(4):
        info["lat"], info["lon"] = graus(gps[2], gps.get(1, "N")), graus(gps[4], gps.get(3, "E"))
    if gps.get(17) is not None:
        info["rumo"] = float(gps[17]) % 360
    data = sub.get(0x9003) or ex.get(0x0132)                 # DateTimeOriginal / DateTime
    if data:
        fuso = sub.get(0x9011) or "+00:00"                   # OffsetTimeOriginal
        try:
            info["t"] = datetime.strptime(f"{data} {fuso}", "%Y:%m:%d %H:%M:%S %z").timestamp()
        except ValueError:
            pass
    if sub.get(0xA405):
        info["f35"] = float(sub[0xA405])                      # FocalLengthIn35mmFilm
    return ImageOps.exif_transpose(im).convert("RGB"), info


def main_fotos(argv=None):
    import cv2
    ap = argparse.ArgumentParser(description="pasta de fotos do telemóvel (com GPS no EXIF) -> formato das amostras")
    ap.add_argument("pasta")
    ap.add_argument("--saida", required=True, help="pasta de dados nova (ex.: data/fotos_rua)")
    ap.add_argument("--largura", type=int, default=1920, help="lado maior das fotos guardadas (px)")
    ap.add_argument("--autor", default="equipa")
    ap.add_argument("--licenca", default="CC BY-SA 4.0")
    ap.add_argument("--sem-desfocar", action="store_true", help="SÓ PARA TESTES com fotos já desfocadas")
    args = ap.parse_args(argv)
    t0 = time.time()
    fs = sorted(p for p in Path(args.pasta).iterdir() if p.suffix.lower() in (".jpg", ".jpeg"))
    heic = [p for p in Path(args.pasta).iterdir() if p.suffix.lower() in (".heic", ".heif")]
    if heic:
        print(f"  aviso: {len(heic)} fotos HEIC ignoradas (passar em JPEG: iPhone > Câmara > Formatos > Mais compatível)")
    if not fs:
        sys.exit("Nenhuma foto JPEG na pasta.")
    from comum import Desfocar
    try:
        desfocar = None if args.sem_desfocar else Desfocar()
    except FileNotFoundError as e:
        sys.exit(str(e))
    lidas, sem_gps = [], []
    for p in fs:
        im, info = exif_foto(p)
        (lidas if info["lat"] is not None else sem_gps).append((p, im, info))
    if sem_gps:
        print(f"  aviso: {len(sem_gps)} fotos sem GPS ignoradas (iPhone: Definições > Privacidade > "
              f"Serviços de localização > Câmara = Durante a utilização): {', '.join(p.name for p, _, _ in sem_gps[:5])}")
    if not lidas:
        sys.exit("Nenhuma foto com GPS no EXIF (o chat/WhatsApp tiram-no: usar os ficheiros originais).")
    lidas.sort(key=lambda z: (z[2]["t"] or 0, z[0].name))
    out = Path(args.saida)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    seq = "".join(c if c.isalnum() else "_" for c in Path(args.pasta).name)
    kx = 111320 * math.cos(math.radians(lidas[0][2]["lat"]))
    metas, n_blur, n_rumo = [], 0, 0
    for k, (p, im, info) in enumerate(lidas):
        rumo = info["rumo"]
        if rumo is None:                                     # sem bússola: sentido do movimento
            a = lidas[max(0, k - 1)][2]
            b = lidas[min(len(lidas) - 1, k + 1)][2]
            rumo = math.degrees(math.atan2((b["lon"] - a["lon"]) * kx, (b["lat"] - a["lat"]) * 110540)) % 360
            n_rumo += 1
        im.thumbnail((args.largura, args.largura))
        bgr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        if desfocar:
            bgr, nb = desfocar(bgr)
            n_blur += nb
        h, w = bgr.shape[:2]
        iid = f"{seq}_{k:04d}"
        fn = f"{iid}.jpg"
        cv2.imwrite(str(out / "images" / fn), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        # distância focal / lado maior: equivalente 35 mm é sobre a diagonal (43,27 mm)
        focal = info["f35"] / 43.27 * math.hypot(w, h) / max(w, h) if info["f35"] else 0.8
        meta = {"id": iid, "file": fn, "thumb_w": w, "thumb_h": h, "width": w, "height": h,
                "lon": round(info["lon"], 8), "lat": round(info["lat"], 8),
                "lon_gps": round(info["lon"], 8), "lat_gps": round(info["lat"], 8),
                "compass_angle": round(rumo, 2), "camera_type": "perspective",
                "camera_parameters": json.dumps([round(focal, 4), 0.0, 0.0]), "is_pano": "False",
                "captured_at": int((info["t"] or 0) * 1000), "sequence": seq, "creator_username": args.autor,
                "creator_id": "", "license": args.licenca, "source_url": "", "atribuicao": ""}
        meta["atribuicao"] = write_exif(out / "images" / fn, meta, fonte="Fotos", licenca=args.licenca)
        metas.append(meta)
    with open(out / "meta" / "sample_images.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(metas[0].keys()))
        w.writeheader()
        w.writerows(metas)
    for name in ("osm_aveiro.geojson", "mapillary_features_aveiro.geojson"):   # só para medir
        src = ROOT / "data" / "ground_truth" / name
        if src.exists():
            (out / "ground_truth").mkdir(exist_ok=True)
            shutil.copyfile(src, out / "ground_truth" / name)
    passo = [math.hypot((b["lon"] - a["lon"]) * kx, (b["lat"] - a["lat"]) * 110540) for a, b in zip(metas, metas[1:])]
    print(f"{len(metas)} fotos ({time.time() - t0:.0f} s) · uma a cada {statistics.median(passo) if passo else 0:.1f} m · "
          f"rumo pela bússola em {len(metas) - n_rumo}, pelo movimento em {n_rumo} · "
          + (f"{n_blur} caras/matrículas desfocadas" if desfocar else "SEM desfocagem (teste)") + f" -> {out}")
    print(f"A seguir: python scripts/pipeline.py --dados {out} --triangulacao --altura-camara 1.4")


####################################################################################################
SECOES = {
    "amostra": (main_amostra, "índice, escolha e descarga da amostra"),
    "sequencia": (main_sequencia, "fotos seguidas (ponto ou rua)"),
    "referencias": (main_referencias, "pontos Mapillary/OSM só para medir"),
    "poses": (main_poses, "rotação da câmara e pontos 3D"),
    "modelos": (main_modelos, "descarrega os modelos de IA"),
    "video": (main_video, "vídeo + GPX -> fotos a cada N metros"),
    "fotos": (main_fotos, "fotos do telemóvel (GPS no EXIF) -> formato das amostras"),
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in SECOES:
        print(__doc__)
        print("Comandos: " + ", ".join(SECOES))
        sys.exit(0 if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help") else 1)
    cmd = sys.argv[1]
    fn = SECOES[cmd][0]
    # nos comandos que já têm subcomando próprio (lambda) o argparse acrescenta o nome sozinho
    sys.argv[0] = os.path.basename(sys.argv[0]) + ("" if fn.__name__ == "<lambda>" else f" {cmd}")
    fn(sys.argv[2:])


if __name__ == "__main__":
    main()
