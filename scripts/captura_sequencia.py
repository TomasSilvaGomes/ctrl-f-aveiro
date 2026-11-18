"""
Fotos SEGUIDAS (frames consecutivos) de uma rua -> pasta de dados própria, pronta para a
pipeline e para a triangulação. Dois modos:

  Por ponto (--lat --lon): procura a passagem mais perto do ponto (API Mapillary; funciona em
      qualquer município, sem índice): 360° primeiro, depois a mais recente, depois a mais perto.
      Ficam --n fotos seguidas a partir da mais próxima (o sentido não interessa).
  Por rua (--inicio --direcao): a rua começa em --inicio, segue na direção de --direcao durante
      --comprimento m. Ficam TODAS as fotos a <= --largura m do eixo, das sequências com pelo
      menos --min-fotos fotos (usa data/meta/aveiro_index.csv; por omissão só 360°).

Para cada foto: resolução original, metadados da pipeline, marcações Mapillary (SÓ para medir)
e EXIF com GPS, direção da câmara e atribuição CC BY-SA 4.0 (autor + id).

  <saida>/images/*.jpg                       fotos com EXIF
  <saida>/meta/sample_images.csv             metadados (mesmo formato da amostra)
  <saida>/meta/manifesto.json                ordem das fotos, posição, rumo, atribuição
  <saida>/ground_truth/mly_detections.jsonl  marcações Mapillary nas fotos (só para medir)
  <saida>/ground_truth/*.geojson             referências Mapillary/OSM (cópia, só para medir)

Uso:
    python scripts/captura_sequencia.py --lat 40.631135 --lon -8.648327 --n 100
    python scripts/captura_sequencia.py --lat 40.643331 --lon -8.645127 --tipo frontal --saida data/piloto_frontal
    python scripts/captura_sequencia.py --inicio 40.643331,-8.645127 --direcao 40.643185,-8.645803 --saida data/piloto_rua
Depois: python scripts/pipeline.py --dados <saida> --triangulacao
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from comum import ROOT  # noqa: E402
from mapillary import GRAPH, fetch_one, session, token  # noqa: E402

MAIN_DATA = ROOT / "data"


# ------------------------------------------------------------------ EXIF
def dms(v):
    """Graus decimais -> graus, minutos, segundos (formato EXIF)."""
    v = abs(v)
    d = int(v)
    m = int((v - d) * 60)
    s = Fraction((v - d - m / 60) * 3600).limit_denominator(10000)
    return ((d, 1), (m, 1), (s.numerator, s.denominator))


def write_exif(path, meta):
    import piexif  # MIT
    lat, lon = float(meta["lat"]), float(meta["lon"])
    ang = Fraction(float(meta["compass_angle"] or 0) % 360).limit_denominator(100)
    attribution = f"Mapillary / {meta['creator_username']} (CC BY-SA 4.0) - ID: {meta['id']}"
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
    print(f"{'sequência':<24}{'fotos':>6}{'espaço':>9}  data")
    for s, v in sorted(seqs.items(), key=lambda kv: -len(kv[1])):
        if len(v) < args.min_fotos:
            continue
        v.sort(key=lambda z: z[0])
        gaps = sorted(q - p for (p, _), (q, _) in zip(v, v[1:]))
        date = datetime.fromtimestamp(int(v[0][1]["captured_at"]) / 1000, timezone.utc).date()
        print(f"{s:<24}{len(v):>6}{gaps[len(gaps) // 2]:>7.1f} m  {date}")
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
def main():
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
    ap.add_argument("--tipo", choices=["360", "frontal", "qualquer"], default="360")
    ap.add_argument("--size", choices=["1024", "2048", "original"], default="original")
    ap.add_argument("--saida", default=str(MAIN_DATA / "piloto_sequencia"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
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


if __name__ == "__main__":
    main()
