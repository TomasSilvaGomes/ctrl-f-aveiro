"""
Captura N fotos SEGUIDAS (frames consecutivos) de uma passagem Mapillary, a partir
de um ponto (lat, lon). Funciona em qualquer município (não precisa do índice).

  1. Procura fotos a <= --raio m do ponto e escolhe a passagem: 360° primeiro, depois
     a mais recente, depois a mais perto.
  2. Lista todas as fotos dessa passagem e fica com --n seguidas a partir da foto mais
     próxima (o sentido não interessa). Se a passagem acabar, completa para trás.
  3. Descarrega cada foto (resolução original) com os metadados que a pipeline usa
     e as marcações da Mapillary (só para medir), e grava no EXIF da foto:
     GPS e direção da câmara (posição corrigida pela Mapillary) e a atribuição
     CC BY-SA 4.0 (autor + id).

Cria uma pasta de dados própria (usar com $env:DATA_DIR = <saida>):
  <saida>/images/*.jpg                       fotos com EXIF
  <saida>/meta/sample_images.csv             metadados (mesmo formato da amostra)
  <saida>/meta/manifesto.json                ordem das fotos, posição, rumo, atribuição
  <saida>/ground_truth/mly_detections.jsonl  marcações Mapillary nas fotos (só para medir)
  <saida>/ground_truth/*.geojson             referências Mapillary/OSM (cópia, só para medir)

Ideia de captura por coordenadas adaptada de um script da equipa.

Uso:
    python scripts/captura_sequencia.py --lat 40.631135 --lon -8.648327 --n 100
    python scripts/captura_sequencia.py --lat 40.643331 --lon -8.645127 --tipo frontal --saida data/piloto_frontal
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRAPH = "https://graph.mapillary.com"


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--n", type=int, default=100, help="n.º de fotos seguidas")
    ap.add_argument("--raio", type=float, default=30, help="procura fotos a <= raio m do ponto")
    ap.add_argument("--tipo", choices=["360", "frontal", "qualquer"], default="360",
                    help="tipo de passagem: 360° (por omissão), frontal (sem 360°) ou qualquer")
    ap.add_argument("--size", choices=["1024", "2048", "original"], default="original")
    ap.add_argument("--saida", default=str(ROOT / "data" / "piloto_sequencia"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.saida)
    os.environ["DATA_DIR"] = str(out)        # os módulos da pipeline passam a usar esta pasta
    sys.path.insert(0, os.path.dirname(__file__))
    from concurrent.futures import ThreadPoolExecutor
    from download_sample import fetch_one, session  # noqa: E402
    from estimate_mapillary import ROOT as _R  # noqa: E402,F401  (carrega o .env)

    token = os.getenv("MAPILLARY_TOKEN", "").strip()
    if not token or "COLA_AQUI" in token:
        sys.exit("Falta o MAPILLARY_TOKEN no ficheiro .env")
    s = session()

    def get(url, **params):
        r = s.get(url, params={"access_token": token, **params}, timeout=30)
        r.raise_for_status()
        js = r.json()
        if "error" in js:
            raise RuntimeError(js["error"].get("message"))
        return js

    # 1. passagem: fotos perto do ponto
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

    # 2. fotos seguidas da passagem
    ids = [x["id"] for x in get(f"{GRAPH}/image_ids", sequence_id=seq)["data"]]
    if start["id"] not in ids:
        sys.exit("A foto inicial não aparece na lista da passagem (resposta inesperada da API).")
    k = ids.index(start["id"])
    lo = max(0, min(k, len(ids) - args.n))
    chosen = ids[lo:lo + args.n]
    print(f"Passagem com {len(ids)} fotos; ficam {len(chosen)} seguidas (posições {lo}–{lo + len(chosen) - 1})")

    # 3. descarregar (metadados + marcações Mapillary) e gravar EXIF
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "ground_truth").mkdir(parents=True, exist_ok=True)
    thumb = f"thumb_{args.size}_url"
    metas, dets, errors = [], [], 0
    with ThreadPoolExecutor(args.workers) as ex:
        for i, res in zip(chosen, ex.map(lambda x: _safe(fetch_one, x, token, thumb), chosen)):
            if res is None:
                errors += 1
                continue
            meta, ds = res
            meta["atribuicao"] = write_exif(out / "images" / meta["file"], meta)
            metas.append(meta)
            dets += [{"image_id": meta["id"], **x} for x in ds]
    metas.sort(key=lambda m: int(m["captured_at"] or 0))   # ordem real de captura

    with open(out / "meta" / "sample_images.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(metas[0].keys()))
        w.writeheader()
        w.writerows(metas)
    with open(out / "ground_truth" / "mly_detections.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in dets)
    steps = [math.hypot((float(b["lon"]) - float(a["lon"])) * kx, (float(b["lat"]) - float(a["lat"])) * ky)
             for a, b in zip(metas, metas[1:])]
    manifesto = [{"ordem": n, "image_id": m["id"], "lat": m["lat"], "lon": m["lon"],
                  "rumo": m["compass_angle"], "captured_at": m["captured_at"], "is_pano": m["is_pano"],
                  "ficheiro": f"images/{m['file']}", "atribuicao": m["atribuicao"]}
                 for n, m in enumerate(metas, 1)]
    with open(out / "meta" / "manifesto.json", "w", encoding="utf-8") as f:
        json.dump(manifesto, f, ensure_ascii=False, indent=1)
    for name in ("osm_aveiro.geojson", "mapillary_features_aveiro.geojson"):
        src = ROOT / "data" / "ground_truth" / name
        if src.exists():
            shutil.copyfile(src, out / "ground_truth" / name)

    mb = sum((out / "images" / m["file"]).stat().st_size for m in metas) / 1e6
    print(f"{len(metas)} fotos ({mb:.0f} MB) · percurso {sum(steps):.0f} m · "
          f"espaço mediano {sorted(steps)[len(steps) // 2]:.1f} m · {len(dets)} marcações Mapillary"
          + (f" · erros: {errors}" if errors else ""))
    print(f"-> {out}")


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception as e:  # noqa: BLE001
        print(f"  erro {a[0]}: {e}")
        return None


if __name__ == "__main__":
    main()
