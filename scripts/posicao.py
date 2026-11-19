"""
POSIÇÃO de cada objeto no mapa (passos da pipeline):

  contorno      contorno SAM 2 (iluminação/contentores): pé real e pontos 3D só do objeto
  calcular      posição com 1 foto: direção + distância (chão, tamanho, altura, pontos 3D)
  triangulacao  posição com várias fotos seguidas (as direções cruzam-se no objeto)

Uso:
    python scripts/posicao.py contorno
    python scripts/posicao.py calcular
    python scripts/posicao.py triangulacao
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import statistics as st
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import yaml
from PIL import Image
from shapely.geometry import Point
from shapely.strtree import STRtree

sys.path.insert(0, os.path.dirname(__file__))
from catalogo import cluster  # noqa: E402
from comum import DATA_DIR, limiares, limit_vram, perspective_view  # noqa: E402


####################################################################################################
# CONTORNO  (antes: scripts/contorno.py)
####################################################################################################
# Contorno de cada objeto com o SAM 2 (Segment Anything 2.1, Meta, Apache 2.0): passo entre
# sinais.py limites e posicao.py calcular. A caixa do OWLv2 é dada ao SAM 2 como pista; do contorno saem:
#
#   base_y   píxel mais baixo do objeto (o pé real do poste/contentor) -> distância pelo chão
#   d3_mask  distância pelos pontos 3D da Mapillary (sfm/<id>.npz, dados.py poses) que caem DENTRO
#            do contorno (percentil 5) -> a caixa apanhava pontos do fundo (fachadas)
#
# O posicao.py calcular usa estes valores quando existem (senão, a caixa como antes).
# Só iluminação e contentores (nos sinais não ganha: a distância vem do tamanho do sinal).
# Teste na rua seguida (100 fotos): iluminação até 5 m 54% -> 62%, erro típico 4,1 -> 1,7 m.
#
# 360°: o SAM 2 corre numa vista plana centrada no objeto (sem a deformação da panorâmica).
# Os resultados ficam guardados em preds/contorno_cache.jsonl (corridas seguintes só fazem os novos).
#
# Uso:
#     python scripts/posicao.py contorno
#     python scripts/posicao.py contorno --categorias iluminacao


SAM_MODELOS = {t: f"facebook/sam2.1-hiera-{t}" for t in ("large", "base-plus", "small", "tiny")}   # Apache 2.0
MODEL = SAM_MODELOS["large"]
SIZE = 1024
Image.MAX_IMAGE_PIXELS = None


def key(d):
    return f"{d['image_id']}|" + ",".join(str(int(round(v))) for v in d["bbox"])


def pano_to_view(u, v, W, H, yaw, f):
    lon = (np.asarray(u) / W - 0.5) * 2 * np.pi - math.radians(yaw)
    lon = (lon + np.pi) % (2 * np.pi) - np.pi
    lat = (0.5 - np.asarray(v) / H) * np.pi
    ok = np.abs(lon) < np.radians(85)
    return np.tan(lon) * f + SIZE / 2, -np.tan(lat) / np.cos(lon) * f + SIZE / 2, ok


def view_to_pano_v(yv, xv, H, f):
    x, y = (xv - SIZE / 2) / f, (yv - SIZE / 2) / f
    return (0.5 + np.arctan2(y, np.sqrt(x * x + 1)) / np.pi) * H


class Sam:
    def __init__(self, dev, model=MODEL):
        from transformers import Sam2Model, Sam2Processor
        limit_vram()
        self.proc = Sam2Processor.from_pretrained(model)
        self.model = Sam2Model.from_pretrained(model).to(dev).eval()
        self.dev = dev

    @torch.inference_mode()
    def __call__(self, img, box):
        inp = self.proc(images=Image.fromarray(img), input_boxes=[[list(map(float, box))]],
                        return_tensors="pt").to(self.dev)
        out = self.model(**inp, multimask_output=False)
        masks = self.proc.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"])[0]
        return np.asarray(masks).reshape(-1, *masks.shape[-2:])[0] > 0


class Mask2FormerVistas:
    """Segmentação semântica treinada no Mapillary Vistas (Meta). SÓ PARA COMPARAÇÃO: o Vistas
    público é não comercial e os pesos têm licença "other" (ver docs/LICENCAS.md).
    Contorno de um objeto = pedaço das classes certas que mais se sobrepõe à caixa do OWLv2:
      iluminação  Street Light + Pole + Utility Pole      contentores  Trash Can"""
    MODEL = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
    CLASSES = {"iluminacao": ("Street Light", "Pole", "Utility Pole"), "residuos": ("Trash Can",)}

    def __init__(self, dev):
        from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor
        limit_vram()
        self.proc = AutoImageProcessor.from_pretrained(self.MODEL)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(self.MODEL).to(dev).eval()
        if dev == "cuda":
            self.model = self.model.half()
        self.dev = dev
        lab = {v: int(k) for k, v in self.model.config.id2label.items()}
        self.ids = {c: [lab[n] for n in ns] for c, ns in self.CLASSES.items()}
        self.categoria = "iluminacao"

    @torch.inference_mode()
    def __call__(self, img, box):
        import cv2
        H, W = img.shape[:2]
        inp = self.proc(images=Image.fromarray(img), return_tensors="pt").to(self.dev)
        if self.dev == "cuda":
            inp["pixel_values"] = inp["pixel_values"].half()
        out = self.model(**inp)
        sem = self.proc.post_process_semantic_segmentation(out, target_sizes=[(H, W)])[0].cpu().numpy()
        m = np.isin(sem, self.ids[self.categoria]).astype(np.uint8)
        n, lab = cv2.connectedComponents(m)
        if n <= 1:
            return np.zeros((H, W), bool)
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        inside = lab[max(0, y0):max(0, y1) + 1, max(0, x0):max(0, x1) + 1]
        cnt = np.bincount(inside.ravel(), minlength=n)
        cnt[0] = 0
        if cnt.max() == 0:
            return np.zeros((H, W), bool)
        return lab == int(cnt.argmax())


def d3_in(mask, px, py, ok, hdist):
    ins = ok & (px >= 0) & (px < mask.shape[1]) & (py >= 0) & (py < mask.shape[0])
    ins[ins] = mask[py[ins].astype(int), px[ins].astype(int)]
    return float(np.percentile(hdist[ins], 5)) if ins.sum() >= 5 else None


def view(det, meta, im):
    """Imagem onde o SAM 2 corre + caixa nessa imagem + forma de levar pontos da foto para lá.
    360°: vista plana centrada no objeto; frontal: recorte à volta da caixa."""
    x0, y0, x1, y1 = det["bbox"]
    H, W = im.shape[:2]
    if meta["is_pano"] == "True":
        yaw = ((x0 + x1) / 2 / W - 0.5) * 360
        half_w = (x1 - x0) / W * 180
        lat_max = max(abs((0.5 - y0 / H) * 180), abs((0.5 - y1 / H) * 180))
        fov = float(np.clip(max(half_w * 2 * 1.6, lat_max * 2 * 1.25, 30), 30, 120))
        f = (SIZE / 2) / math.tan(math.radians(fov) / 2)
        n = 16
        bu = np.concatenate([np.linspace(x0, x1, n), np.full(n, x1), np.linspace(x0, x1, n), np.full(n, x0)])
        bv = np.concatenate([np.full(n, y0), np.linspace(y0, y1, n), np.full(n, y1), np.linspace(y0, y1, n)])
        vx, vy, _ = pano_to_view(bu, bv, W, H, yaw, f)
        box = [max(0, vx.min()), max(0, vy.min()), min(SIZE - 1, vx.max()), min(SIZE - 1, vy.max())]
        uc = (x0 + x1) / 2

        def to_photo(xs, ys):   # vista -> foto (u contínuo à volta do centro da caixa)
            x, y = (np.asarray(xs, float) - SIZE / 2) / f, (np.asarray(ys, float) - SIZE / 2) / f
            return uc + np.arctan(x) / (2 * np.pi) * W, (0.5 + np.arctan2(y, np.sqrt(x * x + 1)) / np.pi) * H
        return {"img": perspective_view(im, yaw, fov, SIZE), "box": box,
                "to_view": lambda u, v: pano_to_view(u, v, W, H, yaw, f),
                "to_photo": to_photo,
                "base_y": lambda ys, xs: float(view_to_pano_v(ys + 0.5, xs + 0.5, H, f))}
    mw, mh = (x1 - x0) * 0.5 + 20, (y1 - y0) * 0.5 + 20
    cx0, cy0 = int(max(0, x0 - mw)), int(max(0, y0 - mh))
    cx1, cy1 = int(min(W, x1 + mw)), int(min(H, y1 + mh))
    return {"img": np.ascontiguousarray(im[cy0:cy1, cx0:cx1]),
            "box": [x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0],
            "to_view": lambda u, v: (np.asarray(u) - cx0, np.asarray(v) - cy0, np.ones(len(u), bool)),
            "to_photo": lambda xs, ys: (np.asarray(xs, float) + cx0, np.asarray(ys, float) + cy0),
            "base_y": lambda ys, xs: float(ys + cy0)}


def polygons(mask, v, max_pts=80):
    """Máscara -> contornos simplificados na foto original: [[[u, v], ...], ...] (1 por pedaço)."""
    import cv2
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = float(mask.sum())
    out = []
    for c in sorted(cs, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) < max(30, 0.02 * total):
            continue
        eps = max(1.0, 0.003 * cv2.arcLength(c, True))
        c = cv2.approxPolyDP(c, eps, True)
        while len(c) > max_pts:
            eps *= 1.5
            c = cv2.approxPolyDP(c, eps, True)
        if len(c) < 3:
            continue
        u, vv = v["to_photo"](c[:, 0, 0] + 0.5, c[:, 0, 1] + 0.5)
        out.append([[round(float(a), 1), round(float(b), 1)] for a, b in zip(u, vv)])
    return out


def segment(sam, det, meta, im, pts, keep=False):
    """-> (base_y na foto original, d3_mask) ou (None, None). keep=True junta (vista, máscara)."""
    v = view(det, meta, im)
    mask = sam(v["img"], v["box"])
    if mask.sum() < 20:
        return (None, None, v, mask) if keep else (None, None)
    ys, xs = np.nonzero(mask)
    k = np.argmax(ys)
    base_y = v["base_y"](ys[k], xs[k])
    d3 = None
    if pts is not None:
        px, py, ok = v["to_view"](pts["u"], pts["v"])
        d3 = d3_in(mask, px, py, ok, pts["hdist"])
    return (base_y, d3, v, mask) if keep else (base_y, d3)


def main_contorno(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="owlv2")
    ap.add_argument("--categorias", default="iluminacao,residuos")
    ap.add_argument("--sam", choices=list(SAM_MODELOS) + ["mask2former"], default="small",
                    help="SAM 2 small (por omissão desde 19/11; o catálogo principal já foi feito com o large, "
                         "cache contorno_cache.jsonl) / large / base-plus / tiny, ou mask2former (só comparação)")
    args = ap.parse_args(argv)
    cats = set(args.categorias.split(","))
    path = DATA_DIR / "preds" / f"{args.model}_classified.jsonl"
    cache_path = DATA_DIR / "preds" / ("contorno_cache.jsonl" if args.sam == "large" else f"contorno_cache_{args.sam}.jsonl")
    dets = [json.loads(ln) for ln in open(path, encoding="utf-8")]
    cache = {}
    if cache_path.exists():
        for ln in open(cache_path, encoding="utf-8"):
            r = json.loads(ln)
            cache[r["k"]] = r
    with open(DATA_DIR / "meta" / "sample_images.csv", encoding="utf-8") as f:
        metas = {r["id"]: r for r in csv.DictReader(f)}
    todo = defaultdict(list)
    for d in dets:   # novos; e contentores feitos antes de se guardar o contorno ("poly"), que
        r = cache.get(key(d))   # se desenha nas imagens (na iluminação só contam o pé e os pontos 3D)
        if d["category"] in cats and (r is None or (d["category"] == "residuos" and "poly" not in r)):
            todo[d["image_id"]].append(d)
    n_todo = sum(len(v) for v in todo.values())
    print(f"Contornos: {len(dets)} deteções · já feitos {len(cache)} · a fazer {n_todo} em {len(todo)} fotos", flush=True)

    if n_todo:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        sam = Mask2FormerVistas(dev) if args.sam == "mask2former" else Sam(dev, SAM_MODELOS[args.sam])
        t0, n = time.time(), 0
        with open(cache_path, "a", encoding="utf-8") as fc:
            for k, (iid, ds) in enumerate(todo.items(), 1):
                meta = metas[iid]
                im = np.asarray(Image.open(DATA_DIR / "images" / meta["file"]).convert("RGB"))
                pts = None
                f = DATA_DIR / "sfm" / f"{iid}.npz"
                if f.exists():
                    try:
                        with np.load(f) as z:
                            pts = dict(z)
                    except Exception:  # noqa: BLE001
                        pts = None
                for d in ds:
                    poly = []
                    try:
                        sam.categoria = d["category"]          # (só o Mask2Former usa)
                        base_y, d3, v, mask = segment(sam, d, meta, im, pts, keep=True)
                        if base_y is not None:
                            poly = polygons(mask, v)
                    except Exception as e:  # noqa: BLE001
                        print(f"  aviso {iid}: {str(e)[:80]}", flush=True)
                        base_y, d3 = None, None
                    r = {"k": key(d), "base_y": None if base_y is None else round(base_y, 1),
                         "d3_mask": None if d3 is None else round(d3, 2), "poly": poly}
                    cache[r["k"]] = r
                    fc.write(json.dumps(r) + "\n")
                    n += 1
                if k % 100 == 0:
                    fc.flush()
                    el = time.time() - t0
                    print(f"  {k}/{len(todo)} fotos, {n}/{n_todo} contornos, {el / 60:.0f} min "
                          f"(faltam ~{el / n * (n_todo - n) / 60:.0f} min)", flush=True)

        with open(cache_path, "w", encoding="utf-8") as fc:   # reescreve sem entradas repetidas
            for r in cache.values():
                fc.write(json.dumps(r) + "\n")

    n_base = n_d3 = 0
    with open(path, "w", encoding="utf-8") as fo:
        for d in dets:
            d.pop("base_y", None)
            d.pop("d3_mask", None)
            r = cache.get(key(d)) if d["category"] in cats else None
            if r:
                if r["base_y"] is not None:
                    d["base_y"] = r["base_y"]
                    n_base += 1
                if r["d3_mask"] is not None:
                    d["d3_mask"] = r["d3_mask"]
                    n_d3 += 1
            fo.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"-> {path}: base pelo contorno {n_base} · distância 3D pelo contorno {n_d3}")


####################################################################################################
# CALCULAR  (antes: scripts/geolocate.py)
####################################################################################################
# 3.ª fase: calcula as coordenadas (lat/lon) de cada deteção classificada.
#
#   rumo      = rumo da câmara + ângulo horizontal do objeto na imagem
#   distância = plano do chão (candeeiros, contentores: a base da caixa toca no chão)
#               ou tamanho conhecido (sinais: ~0,65 m de largura)
#   posição   = posição da câmara projetada (rumo, distância)
#   rotação   = se houver meta/poses.csv (dados.py poses), as direções usam a rotação REAL da câmara
#               (inclinação e rolamento; as frontais estão muitas vezes inclinadas 5–10°)
#   altura    = iluminação cuja caixa não chega ao chão (só a luminária, ou a base escondida):
#               distância pela altura conhecida da luminária (--h-poste, --h-candeeiro)
#   3d        = com --sfm: distância medida nos pontos 3D da reconstrução da Mapillary que caem
#               dentro da caixa (sfm/<id>.npz, dados.py poses)
#   fotos normais: direções corrigidas da distorção da lente (camera_parameters da
#               Mapillary); caixas encostadas às bordas laterais (--borda) são ignoradas
#   largura   = distância x largura angular da caixa (metros). Se a taxonomia tiver
#               "largura_min_m" num subtipo e o objeto for mais estreito, passa para o
#               subtipo "senao", ou é descartado se senao = "descartar"
#               (ex.: subterrâneo < 1,2 m é quase sempre uma papeleira -> descartado).
#
# Valida contra os pontos de mapa da Mapillary/OSM (vizinho mais próximo da mesma
# categoria) e verifica a convenção de rumo (erro angular médio deve ser ~0°).
#
# Entrada:  data/preds/<modelo>_classified.jsonl
# Saída:    data/preds/<modelo>_geo.jsonl
#
# Uso:
#     python scripts/posicao.py calcular --model owlv2


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


def ray(px, py, meta, R=None):
    """Píxel -> (rumo em graus, elevação em radianos; negativa = abaixo do horizonte).
    Com R (rotação real da câmara, mundo->câmara) usa a inclinação/rolamento verdadeiros;
    sem R assume a câmara direita (só o rumo da foto)."""
    W, H = float(meta["thumb_w"]), float(meta["thumb_h"])
    if meta["is_pano"] == "True":
        lon, lat = (px / W - 0.5) * 2 * math.pi, (0.5 - py / H) * math.pi
        d = (math.cos(lat) * math.sin(lon), -math.sin(lat), math.cos(lat) * math.cos(lon))
    else:
        xu, yu = undistort(px, py, meta)
        n = math.sqrt(xu * xu + yu * yu + 1)
        d = (xu / n, yu / n, 1 / n)
    if R is None:
        heading = float(meta["compass_angle"])
        return (heading + math.degrees(math.atan2(d[0], d[2]))) % 360, math.asin(-d[1])
    e, nn, u = (R[0][k] * d[0] + R[1][k] * d[1] + R[2][k] * d[2] for k in range(3))   # R^T d
    return math.degrees(math.atan2(e, nn)) % 360, math.asin(max(-1.0, min(1.0, u)))


def is_vertical(b):
    return (b[3] - b[1]) > 1.5 * (b[2] - b[0])


def geometry(det, meta, h_pano, h_persp, R=None, lamp_h=None):
    """Devolve (rumo, distância, método) ou None.
    lamp_h: altura da luminária (m) para a iluminação cuja caixa não chega ao chão."""
    x0, y0, x1, y1 = det["bbox"]
    y1 = det.get("base_y", y1)                          # base do contorno (SAM 2), se houver
    xc = (x0 + x1) / 2
    h = h_pano if meta["is_pano"] == "True" else h_persp
    if det["category"] == "sinalizacao":
        bearing, _ = ray(xc, (y0 + y1) / 2, meta, R)
        ang = angular_width(det, meta)
        return bearing, SIGN_WIDTH_M / ang if ang > 0 else None, "tamanho"
    bearing, elev = ray(xc, y1, meta, R)
    below = -elev                                       # ângulo da base abaixo do horizonte
    head_only = det["category"] == "iluminacao" and not is_vertical(det["bbox"])
    if lamp_h and det["category"] == "iluminacao" and (head_only or below < math.radians(0.5)):
        _, up = ray(xc, y0, meta, R)                    # topo da caixa = luminária
        if up > math.radians(1) and lamp_h > h:
            return bearing, (lamp_h - h) / math.tan(up), "altura"
    if below < math.radians(0.5):
        return bearing, None, "chao"
    return bearing, h / math.tan(below), "chao"


def sfm_distance(det, pts, pct, min_pts):
    """Distância (m, na horizontal) pelos pontos 3D dentro da caixa (faixa central de 60%)."""
    if pts is None:
        return None
    import numpy as np
    x0, y0, x1, y1 = det["bbox"]
    cx, hw = (x0 + x1) / 2, (x1 - x0) * 0.3
    m = (pts["u"] >= cx - hw) & (pts["u"] <= cx + hw) & (pts["v"] >= y0) & (pts["v"] <= y1)
    if m.sum() < min_pts:
        return None
    return float(np.percentile(pts["hdist"][m], pct))


def load_rotations():
    """meta/poses.csv (dados.py poses) -> {id: matriz de rotação 3x3 (mundo->câmara)}."""
    path = DATA_DIR / "meta" / "poses.csv"
    if not path.exists():
        return {}
    from comum import rodrigues
    with open(path, encoding="utf-8") as f:
        return {r["id"]: rodrigues(json.loads(r["rotation"])).tolist() for r in csv.DictReader(f) if r["rotation"]}


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


def reference_index(fontes=("mapillary", "osm")):
    """Pontos de referência por categoria (Mapillary + OSM), sem postes de eletricidade."""
    pts = defaultdict(list)
    files = {"mapillary": "mapillary_features_aveiro.geojson", "osm": "osm_aveiro.geojson"}
    for name in (files[f] for f in fontes):
        with open(DATA_DIR / "ground_truth" / name, encoding="utf-8") as f:
            for feat in json.load(f)["features"]:
                p = feat["properties"]
                if p["subtype"] in ("poste", "poste_eletrico"):  # postes sem luminária não contam
                    continue
                pts[p["category"]].append((*feat["geometry"]["coordinates"], p))
    trees = {c: (STRtree([Point(*xy(lon, lat)) for lon, lat, _ in v]), v) for c, v in pts.items()}
    return trees


def main_calcular(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="owlv2")
    ap.add_argument("--h-pano", type=float, default=2.0,
                    help="altura da câmara 360° (m) — 2.0 calibrado com a rotação real (18/11)")
    ap.add_argument("--h-persp", type=float, default=1.95,
                    help="altura da câmara normal/frontal (m) — 1.95 sugerido pela calibração (17/11)")
    ap.add_argument("--max-dist", type=float, default=80,
                    help="mais longe do que isto não entra (tudo o que é visto até aqui entra)")
    ap.add_argument("--perto", type=float, default=35,
                    help="acima disto o objeto fica marcado longe=true (posição menos certa)")
    ap.add_argument("--borda", type=float, default=0.03,
                    help="fotos normais: ignora caixas a menos desta fração das bordas laterais (0 = não)")
    ap.add_argument("--match-m", type=float, default=20)
    ap.add_argument("--ref", choices=["ambos", "osm", "mapillary"], default="ambos",
                    help="pontos de referência da validação (osm = independente da reconstrução 3D)")
    ap.add_argument("--sem-rotacao", action="store_true", help="ignora meta/poses.csv (câmara direita)")
    ap.add_argument("--h-poste", type=float, default=8.7, help="altura da luminária de um poste de luz (m)")
    ap.add_argument("--h-candeeiro", type=float, default=4.4, help="altura de um candeeiro (m)")
    ap.add_argument("--sem-altura", action="store_true", help="não usa a altura da luminária")
    ap.add_argument("--sfm", default="sinalizacao,iluminacao,residuos",
                    help="categorias que usam a distância 3D quando há pontos ('' = nenhuma)")
    ap.add_argument("--sfm-pct", type=float, default=5, help="percentil da distância dos pontos 3D na caixa")
    ap.add_argument("--sfm-min", type=int, default=5, help="mínimo de pontos 3D na caixa")
    args = ap.parse_args(argv)
    rots = {} if args.sem_rotacao else load_rotations()
    sfm_cats = set(filter(None, args.sfm.split(",")))
    lamp = {"poste_iluminacao": args.h_poste, "candeeiro": args.h_candeeiro}
    sfm_cache = {}

    def sfm_pts(iid):
        if iid not in sfm_cache:
            import numpy as np
            if len(sfm_cache) > 64:
                sfm_cache.clear()
            f = DATA_DIR / "sfm" / f"{iid}.npz"
            try:
                with np.load(f) as z:
                    sfm_cache[iid] = dict(z)
            except Exception:  # noqa: BLE001  (sem ficheiro ou estragado)
                sfm_cache[iid] = None
        return sfm_cache[iid]
    print(f"Rotação real da câmara: {len(rots)} fotos · distância 3D: {', '.join(sorted(sfm_cats)) or 'não'}")

    with open(DATA_DIR / "meta" / "sample_images.csv", encoding="utf-8") as f:
        imgs = {r["id"]: r for r in csv.DictReader(f)}
    trees = reference_index(("osm",) if args.ref == "osm" else ("mapillary",) if args.ref == "mapillary"
                            else ("mapillary", "osm"))
    rules = size_rules()
    n_resized = defaultdict(int)
    n_border = 0

    out_path = DATA_DIR / "preds" / f"{args.model}_geo.jsonl"
    n_in = n_out = 0
    errs = defaultdict(list)       # categoria -> erros (m) até à referência mais próxima
    bearing_err = defaultdict(list)
    ratios = defaultdict(list)     # (tipo câmara) -> distância real / estimada (método chão)
    unmatched = defaultdict(int)
    by_method = defaultdict(list)  # (categoria, método, tipo de foto) -> erros

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
            g = geometry(det, meta, args.h_pano, args.h_persp, rots.get(det["image_id"]),
                         None if args.sem_altura else lamp.get(det["subtype"]))
            if (not g or g[1] is None or not (1 <= g[1] <= args.perto)) and det["image_id"] in rots:
                # com a rotação a base ficou acima do horizonte ou muito longe (inclinação estimada
                # pode estar errada em passagens antigas): tenta com a câmara direita
                g0 = geometry(det, meta, args.h_pano, args.h_persp, None,
                              None if args.sem_altura else lamp.get(det["subtype"]))
                if g0 and g0[1] is not None and 1 <= g0[1] <= args.perto or not g or g[1] is None:
                    g = g0
            if g and det["category"] in sfm_cats:
                d3 = det.get("d3_mask") or sfm_distance(det, sfm_pts(det["image_id"]), args.sfm_pct, args.sfm_min)
                if d3:
                    g = (g[0], d3, "3d")
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
                        "pano": meta["is_pano"] == "True", "longe": dist > args.perto})
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
            true_d = math.hypot(rx - cx, ry - cy)
            if method == "chao":
                ratios["360" if det["pano"] else "normal"].append(true_d / dist)
            elif method == "altura":
                ratios[f"altura {det['subtype']}"].append(true_d / dist)
            by_method[(cat, method, "360" if det["pano"] else "frontal")].append(errs[cat][-1])

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
    print("\nPor método de distância e tipo de foto (n, erro mediano, % até 5 m):")
    for key, e in sorted(by_method.items()):
        print(f"  {key[0]:<12} {key[1]:<8} {key[2]:<8} n={len(e):>6,}  {statistics.median(e):5.1f} m  "
              f"{sum(x < 5 for x in e) / len(e):4.0%}")
    for k, r in ratios.items():
        if len(r) > 20:
            m = statistics.median(r)
            if k.startswith("altura"):
                hl = lamp[k.split()[1]]
                print(f"Calibração ({k}): real/estimada = {m:.2f} -> altura sugerida ≈ "
                      f"{args.h_pano + (hl - args.h_pano) * m:.1f} m (atual {hl} m)")
                continue
            h = args.h_pano if k == "360" else args.h_persp
            print(f"\nCalibração ({k}): distância real/estimada mediana = {m:.2f} "
                  f"-> altura de câmara sugerida ≈ {h * m:.2f} m (atual {h} m)")
    print("\nNota: o vizinho mais próximo favorece erros baixos onde há muitos pontos "
          "(ex.: candeeiros); a referência também tem erro próprio (alguns metros).")


####################################################################################################
# TRIANGULACAO  (antes: scripts/triangulacao.py)
####################################################################################################
# Triangulação multivista: posição de cada objeto a partir de VÁRIAS fotos seguidas.
#
# Numa só foto sabemos bem a DIREÇÃO do objeto (erro ~1°), mas a DISTÂNCIA é um
# palpite (altura da câmara ou tamanho do sinal). Com várias fotos tiradas de sítios
# diferentes, as linhas de direção cruzam-se no objeto: a distância passa a ser medida
# pela geometria, sem palpite.
#
#   1. Juntar as deteções do mesmo objeto (mesma passagem, mesmo tipo, posições de
#      uma só foto a <= --raio m). Fica 1 linha por foto (a de maior confiança).
#   2. Ponto que fica mais perto de todas as linhas (mínimos quadrados, no chão, 2D).
#      Linhas que passam a mais de --residuo m do ponto saem e formam outro objeto.
#   3. Só conta como "triangulado" se as fotos estiverem afastadas (>= --base m) e as
#      linhas se cruzarem com ângulo suficiente (>= --angulo graus). Senão fica a
#      posição de uma só foto (mediana).
#   4. Medições (sem referência externa):
#        - erro da distância de uma só foto = distância (1 foto) - distância triangulada;
#        - repetibilidade: o mesmo objeto triangulado em passagens de dias diferentes.
#      E, como antes, distância ao ponto Mapillary/OSM mais próximo (só para comparar).
#
# A posição de cada câmara é a "computed_geometry" da Mapillary (GPS corrigido pela
# reconstrução 3D da própria Mapillary); a escala em metros vem do GPS.
#
# Entrada:  <DATA_DIR>/preds/<modelo>_geo.jsonl, <DATA_DIR>/meta/sample_images.csv
# Saída:    <DATA_DIR>/preds/<modelo>_tri.jsonl, <DATA_DIR>/relatorios/triangulacao.md
#
# Uso (PowerShell, no piloto da rua):
#     $env:DATA_DIR = "data\\piloto_rua"
#     python scripts/posicao.py triangulacao


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


def main_triangulacao(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="owlv2")
    ap.add_argument("--min-conf", type=float, default=None, help="omissão: confianca_minima da taxonomia")
    ap.add_argument("--raio", type=float, default=8, help="junta posições de 1 foto a <= raio (m)")
    ap.add_argument("--residuo", type=float, default=1.5, help="linha a mais disto do ponto sai (m)")
    ap.add_argument("--base", type=float, default=3, help="distância mínima entre fotos (m)")
    ap.add_argument("--angulo", type=float, default=8, help="ângulo mínimo entre linhas (graus)")
    args = ap.parse_args(argv)
    thr = limiares("", args.min_conf)

    with open(DATA_DIR / "meta" / "sample_images.csv", encoding="utf-8") as f:
        seq = {r["id"]: r["sequence"] for r in csv.DictReader(f)}
    groups = defaultdict(list)
    with open(DATA_DIR / "preds" / f"{args.model}_geo.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["confidence"] < thr[d["category"]]:
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


####################################################################################################
SECOES = {
    "contorno": (main_contorno, "contorno SAM 2"),
    "calcular": (main_calcular, "posição com 1 foto"),
    "triangulacao": (main_triangulacao, "posição com várias fotos"),
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
