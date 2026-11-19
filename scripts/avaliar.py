"""
MEDIR o sistema contra referências (Mapillary/OSM só servem para medir):

  erros      relatório completo: encontra, tipo, limites, posição, repetidos
             -> relatorios/analise_erros.md
  confianca  a partir de que confiança se aceita um objeto (por foto e por objeto)
             -> relatorios/confianca.md + .svg + folhas de exemplos
  prompts    escolha das frases dos modelos (vocabulario | ccas | detetor | auditar)

Uso:
    python scripts/avaliar.py erros
    python scripts/avaliar.py confianca --sem-folhas
    python scripts/avaliar.py prompts detetor
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import re
import statistics as st
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from shapely.geometry import Point

sys.path.insert(0, os.path.dirname(__file__))
from comum import DATA_DIR, ROOT, font, load_jsonl, load_taxonomy, mapillary_to_subtype, matches, nms_by_category  # noqa: E402
from posicao import MIN_CONTENTOR_M, box_width_m, project, reference_index, xy  # noqa: E402


####################################################################################################
# ERROS  (antes: scripts/analisa_erros.py)
####################################################################################################
# Análise de erros do sistema, num relatório simples (ecrã + data/relatorios/analise_erros.md).
#
#   1. Encontra os objetos?          (deteção, contra as marcações da Mapillary)
#   2. Acerta no tipo de sinal?      (contra as classes da Mapillary)
#   3. Lê o número do limite?        (SigLIP; contra as fotos e contra os pontos do mapa)
#   4. Acerta no sítio do mapa?      (contra pontos Mapillary + OpenStreetMap)
#   5. Há registos repetidos?        (catálogo)
#
# As referências (Mapillary/OSM) também têm erros e não estão completas: os números servem
# sobretudo para comparar versões do sistema.
#
# Uso:
#     python scripts/avaliar.py erros --model owlv2


CATS = ["sinalizacao", "iluminacao", "residuos"]
NOMES = {"sinalizacao": "Sinais", "iluminacao": "Iluminação", "residuos": "Contentores"}


# ------------------------------------------------------------------ formato
def n_(x):
    return f"{x:,}".replace(",", " ")


def m_(x):
    return f"{x:.1f} m".replace(".", ",")


def p_(a, b):
    return f"{a / b:.0%}" if b else "—"


class Report:
    def __init__(self):
        self.lines = []

    def h(self, t):
        self.lines += ["", f"## {t}", ""]
        print(f"\n=== {t}")

    def p(self, t):
        self.lines += [t, ""]
        print(t)

    def code(self, t):
        self.lines += ["```text", t, "```", ""]
        print(t)

    def table(self, head, rows, legenda=None):
        self.lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        self.lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
        self.lines.append("")
        w = [max(len(str(x)) for x in col) for col in zip(head, *rows)]
        for r in [head] + rows:
            print("  ".join(str(c).ljust(k) for c, k in zip(r, w)))
        if legenda:
            self.lines += [f"*{legenda}*", ""]
            print(f"({legenda})")


# ------------------------------------------------------------------ 1. deteção
def detection(rep, imgs, refs, ours, min_px):
    hit, tot = Counter(), Counter()
    for iid, rs in refs.items():
        if iid not in imgs:
            continue
        area = float(imgs[iid]["thumb_w"]) * float(imgs[iid]["thumb_h"])
        ps = ours.get(iid, [])
        for r in rs:
            b = r["bbox"]
            if r["category"] not in CATS or max(b[2] - b[0], b[3] - b[1]) < min_px:
                continue
            if r["category"] == "residuos" and (box_width_m(b, imgs[iid]) or 0) < MIN_CONTENTOR_M:
                continue  # papeleira (não pedida) ou sem medida
            tot[r["category"]] += 1
            if any(p["category"] == r["category"] and matches({"bbox": p["bbox"]}, r, area) for p in ps):
                hit[r["category"]] += 1
    rep.h("1. Encontra os objetos?")
    rep.table(["Categoria", "Objetos nas fotos", "Encontrados", "Taxa"],
              [[NOMES[c], n_(tot[c]), n_(hit[c]), p_(hit[c], tot[c])] for c in CATS],
              "Objetos nas fotos = marcados pela Mapillary (com pelo menos "
              f"{min_px:.0f} píxeis). Contentores: só os com >= {MIN_CONTENTOR_M} m de largura "
              "(a Mapillary marca também papeleiras, que a ficha não pede).")


# ------------------------------------------------------------------ 2 + 3. tipo e OCR
def signs(rep, imgs, refs, ours, tax, min_px):
    to_sub = mapillary_to_subtype(tax)
    def short(label):  # "Sinal de obrigação" -> "Obrigação"; "Sinal (outro)" -> "Outros"
        t = label.replace(" {n} km/h", "").replace("Sinal ", "")
        t = t[3:] if t.startswith("de ") else t
        return "Outros" if t == "(outro)" else t[0].upper() + t[1:]
    names = {s["id"]: short(s["label"]) for s in tax["categorias"]["sinalizacao"]["subtipos"]}
    ok, n, confu = Counter(), Counter(), Counter()
    ocr = defaultdict(Counter)
    for iid, rs in refs.items():
        if iid not in imgs:
            continue
        area = float(imgs[iid]["thumb_w"]) * float(imgs[iid]["thumb_h"])
        ps = [p for p in ours.get(iid, []) if p["category"] == "sinalizacao"]
        for r in rs:
            b = r["bbox"]
            if r["category"] != "sinalizacao" or max(b[2] - b[0], b[3] - b[1]) < min_px:
                continue
            ref = to_sub(r["value"])
            cand = [p for p in ps if matches({"bbox": p["bbox"]}, r, area)]
            if not ref or not cand:
                continue
            best = max(cand, key=lambda p: p["confidence"])
            n[ref] += 1
            ok[ref] += best["subtype"] == ref
            if best["subtype"] != ref:
                confu[(ref, best["subtype"])] += 1
            m = re.search(r"limit-(\d+)", r["value"])
            if ref == "limite_velocidade" and best["subtype"] == ref and m:
                w = best["bbox"][2] - best["bbox"][0]
                band = band_of(w)
                v = best.get("valor")
                ocr[band]["certo" if v == int(m.group(1)) else ("nada" if v is None else "errado")] += 1

    rep.h("2. Acerta no tipo de sinal?")
    rows = sorted(([names[s], n_(n[s]), p_(ok[s], n[s])] for s in n), key=lambda r: -int(r[1].replace(" ", "")))
    rows.append(["**Total**", f"**{n_(sum(n.values()))}**", f"**{p_(sum(ok.values()), sum(n.values()))}**"])
    rep.table(["Tipo de sinal", "Sinais", "Acerto"], rows,
              "Sinais = sinais encontrados pelo sistema que a Mapillary também identificou. "
              "Acerto = o sistema deu o mesmo tipo.")
    if confu:
        rep.p("Trocas mais frequentes:")
        rep.table(["Era", "O sistema disse", "Vezes"],
                  [[names[a], names[b], k] for (a, b), k in confu.most_common(5)])

    rep.h("3. Lê o número do limite de velocidade?")
    rep.p("O número é lido pelo SigLIP num \"print\" do sinal (vista normal, ampliada 4×), "
          "no passo sinais.py limites. Duas verificações:")
    rep.p("**3.1 Contra as marcações da Mapillary nas fotos**")
    speed_table(rep, ocr)
    return ocr


BANDS = ["Pequeno (< 40 px)", "Médio (40–70 px)", "Grande (≥ 70 px)"]


def band_of(w):
    return BANDS[0] if w < 40 else (BANDS[1] if w < 70 else BANDS[2])


def speed_table(rep, res):
    rows = []
    for b in BANDS + ["Total"]:
        c = sum((res[x] for x in BANDS), Counter()) if b == "Total" else res[b]
        t = sum(c.values())
        rows.append([b if b != "Total" else "**Total**", t, p_(c["certo"], t), p_(c["errado"], t), p_(c["nada"], t)])
    rep.table(["Tamanho do sinal na foto", "Sinais", "Número certo", "Número errado", "Não leu"], rows,
              "Tamanho = largura do sinal em píxeis. Número certo = igual ao da referência.")


def speed_map_check(rep, geo, radius=12):
    """3.2: compara o número lido com o ponto de sinal de limite da Mapillary mais próximo
    (até 12 m e sem outro número diferente por perto)."""
    pts = []
    with open(DATA_DIR / "ground_truth" / "mapillary_features_aveiro.geojson", encoding="utf-8") as f:
        for ft in json.load(f)["features"]:
            m = re.search(r"maximum-speed-limit-(\d+)", ft["properties"].get("value") or "")
            if m:
                pts.append((*xy(*ft["geometry"]["coordinates"]), int(m.group(1))))
    res, seen = defaultdict(Counter), Counter()
    for d in geo:
        if d["subtype"] != "limite_velocidade":
            continue
        px, py = xy(d["lon"], d["lat"])
        near = sorted((math.hypot(x - px, y - py), v) for x, y, v in pts)
        near = [n for n in near if n[0] <= radius]
        if not near or len({v for _, v in near}) > 1:
            continue
        truth, v = near[0][1], d.get("valor")
        seen[truth] += 1
        res[band_of(d["bbox"][2] - d["bbox"][0])]["certo" if v == truth else ("nada" if v is None else "errado")] += 1
    rep.p(f"**3.2 Contra os pontos de sinais da Mapillary no mapa** (até {radius} m)")
    speed_table(rep, res)
    rep.p("Números de referência nesta amostra: " +
          ", ".join(f"{v} km/h: {n}" for v, n in sorted(seen.items())) +
          ". A maioria das ruas da cidade é de 30 km/h, por isso há poucos exemplos de outros números.")


# ------------------------------------------------------------------ 4. posição
def position(rep, geo, match_m, seed):
    trees = reference_index()
    rnd = random.Random(seed)

    def nearest(cat, lon, lat):
        if cat not in trees:
            return None
        tree, ref = trees[cat]
        px, py = xy(lon, lat)
        cand = tree.query(Point(px, py).buffer(match_m))
        if not len(cand):
            return None
        j = min(cand, key=lambda i: math.hypot(xy(*ref[i][:2])[0] - px, xy(*ref[i][:2])[1] - py))
        rx, ry = xy(*ref[j][:2])
        return rx, ry, math.hypot(rx - px, ry - py)

    rows, sem_ref = [], Counter()
    for d in geo:
        r = nearest(d["category"], d["lon"], d["lat"])
        lon_r, lat_r = project(d["cam_lon"], d["cam_lat"], rnd.uniform(0, 360), d["dist_m"])
        rr = nearest(d["category"], lon_r, lat_r)
        if r is None:
            sem_ref[d["category"]] += 1
            continue
        rx, ry, err = r
        cx, cy = xy(d["cam_lon"], d["cam_lat"])
        true_d = math.hypot(rx - cx, ry - cy)
        true_b = math.degrees(math.atan2(rx - cx, ry - cy)) % 360
        db = math.radians((d["bearing"] - true_b + 180) % 360 - 180)
        rows.append({"cat": d["category"], "pano": d["pano"], "dist": d["dist_m"], "err": err,
                     "err_dist": abs(d["dist_m"] - true_d), "err_lat": abs(true_d * math.sin(db)),
                     "acaso": rr[2] if rr else match_m})
    by = {c: [x for x in rows if x["cat"] == c] for c in CATS}

    def rate(sel, t):
        return p_(sum(x["err"] <= t for x in sel), len(sel))

    rep.h("4. Acerta no sítio do mapa?")
    rep.p("**Como se mede** (para cada objeto):")
    rep.code(
        "  P = ponto calculado pelo sistema\n"
        "  R = ponto de referência mais próximo (Mapillary ou OpenStreetMap)\n"
        "  erro = distância entre P e R\n"
        "\n"
        "        . - - - - .\n"
        "      /     R       \\      círculo de 5 m à volta de R\n"
        "     |    P1          |    P1 dentro  ->  ACERTA (erro até 5 m)\n"
        "      \\             /\n"
        "        ' - - - - '      P2 fora    ->  FALHA  (erro maior que 5 m)\n"
        "                     P2")

    rep.p("### 4.1 Resumo")
    rep.table(["Categoria", "Objetos medidos", "Acertam até 5 m", "Acertam até 10 m", "Erro típico"],
              [[NOMES[c], n_(len(by[c])), rate(by[c], 5), rate(by[c], 10),
                m_(st.median(x["err"] for x in by[c]))] for c in CATS if by[c]],
              "Acertam até 5 m = % de objetos a 5 metros ou menos do sítio certo (os restantes falham). "
              "Erro típico = metade dos objetos erra menos do que isto.")

    rep.p("### 4.2 Taxa de acerto conforme a margem aceite")
    rep.table(["Margem aceite"] + [NOMES[c] for c in CATS],
              [[f"até {t} m"] + [rate(by[c], t) for c in CATS] for t in (2, 5, 10, 20)],
              "Cada valor = % de objetos que ficam dentro dessa margem. A taxa de falha é o resto até 100%.")

    rep.p("### 4.3 Perto ou longe da câmara")
    rep.table(["Categoria", "Objeto perto (até 10 m)", "Objeto longe (mais de 10 m)"],
              [[NOMES[c], rate([x for x in by[c] if x["dist"] <= 10], 5),
                rate([x for x in by[c] if x["dist"] > 10], 5)] for c in CATS if by[c]],
              "Valores = % que acertam até 5 m. Objetos perto da câmara ficam mais bem posicionados.")

    rep.p("### 4.4 Foto 360° ou foto frontal")
    rep.table(["Categoria", "Foto 360°", "Foto frontal"],
              [[NOMES[c], rate([x for x in by[c] if x["pano"]], 5),
                rate([x for x in by[c] if not x["pano"]], 5)] for c in CATS if by[c]],
              "Valores = % que acertam até 5 m. As fotos 360° (recentes) posicionam melhor.")

    rep.p("### 4.5 O resultado é mérito do sistema?")
    rep.table(["Categoria", "Erro típico do sistema", "Erro típico ao acaso", "Melhor que o acaso?"],
              [[NOMES[c], m_(st.median(x["err"] for x in by[c])), m_(st.median(x["acaso"] for x in by[c])),
                (lambda k: "sim" if k >= 2 else ("só um pouco" if k >= 1.3 else "não"))(
                    st.median(x["acaso"] for x in by[c]) / st.median(x["err"] for x in by[c]))]
               for c in CATS if by[c]],
              "Ao acaso = o mesmo objeto à mesma distância da câmara, mas numa direção sorteada. "
              "Se o sistema não for bem melhor do que isto, o número é sorte.")

    rep.p("### 4.6 O que falha: a distância ou a direção?")
    rep.table(["Categoria", "Erro na distância à câmara", "Erro na direção"],
              [[NOMES[c], m_(st.median(x["err_dist"] for x in by[c])),
                m_(st.median(x["err_lat"] for x in by[c]))] for c in CATS if by[c]],
              "Valores típicos. A direção está bem medida; o que falha é a distância. "
              "A triangulação (usar várias fotos) corrige isto.")

    if sem_ref:
        rep.p("Não medidos, por não haver referência a menos de "
              f"{match_m:.0f} m: " + ", ".join(f"{NOMES[c].lower()} {n_(sem_ref[c])}" for c in CATS if sem_ref[c]) + ".")


# ------------------------------------------------------------------ 5. catálogo
def catalog(rep):
    path = DATA_DIR / "catalogo" / "catalogo.geojson"
    if not path.exists():
        return
    trees = reference_index()
    with open(path, encoding="utf-8") as f:
        feats = json.load(f)["features"]
    rows = []
    for c in CATS:
        if c not in trees:
            continue
        tree, ref = trees[c]
        es = [ft for ft in feats if ft["properties"]["categoria"] == c]
        hits = Counter()
        pts = [(i["lon"], i["lat"]) for ft in es
               for i in (ft["properties"].get("itens") or [dict(zip(("lon", "lat"), ft["geometry"]["coordinates"]))])]
        for lon, lat in pts:   # ilhas: cada contentor conta por si
            px, py = xy(lon, lat)
            cand = tree.query(Point(px, py).buffer(8))
            if len(cand):
                hits[min(cand, key=lambda i: math.hypot(xy(*ref[i][:2])[0] - px, xy(*ref[i][:2])[1] - py))] += 1
        if hits:
            rows.append([NOMES[c], p_(sum(v > 1 for v in hits.values()), len(hits)),
                         f"{sum(hits.values()) / len(hits):.1f}".replace(".", ",")])
    rep.h("5. Há registos repetidos no catálogo?")
    rep.table(["Categoria", "Objetos repetidos", "Registos por objeto"], rows,
              "Objetos repetidos = % de objetos reais com mais de um registo no catálogo. "
              "Registos por objeto: o ideal é 1,0.")


def main_erros(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="owlv2")
    ap.add_argument("--min-px", type=float, default=24)
    ap.add_argument("--match-m", type=float, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    with open(ROOT / "config" / "taxonomia.yaml", encoding="utf-8") as f:
        tax = yaml.safe_load(f)
    with open(DATA_DIR / "meta" / "sample_images.csv", encoding="utf-8") as f:
        imgs = {r["id"]: r for r in csv.DictReader(f)}
    refs = load_jsonl(DATA_DIR / "ground_truth" / "mly_detections.jsonl")
    ours = load_jsonl(DATA_DIR / "preds" / f"{args.model}_classified.jsonl")
    with open(DATA_DIR / "preds" / f"{args.model}_geo.jsonl", encoding="utf-8") as f:
        geo = [json.loads(line) for line in f]

    rep = Report()
    rep.lines = ["# Análise de erros", "",
                 f"{n_(len(imgs))} fotos analisadas · modelo {args.model}.", "",
                 "As referências (Mapillary e OpenStreetMap) servem só para medir. Também têm erros e "
                 "não estão completas: use estes números sobretudo para comparar versões do sistema."]
    detection(rep, imgs, refs, ours, args.min_px)
    signs(rep, imgs, refs, ours, tax, args.min_px)
    speed_map_check(rep, geo)
    position(rep, geo, args.match_m, args.seed)
    catalog(rep)

    out = DATA_DIR / "relatorios"
    out.mkdir(parents=True, exist_ok=True)
    (out / "analise_erros.md").write_text("\n".join(rep.lines), encoding="utf-8")
    print(f"\n-> {out / 'analise_erros.md'}")


####################################################################################################
# CONFIANCA  (antes: scripts/analise_confianca.py)
####################################################################################################
# Análise exploratória da CONFIANÇA: a partir de que confiança se aceita um objeto?
# (em vez de cortar pela distância). Para cada categoria e cada limiar t:
#
#   encontra   % dos objetos marcados pela Mapillary nas fotos que têm uma deteção nossa >= t
#   precisão   % das nossas deteções >= t que batem com uma marcação Mapillary
#              (valor MÍNIMO: a Mapillary não marca tudo; parte dos "erros" são objetos reais
#              que ela não marcou — ver as folhas de exemplos)
#   F1         equilíbrio entre as duas; o limiar com F1 máximo é o sugerido
#
# Só conta fotos que a Mapillary marcou e objetos com >= --min-px píxeis (como avaliar.py erros).
# Deteções sobre marcações que não contam (papeleiras, objetos pequenos) ficam de fora das contas.
#
# Saída (data/relatorios/):
#   confianca.md                tabela por limiar + limiar sugerido por categoria
#   confianca.svg               curvas encontra / precisão por limiar
#   confianca_<categoria>.jpg   folha de exemplos: uma linha por faixa de confiança
#                               (verde = bate com a Mapillary, cinzento = não marcado por ela)
#
# Uso:
#     python scripts/avaliar.py confianca
#     python scripts/avaliar.py confianca --exemplos 14


Image.MAX_IMAGE_PIXELS = None
CATS = ["sinalizacao", "iluminacao", "residuos"]
NOMES = {"sinalizacao": "Sinais", "iluminacao": "Iluminação", "residuos": "Contentores"}
LIMIARES = [round(x, 2) for x in np.arange(0.10, 0.96, 0.05)]
FAIXAS = [(round(a, 1), round(a + 0.1, 1)) for a in np.arange(0.1, 1.0, 0.1)]
AZUL, LARANJA = "#2a78d6", "#eb6834"   # paleta validada (dataviz): precisão, encontra


def size(b):
    return max(b[2] - b[0], b[3] - b[1])


def evaluate(imgs, refs, ours, min_px):
    """-> por categoria: lista de (confiança, estado) das deteções e lista da melhor confiança
    de cada marcação (0 = não encontrada). estado: 'ok' | 'nao' | None (não conta)."""
    preds, best = defaultdict(list), defaultdict(list)
    for iid, rs in refs.items():
        if iid not in imgs:
            continue
        area = float(imgs[iid]["thumb_w"]) * float(imgs[iid]["thumb_h"])
        ps = ours.get(iid, [])
        for c in CATS:
            rc = [r for r in rs if r["category"] == c]
            counted = [r for r in rc if size(r["bbox"]) >= min_px and not (
                c == "residuos" and (box_width_m(r["bbox"], imgs[iid]) or 0) < MIN_CONTENTOR_M)]
            ignored = [r for r in rc if r not in counted]
            pc = [p for p in ps if p["category"] == c]
            for r in counted:
                m = [p["confidence"] for p in pc if matches({"bbox": p["bbox"]}, r, area)]
                best[c].append(max(m, default=0.0))
            for p in pc:
                if size(p["bbox"]) < min_px:
                    continue
                if any(matches({"bbox": p["bbox"]}, r, area) for r in counted):
                    est = "ok"
                elif any(matches({"bbox": p["bbox"]}, r, area) for r in ignored):
                    est = None
                else:
                    est = "nao"
                preds[c].append((p["confidence"], est, p))
    return preds, best


def curves(preds, best):
    out = {}
    for c in CATS:
        conf = np.array([x for x, s, _ in preds[c] if s])
        ok = np.array([s == "ok" for _, s, _ in preds[c] if s])
        b = np.array(best[c])
        rows = []
        for t in LIMIARES:
            sel = conf >= t
            prec = ok[sel].mean() if sel.any() else float("nan")
            rec = (b >= t).mean() if len(b) else float("nan")
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
            rows.append((t, rec, prec, f1, int(sel.sum())))
        out[c] = rows
    return out


def chart(res, chosen, path):
    """Gráfico SVG (sem dependências): 3 painéis, precisão e encontra por limiar."""
    W, H, pw, ph, ml, mt = 1280, 340, 330, 230, 60, 40
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Segoe UI, sans-serif">',
         f'<rect width="{W}" height="{H}" fill="#fcfcfb"/>']
    for k, c in enumerate(CATS):
        ox = ml + k * (pw + 70)
        X = lambda t: ox + (t - 0.1) / 0.85 * pw  # noqa: E731
        Y = lambda v: mt + ph - v * ph            # noqa: E731
        s.append(f'<text x="{ox}" y="{mt - 14}" font-size="16" font-weight="600" fill="#0b0b0b">{NOMES[c]}</text>')
        for v in (0, .25, .5, .75, 1):
            s.append(f'<line x1="{ox}" x2="{ox + pw}" y1="{Y(v)}" y2="{Y(v)}" stroke="#e6e5e0"/>')
            s.append(f'<text x="{ox - 8}" y="{Y(v) + 4}" font-size="11" text-anchor="end" fill="#52514e">{v:.0%}</text>')
        for t in (0.1, 0.3, 0.5, 0.7, 0.9):
            s.append(f'<text x="{X(t)}" y="{mt + ph + 18}" font-size="11" text-anchor="middle" fill="#52514e">{t:.1f}</text>')
        s.append(f'<text x="{ox + pw / 2}" y="{mt + ph + 38}" font-size="12" text-anchor="middle" fill="#52514e">confiança mínima aceite</text>')
        xc = X(chosen[c])
        s.append(f'<line x1="{xc}" x2="{xc}" y1="{mt}" y2="{mt + ph}" stroke="#0b0b0b" stroke-dasharray="4 3"/>')
        s.append(f'<text x="{xc + 4}" y="{mt + 12}" font-size="11" fill="#0b0b0b">sugerido {chosen[c]:.2f}</text>')
        for idx, col, name in ((2, AZUL, "precisão"), (1, LARANJA, "encontra")):
            pts = " ".join(f"{X(r[0]):.1f},{Y(r[idx]):.1f}" for r in res[c])
            s.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"/>')
            for r in res[c]:
                s.append(f'<circle cx="{X(r[0]):.1f}" cy="{Y(r[idx]):.1f}" r="3.5" fill="{col}" stroke="#fcfcfb" stroke-width="1.5">'
                         f'<title>{name} {r[idx]:.0%} com limiar {r[0]:.2f}</title></circle>')
            last = res[c][-1]
            s.append(f'<text x="{X(last[0]) + 6}" y="{Y(last[idx]) + 4}" font-size="11" fill="#0b0b0b">{name}</text>')
    s.append("</svg>")
    path.write_text("\n".join(s), encoding="utf-8")


def sheet(c, preds, imgs, n, path, seed):
    """Uma linha por faixa de confiança, n recortes ao acaso em cada."""
    rng = random.Random(seed)
    cw, ch = (110, 170) if c == "iluminacao" else (130, 130)
    lab_w, gap = 120, 4
    f, fs = font(14), font(12)
    W = lab_w + n * (cw + gap)
    H = len(FAIXAS) * (ch + 20 + gap) + 34
    out = Image.new("RGB", (W, H), (24, 24, 24))
    dr = ImageDraw.Draw(out)
    dr.text((8, 8), f"{NOMES[c]}: exemplos por faixa de confiança · verde = a Mapillary também marcou · "
                    "cinzento = não marcado por ela (pode ser erro OU objeto que ela não marcou)",
            fill=(230, 230, 230), font=f)
    cache = {}
    for k, (a, b) in enumerate(FAIXAS):
        y = 34 + k * (ch + 20 + gap)
        pool = [(conf, est, p) for conf, est, p in preds[c] if a <= conf < b and est]
        dr.text((8, y + ch // 2 - 16), f"{a:.1f} – {b:.1f}", fill=(240, 240, 240), font=f)
        if pool:
            okp = sum(s == "ok" for _, s, _ in pool) / len(pool)
            dr.text((8, y + ch // 2 + 4), f"{len(pool)} · {okp:.0%} verde", fill=(170, 170, 170), font=fs)
        for j, (conf, est, p) in enumerate(rng.sample(pool, min(n, len(pool)))):
            iid = p["image_id"]
            if iid not in cache:
                if len(cache) > 40:
                    cache.clear()
                cache[iid] = Image.open(DATA_DIR / "images" / imgs[iid]["file"]).convert("RGB")
            im = cache[iid]
            x0, y0, x1, y1 = p["bbox"]
            m = 0.25 * max(x1 - x0, y1 - y0) + 6
            crop = im.crop((int(max(0, x0 - m)), int(max(0, y0 - m)), int(min(im.width, x1 + m)),
                            int(min(im.height, y1 + m))))
            crop.thumbnail((cw, ch))
            x = lab_w + j * (cw + gap)
            out.paste(crop, (x + (cw - crop.width) // 2, y + (ch - crop.height) // 2))
            col = (40, 200, 90) if est == "ok" else (140, 140, 140)
            dr.rectangle([x, y, x + cw - 1, y + ch - 1], outline=col, width=3)
            dr.text((x + 3, y + ch + 2), f"{conf:.2f} {p['label'].replace('Sinal ', '')[:16]}",
                    fill=(210, 210, 210), font=fs)
    out.save(path, quality=85)


def _grid(pts, cell):
    from catalogo import xy
    g = defaultdict(list)
    for i, p in enumerate(pts):
        x, y = xy(*p)
        g[(int(x // cell), int(y // cell))].append(i)
    return g


def _near(p, pts, grid, cell, r):
    """Índices de pts a <= r metros de p (r <= cell)."""
    from catalogo import dist_m, xy
    x, y = xy(*p)
    return [i for a in (-1, 0, 1) for b in (-1, 0, 1) for i in grid[(int(x // cell) + a, int(y // cell) + b)]
            if dist_m(p, pts[i]) <= r]


def ref_objects(refs, juntar_m=2.0):
    """Mapillary + OSM marcam muitas vezes o MESMO objeto (dois pontos a 1–2 m): junta-os num só."""
    out = {}
    for c, rs in refs.items():
        grid, used, objs = _grid(rs, 5), set(), []
        for i, r in enumerate(rs):
            if i in used:
                continue
            grupo = [j for j in _near(r, rs, grid, 5, juntar_m) if j not in used] + [i]
            used.update(grupo)
            objs.append(r)
        out[c] = objs
    return out


def objetos(model, raio=8.0):
    """Por OBJETO (catálogo, várias fotos juntas): para cada limiar monta o catálogo em memória e
    compara com os objetos de referência no mapa (Mapillary + OSM, pontos a <= 2 m juntos num só).
      encontra   % dos objetos de referência com um elemento do catálogo a <= raio
      certos     % dos elementos com um objeto de referência a <= raio (valor mínimo: as
                 referências não têm tudo)
      por objeto elementos por objeto encontrado (1,0 = sem repetidos)
    """
    import json

    import yaml
    from catalogo import dist_m, load_refs, montar, points
    from comum import ROOT
    with open(ROOT / "config" / "taxonomia.yaml", encoding="utf-8") as f:
        tax = yaml.safe_load(f)
    with open(DATA_DIR / "meta" / "sample_images.csv", encoding="utf-8") as f:
        imgs = {r["id"]: r for r in csv.DictReader(f)}
    with open(DATA_DIR / "preds" / f"{model}_geo.jsonl", encoding="utf-8") as f:
        geo = [json.loads(ln) for ln in f]
    refs = ref_objects(load_refs())
    grids = {c: _grid(rs, raio) for c, rs in refs.items()}
    res = {c: [] for c in CATS}
    for t in [0.0, 0.1, 0.2] + [round(x, 2) for x in np.arange(0.30, 0.86, 0.05)]:   # 0.0 = sem limiar
        cat, _, _ = montar([d for d in geo if d["confidence"] >= t], tax, imgs)
        for c in CATS:
            rs = refs.get(c, [])
            es = points([e for e in cat if e["categoria"] == c])   # ilhas: cada contentor
            if not rs or not es:
                continue
            eg = _grid(es, raio)
            found = sum(bool(_near(r, es, eg, raio, raio)) for r in rs)
            nearest = Counter()
            n_ok = 0
            for e in es:
                nb = _near(e, rs, grids[c], raio, raio)
                if nb:
                    n_ok += 1
                    nearest[min(nb, key=lambda i: dist_m(e, rs[i]))] += 1
            enc, cer = found / len(rs), n_ok / len(es)
            f1 = 2 * enc * cer / (enc + cer) if enc + cer else 0
            dup = sum(nearest.values()) / len(nearest) if nearest else 0
            res[c].append((t, len(es), enc, cer, dup, f1))
    return res


def main_confianca(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="owlv2")
    ap.add_argument("--min-px", type=float, default=24)
    ap.add_argument("--exemplos", type=int, default=12, help="recortes por faixa nas folhas")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sem-folhas", action="store_true", help="não desenha as folhas de exemplos")
    args = ap.parse_args(argv)

    with open(DATA_DIR / "meta" / "sample_images.csv", encoding="utf-8") as f:
        imgs = {r["id"]: r for r in csv.DictReader(f)}
    refs = load_jsonl(DATA_DIR / "ground_truth" / "mly_detections.jsonl")
    ours = load_jsonl(DATA_DIR / "preds" / f"{args.model}_classified.jsonl")
    preds, best = evaluate(imgs, refs, ours, args.min_px)
    res = curves(preds, best)
    chosen = {c: max(res[c], key=lambda r: r[3])[0] for c in CATS}

    out = DATA_DIR / "relatorios"
    out.mkdir(parents=True, exist_ok=True)
    total = {c: [d["confidence"] for ds in ours.values() for d in ds if d["category"] == c] for c in CATS}
    L = ["# A partir de que confiança se aceita um objeto?", "",
         f"Fotos com marcações da Mapillary: {sum(1 for i in refs if i in imgs):,}. "
         "Encontra = % dos objetos marcados pela Mapillary que o sistema apanha com confiança >= limiar. "
         "Precisão = % das deteções >= limiar que batem com uma marcação (valor mínimo: a Mapillary "
         "não marca tudo). Deteções na amostra = quantas deteções (todas as fotos) passam o limiar.", ""]
    for c in CATS:
        L += [f"## {NOMES[c]}", "",
              f"Marcações Mapillary contadas: {len(best[c]):,} · deteções nas fotos marcadas: "
              f"{sum(1 for _, s, _ in preds[c] if s):,}", "",
              "| Limiar | Encontra | Precisão | F1 | Deteções na amostra |", "|---|---|---|---|---|"]
        tot = np.array(total[c])
        for t, rec, prec, f1, _ in res[c]:
            mark = " **← sugerido**" if t == chosen[c] else ""
            L.append(f"| {t:.2f}{mark} | {rec:.0%} | {prec:.0%} | {f1:.2f} | {int((tot >= t).sum()):,} |")
        L.append("")
    L += ["## Limiar sugerido (F1 máximo)", "", "| Categoria | Limiar | Encontra | Precisão |", "|---|---|---|---|"]
    for c in CATS:
        r = next(r for r in res[c] if r[0] == chosen[c])
        L.append(f"| {NOMES[c]} | {r[0]:.2f} | {r[1]:.0%} | {r[2]:.0%} |")
    if (DATA_DIR / "preds" / f"{args.model}_geo.jsonl").exists():
        ob = objetos(args.model)
        L += ["", "# Por objeto no catálogo (todas as fotos juntas)", "",
              "Objetos de referência = pontos da Mapillary e do OSM no mapa; quando as duas fontes marcam "
              "o mesmo objeto (pontos a <= 2 m) conta uma vez. Encontra = % desses objetos com um elemento "
              "do catálogo a <= 8 m. Certos = % dos elementos com um objeto de referência a <= 8 m (valor "
              "mínimo: as referências não têm tudo). Registos por objeto: 1,0 = sem repetidos.", ""]
        best_ob = {}
        for c in CATS:
            if not ob[c]:
                continue
            best_ob[c] = max(ob[c], key=lambda r: r[5])[0]
            L += [f"## {NOMES[c]}", "", "| Limiar | Elementos | Encontra | Certos | Registos por objeto | F1 |",
                  "|---|---|---|---|---|---|"]
            for t, n, enc, cer, dup, f1 in ob[c]:
                mark = " **← melhor F1**" if t == best_ob[c] else ""
                L.append(f"| {t:.2f}{mark} | {n:,} | {enc:.0%} | {cer:.0%} | {dup:.2f} | {f1:.2f} |")
            L.append("")
        L += ["Melhor limiar por objeto: " + " · ".join(f"{NOMES[c]} {t:.2f}" for c, t in best_ob.items())]
    L += ["", "Gráfico: confianca.svg · exemplos por faixa: confianca_<categoria>.jpg"]
    (out / "confianca.md").write_text("\n".join(L), encoding="utf-8")
    chart(res, chosen, out / "confianca.svg")
    if not args.sem_folhas:
        for c in CATS:
            sheet(c, preds, imgs, args.exemplos, out / f"confianca_{c}.jpg", args.seed)
    print("\n".join(L))
    print(f"\n-> {out / 'confianca.md'} · confianca.svg · confianca_<categoria>.jpg")


####################################################################################################
# PROMPTS  (antes: scripts/escolhe_prompts.py)
####################################################################################################
# Escolher as frases (prompts) que os modelos entendem melhor. Quatro comandos, por ordem:
#
#   vocabulario   "dicionário" dos modelos: que palavras (config/vocabulario_en.txt + frases atuais)
#                 ficam mais perto das fotos de referência de cada classe?  -> relatorios/vocabulario.md
#   ccas          fase 1, sem anotações (método CCAS, arXiv 2505.09139): frases candidatas
#                 (config/prompts_candidatos.yaml) contra as fotos de referência.
#                   SigLIP: perto da SUA subclasse e longe das outras da mesma categoria;
#                   OWLv2:  pontuação alta nas fotos da SUA categoria e baixa nas outras.
#                 -> relatorios/escolha_prompts.json + .md
#   detetor       fase 2, medição: corre o detetor nas fotos de DATA_DIR com os conjuntos de frases
#                 (atual, antigo, ccas) e mede quantos objetos marcados pela Mapillary encontra.
#                 Para não viciar: escolher numa rua e confirmar noutra (outra DATA_DIR).
#                 -> <DATA_DIR>/relatorios/escolha_prompts_detetor.md
#   auditar       verificação a olho do TIPO (a Mapillary não diz o tipo):
#                   preparar  recortes ao acaso, classificados com as frases atuais e as CCAS
#                             -> relatorios/auditoria_classificacao/folha_<cat>.jpg + auditoria.csv
#                   (pessoa)  preencher a coluna "verdade" (id da classe, ou "nao")
#                   medir     acerto de cada conjunto -> auditoria.md
#
# Fotos de referência: data/referencias/classes/<categoria>/<classe>/ (não vêm das marcações
# Mapillary, para não viciar a medição).
#
# Uso (PowerShell):
#     python scripts/avaliar.py prompts vocabulario --top 10
#     python scripts/avaliar.py prompts ccas
#     $env:DATA_DIR = "data\\piloto_sequencia"; python scripts/avaliar.py prompts detetor
#     python scripts/avaliar.py prompts auditar preparar --pastas data/piloto_rua data/piloto_sequencia
#     python scripts/avaliar.py prompts auditar medir


REF = ROOT / "data" / "referencias" / "classes"
OUT = ROOT / "data" / "relatorios"
AUDIT = OUT / "auditoria_classificacao"
SIGLIP = "google/siglip-so400m-patch14-384"
OWL = "google/owlv2-base-patch16-ensemble"
ANTIGO = {"iluminacao": ["a street light", "a lamp post"],
          "residuos": ["a trash bin", "a waste container", "a recycling container"]}
Image.MAX_IMAGE_PIXELS = None


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ fotos de referência e modelos
def load_refs():
    refs = defaultdict(list)  # (categoria, classe) -> imagens
    for p in sorted(REF.glob("*/*/*")):
        if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
            refs[(p.parent.parent.name, p.parent.name)].append(Image.open(p).convert("RGB"))
    return refs


@torch.inference_mode()
def siglip_sims(refs, texts, dev):
    """Semelhança média (SigLIP) entre as fotos de cada classe e cada frase."""
    from transformers import AutoModel, AutoProcessor
    proc = AutoProcessor.from_pretrained(SIGLIP)
    model = AutoModel.from_pretrained(SIGLIP, dtype=torch.float16).to(dev).eval()
    f = lambda x: (x if torch.is_tensor(x) else x.pooler_output).float()  # noqa: E731
    T = []
    for i in range(0, len(texts), 64):
        tok = proc(text=[f"This is a photo of {t}." for t in texts[i:i + 64]],
                   padding="max_length", return_tensors="pt").to(dev)
        T.append(f(model.get_text_features(**tok)))
    T = torch.cat(T)
    T = T / T.norm(dim=-1, keepdim=True)
    sims = {}
    for key, ims in refs.items():
        pv = proc(images=ims, return_tensors="pt")["pixel_values"].to(dev).half()
        E = f(model.get_image_features(pixel_values=pv))
        E = E / E.norm(dim=-1, keepdim=True)
        sims[key] = (E @ T.T).mean(0).cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return sims


@torch.inference_mode()
def owl_scores(refs, texts, dev, k=5):
    """Pontuação (0–1) do detetor para cada frase, nas k zonas mais 'objeto' de cada foto."""
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    proc = Owlv2Processor.from_pretrained(OWL)
    model = Owlv2ForObjectDetection.from_pretrained(OWL, dtype=torch.float16).to(dev).eval()
    out = {}
    for key, ims in refs.items():
        per = []
        for im in ims:
            s = np.zeros(len(texts))
            for i in range(0, len(texts), 64):
                chunk = texts[i:i + 64]
                # o OWLv2 só lê 16 tokens: frases longas ficam cortadas
                x = proc(text=[chunk], images=im, return_tensors="pt", padding="max_length",
                         truncation=True, max_length=16).to(dev)
                o = model(input_ids=x["input_ids"], attention_mask=x["attention_mask"],
                          pixel_values=x["pixel_values"].half())
                top = o.objectness_logits[0].float().topk(k).indices
                s[i:i + len(chunk)] = torch.sigmoid(o.logits[0, top].float()).max(0).values.cpu().numpy()
            per.append(s)
        out[key] = np.mean(per, 0)
    del model
    torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------------ vocabulario
def cmd_vocabulario(args):
    tax = load_taxonomy()
    refs = load_refs()
    terms = [ln.strip() for ln in open(ROOT / "config" / "vocabulario_en.txt", encoding="utf-8")
             if ln.strip() and not ln.startswith("#")]
    current = {p for cat in tax["categorias"].values() for s in cat["subtipos"] for p in s["prompts"]}
    current |= {p for ps in tax["detetor"].values() for p in ps}
    terms += [t for t in sorted(current) if t not in terms]
    print(f"{sum(len(v) for v in refs.values())} imagens de referência em {len(refs)} classes · {len(terms)} expressões")
    dev = device()
    sig, owl = siglip_sims(refs, terms, dev), owl_scores(refs, terms, dev)
    lines = ["# Vocabulário: que palavras os modelos associam a cada classe", "",
             "Imagens de referência: data/referencias/classes. (atual) = frase que usamos hoje.", ""]
    mark = lambda t: f"{t} (atual)" if t in current else t  # noqa: E731
    rank = lambda v: list(np.argsort(-v))  # noqa: E731
    for key in sorted(refs):
        cat, cls = key
        others = [k for k in refs if k[0] == cat and k != key]
        s, o = sig[key], owl[key]
        dist = s - (np.max([sig[k] for k in others], axis=0) if others else 0)
        r_s, r_d, r_o = rank(s), rank(dist), rank(o)
        title = f"{cat} / {cls} ({len(refs[key])} imagens)"
        print(f"\n=== {title}")
        lines += [f"## {title}", "", "| # | SigLIP: mais perto | SigLIP: distingue das outras classes | "
                  "OWLv2 (detetor): encontra melhor |", "|---|---|---|---|"]
        for n in range(min(args.top, len(terms))):
            row = [f"{mark(terms[r_s[n]])} ({s[r_s[n]]:.3f})",
                   f"{mark(terms[r_d[n]])} ({dist[r_d[n]]:+.3f})" if others else "—",
                   f"{mark(terms[r_o[n]])} ({o[r_o[n]]:.0%})"]
            lines.append(f"| {n + 1} | " + " | ".join(row) + " |")
            print(f"{n + 1:>2}. " + " | ".join(row))
        cur_rows = sorted((r_s.index(terms.index(t)) + 1, r_o.index(terms.index(t)) + 1, t) for t in current)
        lines += ["", f"Posição das frases atuais (de {len(terms)}): " +
                  "; ".join(f"\"{t}\" SigLIP {a}.º, OWLv2 {b}.º" for a, b, t in cur_rows[:6]), ""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "vocabulario.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n-> {OUT / 'vocabulario.md'}")


# ------------------------------------------------------------------ ccas (fase 1)
def cmd_ccas(args):
    cand = yaml.safe_load(open(ROOT / "config" / "prompts_candidatos.yaml", encoding="utf-8"))
    refs = load_refs()
    print(f"{sum(len(v) for v in refs.values())} fotos de referência: "
          + ", ".join(f"{c}/{k} {len(v)}" for (c, k), v in sorted(refs.items())))
    lines = ["# Escolha das frases (fase 1: contraste nas fotos de referência)", "",
             "Contraste = parecida com a sua classe menos a mais parecida das outras. "
             "Escolhidas = as melhores entre as que também são bem parecidas com a sua classe.", ""]
    result = {"classificador": {}, "detetor": {}}
    dev = device()

    texts = sorted({t for cat in cand["classificador"].values() for ts in cat.values() for t in ts})
    sims = siglip_sims(refs, texts, dev)
    idx = {t: i for i, t in enumerate(texts)}
    for cat, classes in cand["classificador"].items():
        for cls, ts in classes.items():
            if (cat, cls) not in refs:
                print(f"  (sem fotos de referência para {cat}/{cls})")
                continue
            others = [k for k in refs if k[0] == cat and k[1] != cls and k[1] in classes]
            rows = []
            for t in ts:
                own = sims[(cat, cls)][idx[t]]
                oth = max(sims[k][idx[t]] for k in others) if others else 0.0
                rows.append((own - oth, own, t))
            med = np.median([r[1] for r in rows])
            rows.sort(reverse=True)
            chosen = [t for c, o, t in rows if o >= med][: args.top_cls]
            result["classificador"].setdefault(cat, {})[cls] = chosen
            print(f"\n[SigLIP] {cat}/{cls}: escolhidas {chosen}")
            lines += [f"## Classificador — {cat} / {cls}", "",
                      "| Frase | Parecida (sua classe) | Contraste | Escolhida |", "|---|---|---|---|"]
            lines += [f"| {t} | {o:.3f} | {c:+.3f} | {'✅' if t in chosen else ''} |" for c, o, t in rows]
            lines.append("")

    texts = sorted({t for ts in cand["detetor"].values() for t in ts})
    sc = owl_scores(refs, texts, dev)
    idx = {t: i for i, t in enumerate(texts)}
    for cat, ts in cand["detetor"].items():
        own_k = [k for k in refs if k[0] == cat]
        oth_k = [k for k in refs if k[0] != cat]
        rows = []
        for t in ts:
            own = np.mean([sc[k][idx[t]] for k in own_k])
            oth = np.mean([sc[k][idx[t]] for k in oth_k]) if oth_k else 0.0
            rows.append((own - oth, own, t))
        med = np.median([r[1] for r in rows])
        rows.sort(reverse=True)
        chosen = [t for c, o, t in rows if o >= med][: args.top_det]
        result["detetor"][cat] = chosen
        print(f"\n[OWLv2] {cat}: escolhidas {chosen}")
        lines += [f"## Detetor — {cat}", "", "| Frase | Pontuação (sua categoria) | Contraste | Escolhida |",
                  "|---|---|---|---|"]
        lines += [f"| {t} | {o:.0%} | {c:+.0%} | {'✅' if t in chosen else ''} |" for c, o, t in rows]
        lines.append("")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "escolha_prompts.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "escolha_prompts.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n-> {OUT / 'escolha_prompts.json'}\n-> {OUT / 'escolha_prompts.md'}")


# ------------------------------------------------------------------ detetor (fase 2)
def cmd_detetor(args):
    import detetar as rd
    from classificar import Classifier, square_crop, suppress_contained
    from posicao import MIN_CONTENTOR_M, box_width_m

    dev = device()
    tax = load_taxonomy()
    chosen = json.loads((OUT / "escolha_prompts.json").read_text(encoding="utf-8"))["detetor"]
    signs = tax["detetor"]["sinalizacao"]
    sets = {"atual": {c: tax["detetor"][c] for c in ("iluminacao", "residuos")}, "antigo": ANTIGO, "ccas": chosen}
    with open(DATA_DIR / "meta" / "sample_images.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limite:
        rows = rows[: args.limite]
    imgs = {r["id"]: r for r in rows}
    refs = load_jsonl(DATA_DIR / "ground_truth" / "mly_detections.jsonl")
    print(f"{len(rows)} fotos em {DATA_DIR}")

    det, clf = rd.Owlv2(dev), Classifier(tax, dev)
    views = {}
    with ThreadPoolExecutor(os.cpu_count() or 8) as pool:   # vistas 960x960 (uma vez só)
        for row, vs in pool.map(rd.prepare, rows):
            views[row["id"]] = vs
    full = {}

    def image(iid):
        if iid not in full:
            full[iid] = Image.open(DATA_DIR / "images" / imgs[iid]["file"]).convert("RGB")
        return full[iid]

    lines = [f"# Escolha das frases (fase 2: medição em {DATA_DIR.name}, {len(rows)} fotos)", "",
             "| Conjunto | Frases | Iluminação encontrada | Contentores encontrados | "
             "Deteções/foto (iluminação) | Deteções/foto (contentores) |", "|---|---|---|---|---|---|"]
    for name, ps in sets.items():
        t0 = time.time()
        prompts = {p: ("sinalizacao", "") for p in signs}
        for cat in ("iluminacao", "residuos"):
            prompts.update({p: (cat, "") for p in ps[cat]})
        rd.PROMPTS.clear()
        rd.PROMPTS.update(prompts)
        tok = det.proc(text=[list(prompts)], return_tensors="pt")
        det.ids, det.mask = tok["input_ids"].to(dev), tok["attention_mask"].to(dev)
        dets = defaultdict(list)
        batch = [(iid, info, a) for iid, vs in views.items() for a, info in vs]
        for i in range(0, len(batch), 12):
            items = batch[i:i + 12]
            for (iid, info, _), res in zip(items, det([a for _, _, a in items], 0.1)):
                for box, score, prompt in res:
                    ob = rd.to_original(box, info)
                    if ob and score >= 0.2:
                        dets[iid].append({"bbox": ob, "score": score, "category": prompts[prompt][0], "prompt": prompt})
        # mesmo filtro que classificar.py: NMS, tamanho mínimo, caixas contidas, SigLIP
        kept, queue = defaultdict(list), []
        for iid, ds in dets.items():
            ds = [d for d in suppress_contained(nms_by_category(ds))
                  if max(d["bbox"][2] - d["bbox"][0], d["bbox"][3] - d["bbox"][1]) >= 16
                  and d["category"] != "sinalizacao"]
            queue += [(iid, d) for d in ds]
        for i in range(0, len(queue), 64):
            items = queue[i:i + 64]
            crops = [square_crop(image(iid), d["bbox"])[0] for iid, d in items]
            for (iid, d), (sub, _, _) in zip(items, clf(crops, [d["category"] for _, d in items])):
                if sub is not None:
                    kept[iid].append(d)
        hit, tot, n = Counter(), Counter(), Counter()
        for iid in imgs:
            n.update(d["category"] for d in kept[iid])
            area = float(imgs[iid]["thumb_w"]) * float(imgs[iid]["thumb_h"])
            for r in refs.get(iid, []):
                b = r["bbox"]
                if r["category"] not in ("iluminacao", "residuos") or max(b[2] - b[0], b[3] - b[1]) < 24:
                    continue
                if r["category"] == "iluminacao" and r.get("subtype") == "poste":
                    continue
                if r["category"] == "residuos" and (box_width_m(b, imgs[iid]) or 0) < MIN_CONTENTOR_M:
                    continue  # papeleira (não pedida)
                tot[r["category"]] += 1
                hit[r["category"]] += any(d["category"] == r["category"] and
                                          matches({"bbox": d["bbox"]}, r, area) for d in kept[iid])
        res = {c: (hit[c] / tot[c] if tot[c] else 0, n[c] / len(imgs)) for c in ("iluminacao", "residuos")}
        frases = "; ".join(f"{c}: " + ", ".join(ps[c]) for c in ("iluminacao", "residuos"))
        print(f"[{name}] iluminação {res['iluminacao'][0]:.0%} ({tot['iluminacao']}) · "
              f"contentores {res['residuos'][0]:.0%} ({tot['residuos']}) · "
              f"deteções/foto {res['iluminacao'][1]:.1f} / {res['residuos'][1]:.1f} · {time.time() - t0:.0f} s")
        lines.append(f"| {name} | {frases} | {res['iluminacao'][0]:.0%} | {res['residuos'][0]:.0%} | "
                     f"{res['iluminacao'][1]:.1f} | {res['residuos'][1]:.1f} |")
    lines += ["", "*Encontrada = % dos objetos marcados pela Mapillary (>= 24 px) que o sistema também "
              "encontrou, depois do filtro 'não é mobiliário'. Deteções/foto: muitas mais = provavelmente "
              "mais falsos alarmes. A Mapillary marca papeleiras como contentores (não pedidas).*"]
    rep = DATA_DIR / "relatorios"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "escolha_prompts_detetor.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {rep / 'escolha_prompts_detetor.md'}")


# ------------------------------------------------------------------ auditar
AUDIT_CATS = ("iluminacao", "residuos")


def tax_ccas(tax):
    chosen = json.loads((OUT / "escolha_prompts.json").read_text(encoding="utf-8"))
    t = copy.deepcopy(tax)
    for cat, classes in chosen["classificador"].items():
        for s in t["categorias"][cat]["subtipos"]:
            if s["id"] in classes:
                s["prompts"] = classes[s["id"]]
    return t


def auditar_preparar(args):
    from classificar import Classifier, square_crop
    rng = random.Random(args.seed)
    pool = {c: [] for c in AUDIT_CATS}
    for base in map(Path, args.pastas):
        with open(base / "meta" / "sample_images.csv", encoding="utf-8") as f:
            fs = {r["id"]: r["file"] for r in csv.DictReader(f)}
        for line in open(base / "preds" / "owlv2.jsonl", encoding="utf-8"):
            d = json.loads(line)
            b = d["bbox"]
            if d["category"] in AUDIT_CATS and d["score"] >= 0.2 and max(b[2] - b[0], b[3] - b[1]) >= 32:
                pool[d["category"]].append((base, fs[d["image_id"]], d))
    dev, tax = device(), load_taxonomy()
    clfs = {"atual": Classifier(tax, dev), "ccas": Classifier(tax_ccas(tax), dev)}
    AUDIT.mkdir(parents=True, exist_ok=True)
    f16 = font(16)
    rows = []
    for cat in AUDIT_CATS:
        sample = rng.sample(pool[cat], min(args.n, len(pool[cat])))
        crops, views = [], []
        for base, fname, d in sample:
            im = Image.open(base / "images" / fname).convert("RGB")
            crops.append(square_crop(im, d["bbox"])[0])           # o que o modelo vê
            x0, y0, x1, y1 = d["bbox"]                             # o que a pessoa vê: foto à volta
            cx, cy, s = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * 0.65
            v = im.crop((int(cx - s), int(cy - s), int(cx + s), int(cy + s)))
            ox, oy = int(cx - s), int(cy - s)
            ImageDraw.Draw(v).rectangle([x0 - ox, y0 - oy, x1 - ox, y1 - oy], outline=(255, 0, 0),
                                        width=max(2, int(s / 60)))
            views.append(v)
        preds = {k: clf(crops, [cat] * len(crops)) for k, clf in clfs.items()}
        cols = 8
        sheet = Image.new("RGB", (cols * 200, ((len(crops) + cols - 1) // cols) * 200), "white")
        dr = ImageDraw.Draw(sheet)
        for i, ((base, fname, d), view) in enumerate(zip(sample, views)):
            x, y = (i % cols) * 200, (i // cols) * 200
            sheet.paste(view.resize((196, 196)), (x + 2, y + 2))
            dr.rectangle([x + 2, y + 2, x + 34, y + 22], fill=(0, 0, 0))
            dr.text((x + 5, y + 3), str(i + 1), fill=(255, 255, 0), font=f16)
            rows.append({"categoria": cat, "n": i + 1, "pasta": base.name, "image_id": d["image_id"],
                         "bbox": json.dumps([round(v) for v in d["bbox"]]),
                         "atual": preds["atual"][i][0] or "nao", "ccas": preds["ccas"][i][0] or "nao",
                         "verdade": ""})
        sheet.save(AUDIT / f"folha_{cat}.jpg", quality=90)
        print(f"{cat}: {len(crops)} recortes -> {AUDIT / f'folha_{cat}.jpg'}")
    with open(AUDIT / "auditoria.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {AUDIT / 'auditoria.csv'}  (preencher a coluna 'verdade')")


def auditar_medir(args):
    with open(AUDIT / "auditoria.csv", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["verdade"].strip()]
    lines = ["# Auditoria da classificação (SigLIP): frases atuais vs CCAS", ""]
    for cat in AUDIT_CATS:
        rs = [r for r in rows if r["categoria"] == cat]
        if not rs:
            continue
        verd = Counter(r["verdade"] for r in rs)
        print(f"\n{cat}: {len(rs)} recortes · verdade: {dict(verd)}")
        lines += [f"## {cat} ({len(rs)} recortes)", "",
                  "Verdade (a olho): " + ", ".join(f"{k} {v}" for k, v in verd.most_common()), "",
                  "| Frases | Tipo certo | Mobiliário verdadeiro descartado | Não-mobiliário aceite |",
                  "|---|---|---|---|"]
        for k in ("atual", "ccas"):
            ok = sum(r[k] == r["verdade"] for r in rs)
            lost = sum(r[k] == "nao" and r["verdade"] != "nao" for r in rs)
            fp = sum(r[k] != "nao" and r["verdade"] == "nao" for r in rs)
            print(f"  {k:<6} certo {ok}/{len(rs)} ({ok / len(rs):.0%}) · descartou verdadeiros {lost} · aceitou falsos {fp}")
            lines.append(f"| {k} | {ok}/{len(rs)} ({ok / len(rs):.0%}) | {lost} | {fp} |")
        lines += ["", "*Tipo certo = o sistema deu a mesma resposta que a pessoa (incluindo 'não é mobiliário').*", ""]
    (AUDIT / "auditoria.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n-> {AUDIT / 'auditoria.md'}")


def main_prompts(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("vocabulario", help="palavras mais perto de cada classe (SigLIP e OWLv2)")
    p.add_argument("--top", type=int, default=8)
    p = sub.add_parser("ccas", help="fase 1: escolher frases nas fotos de referência")
    p.add_argument("--top-cls", type=int, default=3, help="frases por subclasse (SigLIP)")
    p.add_argument("--top-det", type=int, default=4, help="frases por categoria (detetor)")
    p = sub.add_parser("detetor", help="fase 2: medir os conjuntos de frases nas fotos de DATA_DIR")
    p.add_argument("--limite", type=int, default=0)
    p = sub.add_parser("auditar", help="verificação a olho do tipo (preparar / medir)")
    p.add_argument("modo", choices=["preparar", "medir"])
    p.add_argument("--pastas", nargs="+", required=True, help="pastas de dados a auditar (ex.: data data/piloto_rua)")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.cmd == "auditar":
        return auditar_preparar(args) if args.modo == "preparar" else auditar_medir(args)
    {"vocabulario": cmd_vocabulario, "ccas": cmd_ccas, "detetor": cmd_detetor}[args.cmd](args)


####################################################################################################
SECOES = {
    "erros": (main_erros, "relatório de erros"),
    "confianca": (main_confianca, "limiares de confiança"),
    "prompts": (main_prompts, "escolha das frases dos modelos"),
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
