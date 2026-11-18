"""
Análise de erros do sistema, num relatório simples (ecrã + data/relatorios/analise_erros.md).

  1. Encontra os objetos?          (deteção, contra as marcações da Mapillary)
  2. Acerta no tipo de sinal?      (contra as classes da Mapillary)
  3. Lê o número do limite?        (SigLIP; contra as fotos e contra os pontos do mapa)
  4. Acerta no sítio do mapa?      (contra pontos Mapillary + OpenStreetMap)
  5. Há registos repetidos?        (catálogo)

As referências (Mapillary/OSM) também têm erros e não estão completas: os números servem
sobretudo para comparar versões do sistema.

Uso:
    python scripts/analisa_erros.py --model owlv2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from shapely.geometry import Point

sys.path.insert(0, os.path.dirname(__file__))
from estimate_mapillary import DATA_DIR  # noqa: E402
from evaluate import load_jsonl, matches  # noqa: E402
from evaluate_subtypes import mapillary_to_subtype  # noqa: E402
from geolocate import MIN_CONTENTOR_M, box_width_m, project, reference_index, xy  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
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
          "no passo ler_limites.py. Duas verificações:")
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
        for ft in es:
            px, py = xy(*ft["geometry"]["coordinates"])
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="owlv2")
    ap.add_argument("--min-px", type=float, default=24)
    ap.add_argument("--match-m", type=float, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

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


if __name__ == "__main__":
    main()
