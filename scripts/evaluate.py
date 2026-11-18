"""
Compara as deteções de um modelo com as deteções de referência da Mapillary,
imagem a imagem, por categoria.

ATENÇÃO à interpretação: a referência é a saída do modelo da Mapillary, não
anotação humana. "Recall" = % dos objetos que a Mapillary viu que nós também
vimos. "Concordância" = % das nossas deteções que a Mapillary também tem (é um
limite inferior da precisão: a Mapillary também falha objetos).

Correspondência: mesma categoria e (IoU >= 0.3 ou o centro da caixa de
referência dentro da nossa caixa). A segunda regra existe porque a Mapillary
marca só a cabeça do candeeiro, e um detetor pode marcar o candeeiro inteiro.

Uso:
    python scripts/evaluate.py --model owlv2 --min-px 24
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from estimate_mapillary import DATA_DIR  # noqa: E402

CATS = ["sinalizacao", "iluminacao", "residuos", "semaforos"]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


def matches(pred, ref, img_area):
    pb, rb = pred["bbox"], ref["bbox"]
    if iou(pb, rb) >= 0.3:
        return True
    cx, cy = (rb[0] + rb[2]) / 2, (rb[1] + rb[3]) / 2
    p_area = (pb[2] - pb[0]) * (pb[3] - pb[1])
    return pb[0] <= cx <= pb[2] and pb[1] <= cy <= pb[3] and p_area <= 0.15 * img_area


def nms_by_category(dets, thr=0.5):
    """Vários prompts podem apontar ao mesmo objeto: fica a caixa com maior score."""
    keep = []
    for d in sorted(dets, key=lambda d: -d["score"]):
        if all(k["category"] != d["category"] or iou(k["bbox"], d["bbox"]) < thr for k in keep):
            keep.append(d)
    return keep


def load_jsonl(path):
    by_img = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            by_img[d["image_id"]].append(d)
    return by_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="owlv2")
    ap.add_argument("--min-px", type=float, default=24,
                    help="ignora referências cujo lado maior é menor (objetos longínquos)")
    ap.add_argument("--thresholds", default="0.1,0.15,0.2,0.3")
    ap.add_argument("--subtype-threshold", type=float, default=0.1)
    args = ap.parse_args()

    with open(DATA_DIR / "meta" / "sample_images.csv", encoding="utf-8") as f:
        imgs = {r["id"]: r for r in csv.DictReader(f)}
    refs = load_jsonl(DATA_DIR / "ground_truth" / "mly_detections.jsonl")
    preds = load_jsonl(DATA_DIR / "preds" / f"{args.model}.jsonl")

    def big(d):
        b = d["bbox"]
        return max(b[2] - b[0], b[3] - b[1]) >= args.min_px

    print(f"Modelo: {args.model}  |  imagens: {len(imgs)}  |  "
          f"referências com lado >= {args.min_px:.0f}px (postes de eletricidade excluídos)\n")
    header = f"{'limiar':>6} | " + " | ".join(f"{c:^30}" for c in CATS)
    print(header)
    print(f"{'':>6} | " + " | ".join(f"{'refs  recall  preds  concord.':^30}" for _ in CATS))
    print("-" * len(header))

    for thr in [float(t) for t in args.thresholds.split(",")]:
        stats = {c: dict(refs=0, hit=0, preds=0, agree=0) for c in CATS}
        for iid, meta in imgs.items():
            area = float(meta["thumb_w"]) * float(meta["thumb_h"])
            r_img = [r for r in refs.get(iid, []) if big(r) and r["subtype"] != "poste"]
            p_img = nms_by_category([p for p in preds.get(iid, []) if p["score"] >= thr and big(p)])
            for c in CATS:
                rc = [r for r in r_img if r["category"] == c]
                pc = [p for p in p_img if p["category"] == c]
                stats[c]["refs"] += len(rc)
                stats[c]["preds"] += len(pc)
                stats[c]["hit"] += sum(any(matches(p, r, area) for p in pc) for r in rc)
                stats[c]["agree"] += sum(any(matches(p, r, area) for r in rc) for p in pc)
        cells = []
        for c in CATS:
            s = stats[c]
            rec = s["hit"] / s["refs"] if s["refs"] else 0
            agr = s["agree"] / s["preds"] if s["preds"] else 0
            cells.append(f"{s['refs']:>5} {rec:>6.0%} {s['preds']:>6} {agr:>8.0%}  ")
        print(f"{thr:>6.2f} | " + " | ".join(f"{x:^30}" for x in cells))

    subtype_report(imgs, refs, preds, big, args.subtype_threshold)


SIGN_SUBTYPES = ("stop", "cedencia", "limite_velocidade")


def subtype_report(imgs, refs, preds, big, thr):
    """Subtipo dos sinais: para cada sinal de referência stop/cedência/limite que
    foi detetado, compara com o subtipo da nossa melhor deteção correspondente."""
    from collections import Counter
    conf = Counter()
    for iid, meta in imgs.items():
        area = float(meta["thumb_w"]) * float(meta["thumb_h"])
        rs = [r for r in refs.get(iid, []) if r["category"] == "sinalizacao"
              and r["subtype"] in SIGN_SUBTYPES and big(r)]
        if not rs:
            continue
        ps = [p for p in preds.get(iid, []) if p["category"] == "sinalizacao" and p["score"] >= thr]
        for r in rs:
            cand = [p for p in ps if matches(p, r, area)]
            if not cand:
                conf[(r["subtype"], "nao_detetado")] += 1
                continue
            # entre as deteções sobre o mesmo sinal, a de maior score decide o subtipo
            best = max(cand, key=lambda p: p["score"])
            pred_sub = best["subtype"] if best["subtype"] in SIGN_SUBTYPES else "sinal_generico"
            conf[(r["subtype"], pred_sub)] += 1
    cols = list(SIGN_SUBTYPES) + ["sinal_generico", "nao_detetado"]
    print(f"\nSubtipo dos sinais (limiar {thr}) — linhas: referência Mapillary, colunas: nosso modelo")
    print(f"{'':>18}" + "".join(f"{c:>16}" for c in cols) + f"{'acerto':>9}")
    for r in SIGN_SUBTYPES:
        row = [conf[(r, c)] for c in cols]
        detected = sum(row[:-1])
        acc = conf[(r, r)] / detected if detected else 0
        print(f"{r:>18}" + "".join(f"{v:>16}" for v in row) + f"{acc:>9.0%}")


if __name__ == "__main__":
    main()
