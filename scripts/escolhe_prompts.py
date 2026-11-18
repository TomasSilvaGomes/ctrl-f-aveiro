"""
Escolhe as frases (prompts) que os modelos entendem melhor, em 2 fases.

Fase 1 — "ccas" (sem anotações; método CCAS, arXiv 2505.09139):
  Usa as fotos de referência (data/referencias/classes/<categoria>/<classe>/) e as
  frases candidatas (config/prompts_candidatos.yaml; cada frase pertence a 1 classe).
    - Classificador (SigLIP): boa frase = perto das fotos da SUA subclasse e longe das
      outras subclasses da mesma categoria (contraste).
    - Detetor (OWLv2): boa frase = pontuação alta no objeto das fotos da SUA categoria e
      baixa nas fotos das outras categorias.
  Escolhe as --top-cls melhores por subclasse e --top-det melhores por categoria
  -> data/relatorios/escolha_prompts.json + .md

Fase 2 — "detetor" (medição): corre o detetor nas fotos de DATA_DIR com vários
  conjuntos de frases e mede quantos objetos marcados pela Mapillary encontra
  (após o filtro "não é mobiliário" do SigLIP), e quantas deteções faz por foto.
    conjuntos: atual (taxonomia), antigo, ccas (fase 1)
  Para não viciar: escolher numa rua e confirmar noutra (outra DATA_DIR).

Uso (PowerShell):
    python scripts/escolhe_prompts.py --fase ccas
    $env:DATA_DIR = "data\\piloto_sequencia"; python scripts/escolhe_prompts.py --fase detetor
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from estimate_mapillary import DATA_DIR  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data" / "referencias" / "classes"
OUT = ROOT / "data" / "relatorios"
SIGLIP = "google/siglip-so400m-patch14-384"
OWL = "google/owlv2-base-patch16-ensemble"
ANTIGO = {"iluminacao": ["a street light", "a lamp post"],
          "residuos": ["a trash bin", "a waste container", "a recycling container"]}
Image.MAX_IMAGE_PIXELS = None


def load_refs():
    refs = defaultdict(list)
    for p in sorted(REF.glob("*/*/*")):
        if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
            refs[(p.parent.parent.name, p.parent.name)].append(Image.open(p).convert("RGB"))
    return refs


# ------------------------------------------------------------------ fase 1
@torch.inference_mode()
def siglip_embeds(refs, texts, device):
    from transformers import AutoModel, AutoProcessor
    proc = AutoProcessor.from_pretrained(SIGLIP)
    model = AutoModel.from_pretrained(SIGLIP, dtype=torch.float16).to(device).eval()
    f = lambda x: (x if torch.is_tensor(x) else x.pooler_output).float()  # noqa: E731
    T = []
    for i in range(0, len(texts), 64):
        tok = proc(text=[f"This is a photo of {t}." for t in texts[i:i + 64]],
                   padding="max_length", return_tensors="pt").to(device)
        T.append(f(model.get_text_features(**tok)))
    T = torch.cat(T)
    T = T / T.norm(dim=-1, keepdim=True)
    sims = {}
    for key, ims in refs.items():
        pv = proc(images=ims, return_tensors="pt")["pixel_values"].to(device).half()
        E = f(model.get_image_features(pixel_values=pv))
        E = E / E.norm(dim=-1, keepdim=True)
        sims[key] = (E @ T.T).mean(0).cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return sims


@torch.inference_mode()
def owl_scores(refs, texts, device, k=5):
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    proc = Owlv2Processor.from_pretrained(OWL)
    model = Owlv2ForObjectDetection.from_pretrained(OWL, dtype=torch.float16).to(device).eval()
    out = {}
    for key, ims in refs.items():
        per = []
        for im in ims:
            s = np.zeros(len(texts))
            for i in range(0, len(texts), 64):
                chunk = texts[i:i + 64]
                x = proc(text=[chunk], images=im, return_tensors="pt", padding="max_length",
                         truncation=True, max_length=16).to(device)
                o = model(input_ids=x["input_ids"], attention_mask=x["attention_mask"],
                          pixel_values=x["pixel_values"].half())
                top = o.objectness_logits[0].float().topk(k).indices
                s[i:i + len(chunk)] = torch.sigmoid(o.logits[0, top].float()).max(0).values.cpu().numpy()
            per.append(s)
        out[key] = np.mean(per, 0)
    del model
    torch.cuda.empty_cache()
    return out


def fase_ccas(args, device):
    cand = yaml.safe_load(open(ROOT / "config" / "prompts_candidatos.yaml", encoding="utf-8"))
    refs = load_refs()
    print(f"{sum(len(v) for v in refs.values())} fotos de referência: "
          + ", ".join(f"{c}/{k} {len(v)}" for (c, k), v in sorted(refs.items())))
    lines = ["# Escolha das frases (fase 1: contraste nas fotos de referência)", "",
             "Contraste = parecida com a sua classe menos a mais parecida das outras. "
             "Escolhidas = as melhores entre as que também são bem parecidas com a sua classe.", ""]
    result = {"classificador": {}, "detetor": {}}

    # classificador (SigLIP)
    texts = sorted({t for cat in cand["classificador"].values() for ts in cat.values() for t in ts})
    sims = siglip_embeds(refs, texts, device)
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
            lines += [f"## Classificador — {cat} / {cls}", "", "| Frase | Parecida (sua classe) | Contraste | Escolhida |",
                      "|---|---|---|---|"]
            lines += [f"| {t} | {o:.3f} | {c:+.3f} | {'✅' if t in chosen else ''} |" for c, o, t in rows]
            lines.append("")

    # detetor (OWLv2)
    texts = sorted({t for ts in cand["detetor"].values() for t in ts})
    sc = owl_scores(refs, texts, device)
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


# ------------------------------------------------------------------ fase 2
def fase_detetor(args, device):
    import run_detector as rd
    from classify_subtype import Classifier, load_taxonomy, square_crop, suppress_contained
    from evaluate import load_jsonl, matches, nms_by_category
    from geolocate import MIN_CONTENTOR_M, box_width_m

    tax = load_taxonomy()
    chosen = json.loads((OUT / "escolha_prompts.json").read_text(encoding="utf-8"))["detetor"]
    signs = tax["detetor"]["sinalizacao"]
    sets = {"atual": {c: tax["detetor"][c] for c in ("iluminacao", "residuos")},
            "antigo": ANTIGO, "ccas": chosen}
    with open(DATA_DIR / "meta" / "sample_images.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limite:
        rows = rows[: args.limite]
    imgs = {r["id"]: r for r in rows}
    refs = load_jsonl(DATA_DIR / "ground_truth" / "mly_detections.jsonl")
    print(f"{len(rows)} fotos em {DATA_DIR}")

    det = rd.Owlv2(device)
    clf = Classifier(tax, device)
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
    results = {}
    for name, ps in sets.items():
        t0 = time.time()
        prompts = {p: ("sinalizacao", "") for p in signs}
        for cat in ("iluminacao", "residuos"):
            prompts.update({p: (cat, "") for p in ps[cat]})
        rd.PROMPTS.clear()
        rd.PROMPTS.update(prompts)
        tok = det.proc(text=[list(prompts)], return_tensors="pt")
        det.ids, det.mask = tok["input_ids"].to(device), tok["attention_mask"].to(device)
        dets = defaultdict(list)
        batch = [(iid, info, a) for iid, vs in views.items() for a, info in vs]
        for i in range(0, len(batch), 12):
            items = batch[i:i + 12]
            for (iid, info, _), res in zip(items, det([a for _, _, a in items], 0.1)):
                for box, score, prompt in res:
                    ob = rd.to_original(box, info)
                    if ob and score >= 0.2:
                        dets[iid].append({"bbox": ob, "score": score, "category": prompts[prompt][0],
                                          "prompt": prompt})
        # mesmo filtro que classify_subtype: NMS, tamanho mínimo, caixas contidas, SigLIP
        kept = defaultdict(list)
        queue = []
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
        results[name] = res
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fase", choices=["ccas", "detetor"], required=True)
    ap.add_argument("--top-cls", type=int, default=3, help="frases por subclasse (SigLIP)")
    ap.add_argument("--top-det", type=int, default=4, help="frases por categoria (detetor)")
    ap.add_argument("--limite", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fase_ccas(args, device) if args.fase == "ccas" else fase_detetor(args, device)


if __name__ == "__main__":
    main()
