"""
Escolher as frases (prompts) que os modelos entendem melhor. Quatro comandos, por ordem:

  vocabulario   "dicionário" dos modelos: que palavras (config/vocabulario_en.txt + frases atuais)
                ficam mais perto das fotos de referência de cada classe?  -> relatorios/vocabulario.md
  ccas          fase 1, sem anotações (método CCAS, arXiv 2505.09139): frases candidatas
                (config/prompts_candidatos.yaml) contra as fotos de referência.
                  SigLIP: perto da SUA subclasse e longe das outras da mesma categoria;
                  OWLv2:  pontuação alta nas fotos da SUA categoria e baixa nas outras.
                -> relatorios/escolha_prompts.json + .md
  detetor       fase 2, medição: corre o detetor nas fotos de DATA_DIR com os conjuntos de frases
                (atual, antigo, ccas) e mede quantos objetos marcados pela Mapillary encontra.
                Para não viciar: escolher numa rua e confirmar noutra (outra DATA_DIR).
                -> <DATA_DIR>/relatorios/escolha_prompts_detetor.md
  auditar       verificação a olho do TIPO (a Mapillary não diz o tipo):
                  preparar  recortes ao acaso, classificados com as frases atuais e as CCAS
                            -> relatorios/auditoria_classificacao/folha_<cat>.jpg + auditoria.csv
                  (pessoa)  preencher a coluna "verdade" (id da classe, ou "nao")
                  medir     acerto de cada conjunto -> auditoria.md

Fotos de referência: data/referencias/classes/<categoria>/<classe>/ (não vêm das marcações
Mapillary, para não viciar a medição).

Uso (PowerShell):
    python scripts/escolhe_prompts.py vocabulario --top 10
    python scripts/escolhe_prompts.py ccas
    $env:DATA_DIR = "data\\piloto_sequencia"; python scripts/escolhe_prompts.py detetor
    python scripts/escolhe_prompts.py auditar preparar --pastas data/piloto_rua data/piloto_sequencia
    python scripts/escolhe_prompts.py auditar medir
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
from comum import DATA_DIR, ROOT, font, load_jsonl, load_taxonomy, matches, nms_by_category  # noqa: E402

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
    import run_detector as rd
    from classify_subtype import Classifier, square_crop, suppress_contained
    from geolocate import MIN_CONTENTOR_M, box_width_m

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
        # mesmo filtro que classify_subtype: NMS, tamanho mínimo, caixas contidas, SigLIP
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
    from classify_subtype import Classifier, square_crop
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


def main():
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
    p.add_argument("--pastas", nargs="+", default=["data/piloto_rua", "data/piloto_sequencia"])
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "auditar":
        return auditar_preparar(args) if args.modo == "preparar" else auditar_medir(args)
    {"vocabulario": cmd_vocabulario, "ccas": cmd_ccas, "detetor": cmd_detetor}[args.cmd](args)


if __name__ == "__main__":
    main()
