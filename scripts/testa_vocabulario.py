"""
"Dicionário" dos modelos: que palavras (inglês) ficam mais perto das imagens de cada
classe? Serve para escolher as frases (prompts) do detetor e do classificador.

Imagens de referência (NÃO vêm das marcações Mapillary, para não viciar a medição):
  data/referencias/classes/<categoria>/<classe>/*.png|jpg
  (fotos dos mentores e da equipa: candeeiro, poste de luz, subterrâneo, ...)
Vocabulário: config/vocabulario_en.txt (+ as frases atuais da taxonomia e do detetor).

  SigLIP (classificador): semelhança imagem-texto. Para cada classe:
     - "perto"     = expressões mais parecidas com as imagens da classe;
     - "distingue" = parecida com esta classe MENOS a mais parecida das outras classes
                     da mesma categoria (boa para separar, ex.: candeeiro vs poste de luz).
  OWLv2 (detetor): pontuação que o detetor dá ao objeto com cada expressão
     (nas zonas da imagem que ele acha que são objetos). Alta = encontra-o melhor.

Saída: ecrã + data/relatorios/vocabulario.md

Uso:
    python scripts/testa_vocabulario.py
    python scripts/testa_vocabulario.py --top 10
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from estimate_mapillary import DATA_DIR  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data" / "referencias" / "classes"
SIGLIP = "google/siglip-so400m-patch14-384"
OWL = "google/owlv2-base-patch16-ensemble"


def load_refs():
    refs = defaultdict(list)  # (categoria, classe) -> imagens
    for p in sorted(REF.glob("*/*/*")):
        if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
            refs[(p.parent.parent.name, p.parent.name)].append(Image.open(p).convert("RGB"))
    return refs


def load_terms(tax):
    terms = []
    for line in open(ROOT / "config" / "vocabulario_en.txt", encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#"):
            terms.append(line)
    current = set()
    for cat in tax["categorias"].values():
        for s in cat["subtipos"]:
            current.update(s["prompts"])
    from run_detector import PROMPTS
    current.update(PROMPTS)
    for t in current:
        if t not in terms:
            terms.append(t)
    return terms, current


@torch.inference_mode()
def siglip_scores(refs, terms, device):
    from transformers import AutoModel, AutoProcessor
    proc = AutoProcessor.from_pretrained(SIGLIP)
    model = AutoModel.from_pretrained(SIGLIP, dtype=torch.float16).to(device).eval()
    f = lambda x: (x if torch.is_tensor(x) else x.pooler_output).float()  # noqa: E731
    T = []
    for i in range(0, len(terms), 64):
        tok = proc(text=[f"This is a photo of {t}." for t in terms[i:i + 64]],
                   padding="max_length", return_tensors="pt").to(device)
        T.append(f(model.get_text_features(**tok)))
    T = torch.cat(T)
    T = T / T.norm(dim=-1, keepdim=True)
    out = {}
    for key, ims in refs.items():
        pv = proc(images=ims, return_tensors="pt")["pixel_values"].to(device).half()
        E = f(model.get_image_features(pixel_values=pv))
        E = E / E.norm(dim=-1, keepdim=True)
        out[key] = (E @ T.T).mean(0).cpu().numpy()   # semelhança média com as imagens da classe
    del model
    torch.cuda.empty_cache()
    return out


@torch.inference_mode()
def owl_scores(refs, terms, device, k=5):
    """Pontuação (0–1) do detetor para cada expressão, nas k zonas mais 'objeto' de cada imagem."""
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    proc = Owlv2Processor.from_pretrained(OWL)
    model = Owlv2ForObjectDetection.from_pretrained(OWL, dtype=torch.float16).to(device).eval()
    out = {}
    for key, ims in refs.items():
        per_img = []
        for im in ims:
            s = np.zeros(len(terms))
            for i in range(0, len(terms), 64):
                chunk = terms[i:i + 64]
                # o OWLv2 só lê 16 tokens: frases longas ficam cortadas
                x = proc(text=[chunk], images=im, return_tensors="pt", padding="max_length",
                         truncation=True, max_length=16).to(device)
                o = model(input_ids=x["input_ids"], attention_mask=x["attention_mask"],
                          pixel_values=x["pixel_values"].half())
                top = o.objectness_logits[0].float().topk(k).indices
                s[i:i + len(chunk)] = torch.sigmoid(o.logits[0, top].float()).max(0).values.cpu().numpy()
            per_img.append(s)
        out[key] = np.mean(per_img, 0)
    del model
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(ROOT / "config" / "taxonomia.yaml", encoding="utf-8") as f:
        tax = yaml.safe_load(f)
    refs = load_refs()
    terms, current = load_terms(tax)
    print(f"{sum(len(v) for v in refs.values())} imagens de referência em {len(refs)} classes · "
          f"{len(terms)} expressões")

    sig = siglip_scores(refs, terms, device)
    owl = owl_scores(refs, terms, device)

    lines = ["# Vocabulário: que palavras os modelos associam a cada classe", "",
             "Imagens de referência: data/referencias/classes (fotos dos mentores e da equipa). "
             "(atual) = frase que usamos hoje.", ""]

    def mark(t):
        return f"{t} (atual)" if t in current else t

    for key in sorted(refs):
        cat, cls = key
        others = [k for k in refs if k[0] == cat and k != key]
        s = sig[key]
        dist = s - (np.max([sig[k] for k in others], axis=0) if others else 0)
        o = owl[key]
        rank = lambda v: list(np.argsort(-v))  # noqa: E731
        r_s, r_d, r_o = rank(s), rank(dist), rank(o)
        title = f"{cat} / {cls} ({len(refs[key])} imagens)"
        print(f"\n=== {title}")
        lines += [f"## {title}", "",
                  "| # | SigLIP: mais perto | SigLIP: distingue das outras classes | OWLv2 (detetor): encontra melhor |",
                  "|---|---|---|---|"]
        for n in range(args.top):
            row = [f"{mark(terms[r_s[n]])} ({s[r_s[n]]:.3f})",
                   f"{mark(terms[r_d[n]])} ({dist[r_d[n]]:+.3f})" if others else "—",
                   f"{mark(terms[r_o[n]])} ({o[r_o[n]]:.0%})"]
            lines.append(f"| {n + 1} | " + " | ".join(row) + " |")
            print(f"{n + 1:>2}. " + " | ".join(row))
        cur = [t for t in current if t in terms]
        cur_rows = sorted(((r_s.index(terms.index(t)) + 1, r_o.index(terms.index(t)) + 1, t) for t in cur))
        lines += ["", "Posição das frases atuais (de " + str(len(terms)) + "): " +
                  "; ".join(f"\"{t}\" SigLIP {a}.º, OWLv2 {b}.º" for a, b, t in cur_rows[:6]), ""]
    out = DATA_DIR / "relatorios"
    out.mkdir(parents=True, exist_ok=True)
    (out / "vocabulario.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n-> {out / 'vocabulario.md'}")


if __name__ == "__main__":
    main()
