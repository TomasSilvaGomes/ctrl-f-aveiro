#!/usr/bin/env bash
# Instala tudo o que a app e a pipeline precisam num Mac com chip Apple (M1/M2/M3/M4).
#
#   bash instalar_macos.sh               # ambiente Python + modelos de IA (OWLv2, SigLIP, SAM 2.1 small)
#   bash instalar_macos.sh /caminho/pen  # + copia da pen o que não está no git: weights/, data/app/,
#                                        #   data/images/, .env e hf/ (modelos já descarregados)
#
# Depois: .venv/bin/python -m streamlit run app/app.py
# Sem CUDA no Mac: a análise de fotos novas corre no CPU (mais lenta); ver o mapa e as fichas é igual.
set -euo pipefail
cd "$(dirname "$0")"
PEN="${1:-}"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY=.venv/bin/python

passo() { printf '\n== %s\n' "$*"; }
aviso() { printf '   AVISO: %s\n' "$*"; }

passo "Mac com chip Apple"
[ "$(uname -s)" = Darwin ] || { echo "Este script é para macOS (no PC com NVIDIA: instalar_nvidia.sh)."; exit 1; }
[ "$(uname -m)" = arm64 ] || aviso "não é um chip Apple (arm64): deve funcionar, mas não foi pensado para isso"
sw_vers -productVersion

passo "Python 3.11 e ambiente .venv"
if [ ! -f "$PY" ]; then
  BASE=""
  for b in python3.11 /opt/homebrew/bin/python3.11 python3; do
    if $b -c 'import sys; assert sys.version_info[:2] == (3, 11)' >/dev/null 2>&1; then BASE="$b"; break; fi
  done
  if [ -z "$BASE" ] && command -v brew >/dev/null 2>&1; then
    brew install python@3.11
    BASE=/opt/homebrew/bin/python3.11
  fi
  [ -n "$BASE" ] || { echo "Falta o Python 3.11: instalar o Homebrew (https://brew.sh) e voltar a correr, ou python.org."; exit 1; }
  $BASE -m venv .venv
fi
"$PY" --version

passo "Bibliotecas (PyTorch para Mac + versões exatas de versoes.txt, sem as de CUDA)"
TORCH=$(grep -E '^torch==' versoes.txt | sed -E 's/\+.*//')
VISION=$(grep -E '^torchvision==' versoes.txt | sed -E 's/\+.*//')
"$PY" -m pip install "$TORCH" "$VISION"
RESTO=$(mktemp)
grep -vE '^(torch|torchvision)==|^--extra-index-url|^#' versoes.txt > "$RESTO"
if ! "$PY" -m pip install -r "$RESTO"; then
  aviso "alguma versão exata não existe para macOS: a instalar pelo requirements.txt (versões mínimas)"
  "$PY" -m pip install -r requirements.txt
fi
rm -f "$RESTO"

if [ -n "$PEN" ]; then
  passo "Copiar da pen ($PEN) o que falta aqui"
  "$PY" - "$PEN" <<'EOF'
import shutil, sys
from pathlib import Path
pen = Path(sys.argv[1])
for nome in ("weights/ego_blur_face_gen2.jit", "weights/ego_blur_lp_gen2.jit", ".env", "data/app", "data/images"):
    origem, destino = pen / nome, Path(nome)
    if not origem.exists() or (destino.exists() and any(destino.iterdir() if destino.is_dir() else [1])):
        continue
    destino.parent.mkdir(parents=True, exist_ok=True)
    (shutil.copytree if origem.is_dir() else shutil.copy2)(origem, destino, **({"dirs_exist_ok": True} if origem.is_dir() else {}))
    print("   copiado:", nome)
if (pen / "hf" / "hub").is_dir():                       # modelos já descarregados (poupa ~4 GB de download)
    from huggingface_hub.constants import HF_HUB_CACHE
    for pasta in (pen / "hf" / "hub").glob("models--*"):
        shutil.copytree(pasta, Path(HF_HUB_CACHE) / pasta.name, dirs_exist_ok=True)
        print("   modelo copiado:", pasta.name)
EOF
fi

passo "Modelos de IA (OWLv2, SigLIP, SAM 2.1 small -> cache do Hugging Face; o que já existe não se repete)"
"$PY" scripts/dados.py modelos

passo "Recortes da app (data/app)"
if [ -f data/app/catalogo.geojson ]; then
  echo "   já existem"
elif ls data/images/*.jpg >/dev/null 2>&1; then
  "$PY" scripts/imagens.py app
else
  aviso "faltam as fotos em data/images: copiar da pen (bash instalar_macos.sh /caminho/pen) ou descarregar
         com o token no .env (README, passo 2); depois: $PY scripts/imagens.py app"
fi

passo "Verificação"
"$PY" - <<'EOF'
import sys
from pathlib import Path
import torch
sys.path.insert(0, "app")
from photo_preview import sam_available
ok = lambda v: "sim" if v else "NÃO"
print("   PyTorch:        ", torch.__version__, "(CPU; MPS disponível:", ok(torch.backends.mps.is_available()) + ")")
print("   SAM 2.1 small:  ", ok(sam_available()))
print("   EgoBlur Gen2:   ", ok(all((Path("weights") / n).is_file() for n in ("ego_blur_face_gen2.jit", "ego_blur_lp_gen2.jit"))),
      "(desfoque; sem ele o envio de fotos pára — https://www.projectaria.com/tools/egoblur)")
print("   Recortes da app:", ok(Path("data/app/catalogo.geojson").is_file()))
print("   Fotos:          ", len(list(Path("data/images").glob("*.jpg"))), "em data/images")
EOF
printf '\nPronto. App: %s -m streamlit run app/app.py\n' "$PY"
