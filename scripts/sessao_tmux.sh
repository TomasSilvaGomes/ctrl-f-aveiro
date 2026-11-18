#!/usr/bin/env bash
# Sessão tmux (Ubuntu/WSL): corre a pipeline na amostra principal e nas ruas piloto, uma a
# seguir à outra (a placa gráfica é só uma), com logs em ~/hackaton_logs.
#
#   bash scripts/sessao_tmux.sh                    # principal sem detetor + pilotos completos
#   DESDE=detetar bash scripts/sessao_tmux.sh      # principal também com o detetor (~12 min)
#   tmux attach -t hackaton                        # ver | Ctrl-b n: janela seguinte | Ctrl-b d: sair sem parar
set -euo pipefail
cd "$(dirname "$0")/.."

# Python: o .venv do Windows (via WSL) ou o do Linux
if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe
elif [ -x .venv/bin/python ]; then PY=.venv/bin/python
else PY=python3; fi
DESDE="${DESDE:-classificar}"
LOGS="$HOME/hackaton_logs"
mkdir -p "$LOGS"
export PYTHONIOENCODING=utf-8

RUN="$PY scripts/pipeline.py --desde $DESDE 2>&1 | tee $LOGS/principal.log; \
for p in piloto_rua piloto_sequencia piloto_frontal; do \
  $PY scripts/pipeline.py --dados data/\$p --triangulacao 2>&1 | tee $LOGS/\$p.log; \
done; echo; echo 'Tudo terminado. Logs em $LOGS'; exec bash"

tmux kill-session -t hackaton 2>/dev/null || true
tmux new-session -d -s hackaton -n pipeline -c "$PWD" "bash -c \"$RUN\""
if command -v nvidia-smi >/dev/null; then
  tmux new-window -t hackaton -n gpu "watch -n 2 nvidia-smi"
fi
tmux select-window -t hackaton:pipeline
echo "Sessão 'hackaton' criada. Ver: tmux attach -t hackaton   (logs em $LOGS)"
