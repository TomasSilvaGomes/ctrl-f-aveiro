"""
Corre a pipeline toda, por ordem, com um só comando (cada passo também se corre sozinho):

  passo         comando                        o que faz
  detetar       detetar.py                     encontra objetos (OWLv2)            ~12 min com GPU
  classificar   classificar.py                 tipo de cada objeto; descarta o resto (SigLIP)
  limites       sinais.py limites              número dos limites de velocidade
  codigo        sinais.py codigo               código e nome oficial do Guia da DGV (A25 ...)
  contorno      posicao.py contorno            contorno SAM 2: pé real e pontos 3D do objeto
  posicao       posicao.py calcular            posição no mapa com 1 foto
  catalogo      catalogo.py                    junta repetidos, ilhas de reciclagem -> GeoJSON + CSV
  triangulacao  posicao.py triangulacao        (só com --triangulacao) posição com fotos seguidas
  relatorio     avaliar.py erros               relatório de erros (se houver marcações Mapillary)
  imagens       imagens.py final               (só com --imagens N) N fotos com as caixas

Antes (uma vez): dados.py modelos · dados.py amostra ... · dados.py referencias · dados.py poses

Uso:
    python scripts/pipeline.py                                  # tudo, amostra principal
    python scripts/pipeline.py --desde classificar              # sem voltar a correr o detetor
    python scripts/pipeline.py --desde posicao --ate catalogo   # só posição + catálogo
    python scripts/pipeline.py --dados data/piloto_rua --triangulacao --imagens 6
    python scripts/pipeline.py --apenas-frontais                # refaz o detetor só nas frontais
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["detetar", "classificar", "limites", "codigo", "contorno", "posicao", "catalogo", "triangulacao", "relatorio", "imagens"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dados", help="pasta de dados (por omissão data/ ou DATA_DIR)")
    ap.add_argument("--desde", choices=STEPS, default=STEPS[0])
    ap.add_argument("--ate", choices=STEPS, default=STEPS[-1])
    ap.add_argument("--triangulacao", action="store_true", help="inclui a triangulação (fotos seguidas)")
    ap.add_argument("--imagens", type=int, default=0, help="desenha N fotos com as caixas no fim")
    ap.add_argument("--apenas-frontais", action="store_true", help="detetor: refaz só as fotos frontais")
    ap.add_argument("--sem-tiling", action="store_true", help="detetor: frontais sem as 2 metades")
    args = ap.parse_args()

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    if args.dados:
        env["DATA_DIR"] = str(Path(args.dados).resolve())
    data = Path(env.get("DATA_DIR") or HERE.parent / "data")
    has_refs = (data / "ground_truth" / "mly_detections.jsonl").exists()

    det_args = (["--apenas-frontais"] if args.apenas_frontais else []) + (["--sem-tiling"] if args.sem_tiling else [])
    plan = {
        "detetar": ["detetar.py", *det_args],
        "classificar": ["classificar.py"],
        "limites": ["sinais.py", "limites"],
        "codigo": ["sinais.py", "codigo"],
        "contorno": ["posicao.py", "contorno"],
        "posicao": ["posicao.py", "calcular"],
        "catalogo": ["catalogo.py"],
        "triangulacao": ["posicao.py", "triangulacao"] if args.triangulacao else None,
        "relatorio": ["avaliar.py", "erros"] if has_refs else None,
        "imagens": ["imagens.py", "final", "--n", str(args.imagens)] if args.imagens else None,
    }
    todo = STEPS[STEPS.index(args.desde): STEPS.index(args.ate) + 1]
    print(f"Dados: {data}\nPassos: " + " → ".join(s for s in todo if plan[s]))
    t_all = time.time()
    for step in todo:
        cmd = plan[step]
        if not cmd:
            continue
        print(f"\n{'=' * 70}\n[{step}] {' '.join(cmd)}\n{'=' * 70}", flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, str(HERE / cmd[0]), *cmd[1:]], env=env)
        if r.returncode:
            sys.exit(f"\n[{step}] falhou (código {r.returncode}). Para continuar daqui: --desde {step}")
        print(f"[{step}] feito em {time.time() - t0:.0f} s")
    print(f"\nPipeline terminada em {(time.time() - t_all) / 60:.1f} min · catálogo em {data / 'catalogo'}")


if __name__ == "__main__":
    main()
