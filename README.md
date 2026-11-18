# Ctrl+F Aveiro — catálogo de mobiliário urbano

Hackathon Aveiro Tech City 2026, Desafio #2. A partir de fotos abertas da Mapillary, o
sistema **encontra, classifica e põe no mapa** sinais de trânsito, iluminação pública e
contentores de resíduos de Aveiro, e exporta um catálogo (GeoJSON/CSV).

| Categoria | Classes |
|---|---|
| Sinalização vertical | STOP, cedência, limites de velocidade (com o número), sentido proibido, … |
| Iluminação pública | Poste de luz, Candeeiro |
| Resíduos | Ilha de reciclagem (ecoponto), Contentor indiferenciado, Contentor subterrâneo |

Modelos (todos Apache 2.0): **OWLv2** encontra os objetos (só diz a categoria); **SigLIP** dá o
tipo, lê o número dos limites e descarta o que não é mobiliário. Classes, frases e regras estão
em [config/taxonomia.yaml](config/taxonomia.yaml).

## 1. Instalar (Windows, PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env        # e colar o teu token da Mapillary no .env
.\.venv\Scripts\python.exe scripts\descarregar_modelos.py
$env:PYTHONIOENCODING = "utf-8"    # acentos no terminal
```

O `.env` é pessoal e **nunca** vai para o repositório.

## 2. Descarregar as fotos (não estão no repositório: ~2,9 GB)

```powershell
# amostra principal (3 050 fotos, ~2,2 GB)
.\.venv\Scripts\python.exe scripts\amostra.py descarregar --ids-from data\meta\sample_images.csv --n 100000 --size original --workers 10
# ruas piloto (fotos seguidas), cada uma na sua pasta
foreach ($p in "piloto_rua", "piloto_sequencia", "piloto_frontal") {
  $env:DATA_DIR = "data\$p"
  .\.venv\Scripts\python.exe scripts\amostra.py descarregar --ids-from data\$p\meta\sample_images.csv --n 100000 --size original
}
Remove-Item Env:DATA_DIR
.\.venv\Scripts\python.exe scripts\fetch_ground_truth.py     # pontos Mapillary/OSM (só para medir)
```

## 3. Correr a pipeline (um comando)

```powershell
.\.venv\Scripts\python.exe scripts\pipeline.py                         # tudo (~15 min com GPU)
.\.venv\Scripts\python.exe scripts\pipeline.py --desde classificar     # sem voltar a correr o detetor (~1 min)
.\.venv\Scripts\python.exe scripts\pipeline.py --dados data\piloto_rua --triangulacao --imagens 6
```

Passos, por ordem (cada um também se corre sozinho; `--desde` / `--ate` escolhem o intervalo):

| Passo | Script | O que faz | Sai para |
|---|---|---|---|
| detetar | `run_detector.py` | encontra objetos (OWLv2); frontais com a foto inteira + 2 metades | `preds/owlv2.jsonl` |
| classificar | `classify_subtype.py` | tipo de cada objeto, descarta o que não é mobiliário (SigLIP) | `preds/owlv2_classified.jsonl` |
| limites | `ler_limites.py` | número dos limites de velocidade | (mesmo ficheiro) |
| posicao | `geolocate.py` | posição no mapa a partir de 1 foto | `preds/owlv2_geo.jsonl` |
| catalogo | `build_catalog.py` | junta repetidos, forma ilhas de reciclagem | `catalogo/catalogo.geojson` + `.csv` |
| triangulacao | `triangulacao.py` | (opcional) posição com várias fotos seguidas | `preds/owlv2_tri.jsonl` |
| relatorio | `analisa_erros.py` | deteção, tipo, limites, posição, repetidos | `relatorios/analise_erros.md` |
| imagens | `visualize.py final` | (opcional) fotos com as caixas | `viz/` |

Os resultados já calculados estão em `data/preds`, `data/catalogo` e `data/relatorios`.
A classificação guarda a "impressão digital" de cada recorte em `preds/siglip_cache.npz`
(não está no repositório; a primeira corrida cria-a, ~20 min, e as seguintes só calculam recortes novos).
**Tudo de uma vez no tmux (Ubuntu/WSL):** `bash scripts/sessao_tmux.sh`.

## 4. Outros scripts

| Script | Comandos | Para quê |
|---|---|---|
| `amostra.py` | `indice` · `selecionar` · `descarregar` | Criar uma amostra noutro município (todas as fotos → fotos urbanas → download) |
| `captura_sequencia.py` | `--lat --lon` ou `--inicio --direcao` | Fotos seguidas de uma rua (por ponto ou por rua) → pasta piloto |
| `fetch_ground_truth.py` | (nada) · `--listar-tipos` | Pontos Mapillary/OSM para medir · que tipos de points há num município |
| `visualize.py` | `final` · `owl` · `foto <ficheiro>` | Desenhar o resultado final · o OWLv2 em bruto · a cadeia numa foto qualquer |
| `folha_catalogo.py` | `folha` · `guardar` | Rever exemplos de um tipo · guardar os bons como fotos de referência |
| `escolhe_prompts.py` | `vocabulario` · `ccas` · `detetor` · `auditar preparar/medir` | Escolher as frases que os modelos entendem melhor e verificá-las a olho |
| `semelhanca_sinais.py` | `extrair` · `testar` | Experiência: código exato do sinal por semelhança com os desenhos da DGV |
| `descarregar_modelos.py` | | Descarregar e testar os modelos (instalação nova) |

`comum.py`, `mapillary.py` e `panorama.py` são peças partilhadas (não se correm sozinhas).
Ajuda de cada comando: `python scripts\<script>.py -h`.

**Outra pasta de dados:** `--dados data\piloto_rua` no `pipeline.py`, ou `$env:DATA_DIR = "data\piloto_rua"`
antes de qualquer script.

## 5. Pastas

```
config/            taxonomia.yaml (classes, frases, regras), frases candidatas, vocabulário
scripts/           código (ver acima)
data/meta/         lista das fotos da amostra (sample_images.csv) e índice do concelho
data/ground_truth/ marcações e pontos Mapillary/OSM — SÓ para medir
data/preds/        saídas da pipeline (deteções, tipos, posições)
data/catalogo/     catálogo final (GeoJSON + CSV)
data/relatorios/   relatórios de erros, escolha das frases, folhas de revisão
data/referencias/  fotos de referência por classe (créditos em fontes*.csv) e desenhos dos sinais
data/piloto_*/     ruas piloto (fotos seguidas), com a mesma estrutura
```

## 6. Regras (ficha do desafio)

- Fotos Mapillary: **CC BY-SA 4.0** — mostrar sempre autor + link (o catálogo guarda-os).
- **Rostos e matrículas** nunca guardados nem mostrados (desfocar antes de mostrar recortes).
- Pontos Mapillary e OpenStreetMap (ODbL) servem **só para medir**, nunca entram no catálogo.
- Proibido: Google Street View; código AGPL (YOLOE e restantes modelos Ultralytics, PyMuPDF);
  pesos Mapillary Vistas / Objects365. YOLO-World original é GPL-3.0: só se o projeto todo
  for publicado como GPL (por decidir).
- Os PDFs (ficha do desafio, guia da DGV) não estão no repositório: pô-los em `pdfs/`
  (`semelhanca_sinais.py extrair` precisa de `pdfs/SinalizacaoRodoviaria.pdf`).
