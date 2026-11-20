# Ctrl+F Aveiro — catálogo de mobiliário urbano

Hackathon Aveiro Tech City 2026, Desafio #2 (equipa Transformers). A partir de fotos abertas da
Mapillary, o sistema **encontra, classifica e põe no mapa** sinais de trânsito, iluminação pública
e contentores de resíduos de Aveiro, e exporta um catálogo (GeoJSON/CSV).

| Categoria | Classes |
|---|---|
| Sinalização vertical | código e nome oficial do Guia da DGV (STOP, A25 "Trânsito nos dois sentidos", limites com o número, …) |
| Iluminação pública | Poste de luz, Candeeiro |
| Resíduos | Ilha de reciclagem (ecoponto), Contentor indiferenciado, Contentor subterrâneo |

Modelos (todos Apache 2.0, sem treino próprio): **OWLv2** encontra os objetos; **SigLIP** dá o
tipo, lê os limites de velocidade e o código oficial do sinal, e descarta o que não é mobiliário;
**SAM 2** recorta o contorno (pé real do objeto). Classes, frases, regras e limiares de confiança
estão em [config/taxonomia.yaml](config/taxonomia.yaml).

## 1. Instalar (Windows, PowerShell)

Mais simples, um script que faz tudo (ambiente, bibliotecas com as versões de `versoes.txt`, modelos com o
SAM 2.1 small, recortes da app) e, com a pasta de uma pen, copia o que não está no git
(`weights/`, `data/app/`, `data/images/`, `.env`, `hf/`):

```bash
bash instalar_nvidia.sh [pasta_da_pen]   # PC com placa NVIDIA (Windows no Git Bash, ou Linux)
bash instalar_macos.sh  [pasta_da_pen]   # Mac com chip Apple (análise de fotos novas no CPU)
```

À mão:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Set-Content .env "MAPILLARY_TOKEN=MLY|..." -Encoding ascii  # token da Mapillary (dashboard/developers); ascii = sem BOM
.\.venv\Scripts\python.exe scripts\dados.py modelos
$env:PYTHONIOENCODING = "utf-8"    # acentos no terminal
```

O `.env` é pessoal e **nunca** vai para o repositório.

## 2. Descarregar os dados (as fotos não estão no repositório: ~2,9 GB)

```powershell
# amostra principal (3 050 fotos, ~2,2 GB)
.\.venv\Scripts\python.exe scripts\dados.py amostra descarregar --ids-from data\meta\sample_images.csv --n 100000 --size original --workers 10
.\.venv\Scripts\python.exe scripts\dados.py poses          # rotação da câmara + pontos 3D (pasta sfm/)
.\.venv\Scripts\python.exe scripts\dados.py referencias    # pontos Mapillary/OSM (só para medir)
# rua piloto (fotos seguidas num só sentido, pasta própria), ex.: Av. Dr. Lourenço Peixinho
.\.venv\Scripts\python.exe scripts\dados.py sequencia --inicio 40.642207,-8.651474 --direcao 40.644088,-8.639894 `
  --comprimento 1000 --largura 12 --tipo 360 --sentido --passagens 1 --saida data\piloto_rua
$env:DATA_DIR = "data\piloto_rua"; .\.venv\Scripts\python.exe scripts\dados.py poses; Remove-Item Env:DATA_DIR
```

## 3. Correr a pipeline (um comando)

```powershell
.\.venv\Scripts\python.exe scripts\pipeline.py                         # tudo (~15 min com GPU; contorno com o SAM 2 small, por omissão)
.\.venv\Scripts\python.exe scripts\pipeline.py --desde classificar     # sem voltar a correr o detetor
.\.venv\Scripts\python.exe scripts\pipeline.py --dados data\piloto_rua --triangulacao --imagens 6
# vídeo gravado na rua (com trilho GPX): cortar em fotos e correr tudo
.\.venv\Scripts\python.exe scripts\dados.py video gravacao.mp4 --gpx trilho.gpx --saida data\video_rua
$env:DATA_DIR = "data\video_rua"
.\.venv\Scripts\python.exe scripts\dados.py poses --opensfm    # rotação + pontos 3D NOSSOS (OpenSfM em Docker, sem API)
Remove-Item Env:DATA_DIR
.\.venv\Scripts\python.exe scripts\pipeline.py --dados data\video_rua --triangulacao
# imagens para a app: uma por elemento do catálogo, já desenhada (~30 s)
.\.venv\Scripts\python.exe scripts\imagens.py app
```

A imagem Docker do OpenSfM faz-se uma vez (Docker Desktop ligado):
`git clone --recurse-submodules https://github.com/mapillary/OpenSfM` e
`docker build -f OpenSfM/Dockerfile.ubuntu24 -t opensfm:ctrlf OpenSfM`.

Passos, por ordem (cada um também se corre sozinho; `--desde` / `--ate` escolhem o intervalo):

| Passo | Comando | O que faz | Sai para |
|---|---|---|---|
| detetar | `detetar.py` | encontra objetos (OWLv2); frontais com a foto inteira + 2 metades | `preds/owlv2.jsonl` |
| classificar | `classificar.py` | tipo de cada objeto, descarta o que não é mobiliário (SigLIP) | `preds/owlv2_classified.jsonl` |
| limites | `sinais.py limites` | número dos limites de velocidade | (mesmo ficheiro) |
| codigo | `sinais.py codigo` | código e nome oficial do sinal (desenhos do Guia da DGV) | (mesmo ficheiro) |
| contorno | `posicao.py contorno` | contorno SAM 2: pé real e pontos 3D só do objeto | (mesmo ficheiro) |
| posicao | `posicao.py calcular` | posição no mapa a partir de 1 foto | `preds/owlv2_geo.jsonl` |
| catalogo | `catalogo.py` | aplica os limiares de confiança, junta repetidos, ilhas de reciclagem | `catalogo/catalogo.geojson` + `.csv` |
| triangulacao | `posicao.py triangulacao` | (opcional) posição com várias fotos seguidas | `preds/owlv2_tri.jsonl` |
| relatorio | `avaliar.py erros` | deteção, tipo, limites, posição, repetidos | `relatorios/analise_erros.md` |
| imagens | `imagens.py final` | (opcional) fotos com nome e confiança | `viz/` |

Os resultados já calculados estão em `data/preds`, `data/catalogo` e `data/relatorios`.
Exceção: a saída em bruto do detetor da amostra principal (`data/preds/owlv2.jsonl`, 52 MB, acima
do limite recomendado do GitHub) não está no repositório; numa cópia nova, correr primeiro o passo
`detetar` (~12 min com GPU) antes de `--desde classificar`.
O que se volta a gerar e é pesado não vai para o repositório: `preds/siglip_cache.npz` (memória
dos recortes, a primeira corrida cria-a), `sfm/` (pontos 3D) e `viz/` (imagens).

## 4. A app (mapa e fichas, pasta `app/`)

```powershell
.\.venv\Scripts\python.exe scripts\imagens.py app        # recortes já desenhados (~30 s): data/app/
.\.venv\Scripts\python.exe -m streamlit run app\app.py   # http://localhost:8501
```

- Lê `data/app/catalogo.geojson` (o catálogo + campo `imagem`); sem ele, `data/catalogo/catalogo.geojson`.
  Outro ficheiro: variável `CTRL_F_CATALOG_PATH`. Ao carregar num ícone mostra o recorte da pipeline
  (contentores com o contorno SAM 2, sinais com o desenho da DGV); nas ilhas, um por contentor.
- Fotos enviadas na app: mesma cadeia da pipeline (`detetar.py` → `classificar.py` → `sinais.py`), caras e
  matrículas tapadas com o EgoBlur Gen2 (`comum.Desfocar`) e contorno com o SAM 2 small (cache do Hugging Face).
- Fotos enviadas ficam **por rever**; na Administração aprovam-se (entram no mapa e nas exportações) ou rejeitam-se.
- As edições feitas na app ficam em `data/app/catalogo.geojson` e `contribuicoes.sqlite3`: voltar a correr
  `imagens.py app` refaz o catálogo da app (as edições no GeoJSON perdem-se).

## 5. Os scripts

| Ficheiro | Comandos | Para quê |
|---|---|---|
| `pipeline.py` | `--desde` · `--ate` · `--dados` | Corre os passos por ordem |
| `dados.py` | `amostra` (indice · selecionar · descarregar) · `sequencia` · `referencias` · `poses` · `modelos` · `video` | Tudo o que traz dados: fotos, ruas seguidas, pontos para medir, pontos 3D, modelos; e um vídeo próprio + GPX → uma foto a cada N metros |
| `detetar.py` | | Encontra os objetos (OWLv2) |
| `classificar.py` | | Tipo de cada objeto (SigLIP) |
| `sinais.py` | `limites` · `codigo` · `extrair` | Número dos limites · código oficial · desenhos do PDF da DGV |
| `posicao.py` | `contorno` · `calcular` · `triangulacao` | Pé real (SAM 2) · posição com 1 foto · posição com várias fotos |
| `catalogo.py` | `--limiares` | Catálogo final (GeoJSON + CSV) |
| `avaliar.py` | `erros` · `confianca` · `prompts` | Relatório de erros · escolha dos limiares de confiança · escolha das frases dos modelos |
| `imagens.py` | `final` · `owl` · `foto` · `contorno` · `folha` · `guardar` · `app` | Fotos com caixas · detetor em bruto · foto qualquer · caixa vs contorno · rever e guardar exemplos · **uma imagem por elemento para a app** (`data/app/imagens/<id>.jpg` + `data/app/catalogo.geojson/.csv` com o campo `imagem`) |
| `comum.py`, `mapillary.py` | | Peças partilhadas (não se correm sozinhas) |

Ajuda de cada comando: `python scripts\<ficheiro>.py <comando> -h`.

**Outra pasta de dados:** `--dados data\piloto_rua` no `pipeline.py`, ou `$env:DATA_DIR = "data\piloto_rua"`
antes de qualquer script.

## 6. Pastas

```
config/            taxonomia.yaml (classes, frases, regras, limiares), frases candidatas, vocabulário
scripts/           código (ver acima)
data/meta/         fotos da amostra (sample_images.csv), índice do concelho, poses das câmaras
data/ground_truth/ marcações e pontos Mapillary/OSM — SÓ para medir
data/preds/        saídas da pipeline (deteções, tipos, contornos, posições)
data/catalogo/     catálogo final (GeoJSON + CSV)
data/relatorios/   relatórios de erros, confiança, escolha das frases
data/referencias/  fotos de referência por classe (créditos em fontes*.csv) e desenhos dos sinais da DGV
data/app/          catálogo + recortes para a app (imagens.py app)
data/images/, data/sfm/  fotos e pontos 3D (só locais)
```

## 7. Regras (ficha do desafio)

- Fotos Mapillary: **CC BY-SA 4.0** — mostrar sempre autor + link (o catálogo guarda-os).
- **Rostos e matrículas** nunca guardados nem mostrados. A Mapillary já os desfoca; os nossos vídeos passam pelo
  **EgoBlur Gen2** (`comum.Desfocar`) antes de gravar cada foto. Pesos: descarregar em https://www.projectaria.com/tools/egoblur
  e pôr `ego_blur_face_gen2.jit` e `ego_blur_lp_gen2.jit` em `weights/` (fora do git).
- Pontos Mapillary e OpenStreetMap (ODbL) servem **só para medir**, nunca entram no catálogo.
- Proibido: Google Street View; código AGPL (YOLOE e restantes modelos Ultralytics, PyMuPDF);
  pesos Mapillary Vistas / Objects365. O YOLO-World (GPL-3.0) foi testado à parte e posto de parte.
- Os PDFs (ficha do desafio, guia da DGV) não estão no repositório: pô-los em `pdfs/`
  (`sinais.py extrair` precisa de `pdfs/SinalizacaoRodoviaria.pdf`).
