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

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env        # e colar o teu token da Mapillary no .env
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
# ruas piloto (fotos seguidas), cada uma na sua pasta
foreach ($p in "piloto_rua", "piloto_sequencia", "piloto_frontal") {
  $env:DATA_DIR = "data\$p"
  .\.venv\Scripts\python.exe scripts\dados.py amostra descarregar --ids-from data\$p\meta\sample_images.csv --n 100000 --size original
  .\.venv\Scripts\python.exe scripts\dados.py poses
}
Remove-Item Env:DATA_DIR
```

## 3. Correr a pipeline (um comando)

```powershell
.\.venv\Scripts\python.exe scripts\pipeline.py                         # tudo (~15 min com GPU; contorno ~1h30 na 1.ª vez)
.\.venv\Scripts\python.exe scripts\pipeline.py --desde classificar     # sem voltar a correr o detetor
.\.venv\Scripts\python.exe scripts\pipeline.py --dados data\piloto_rua --triangulacao --imagens 6
```

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
**Tudo de uma vez no tmux (Ubuntu/WSL):** `bash scripts/sessao_tmux.sh`.

## 4. Os scripts

| Ficheiro | Comandos | Para quê |
|---|---|---|
| `pipeline.py` | `--desde` · `--ate` · `--dados` | Corre os passos por ordem |
| `dados.py` | `amostra` (indice · selecionar · descarregar) · `sequencia` · `referencias` · `poses` · `modelos` | Tudo o que descarrega: fotos, ruas seguidas, pontos para medir, pontos 3D, modelos |
| `detetar.py` | | Encontra os objetos (OWLv2) |
| `classificar.py` | | Tipo de cada objeto (SigLIP) |
| `sinais.py` | `limites` · `codigo` · `extrair` · `testar` | Número dos limites · código oficial · desenhos do PDF da DGV · medir o código |
| `posicao.py` | `contorno` · `calcular` · `triangulacao` | Pé real (SAM 2) · posição com 1 foto · posição com várias fotos |
| `catalogo.py` | `--limiares` | Catálogo final (GeoJSON + CSV) |
| `avaliar.py` | `erros` · `confianca` · `prompts` | Relatório de erros · escolha dos limiares de confiança · escolha das frases dos modelos |
| `imagens.py` | `final` · `owl` · `foto` · `contorno` · `folha` · `guardar` | Fotos com caixas · detetor em bruto · foto qualquer · caixa vs contorno · rever e guardar exemplos |
| `comum.py`, `mapillary.py` | | Peças partilhadas (não se correm sozinhas) |

Ajuda de cada comando: `python scripts\<ficheiro>.py <comando> -h`.

**Outra pasta de dados:** `--dados data\piloto_rua` no `pipeline.py`, ou `$env:DATA_DIR = "data\piloto_rua"`
antes de qualquer script.

## 5. Pastas

```
config/            taxonomia.yaml (classes, frases, regras, limiares), frases candidatas, vocabulário
scripts/           código (ver acima)
data/meta/         fotos da amostra (sample_images.csv), índice do concelho, poses das câmaras
data/ground_truth/ marcações e pontos Mapillary/OSM — SÓ para medir
data/preds/        saídas da pipeline (deteções, tipos, contornos, posições)
data/catalogo/     catálogo final (GeoJSON + CSV)
data/relatorios/   relatórios de erros, confiança, escolha das frases
data/referencias/  fotos de referência por classe (créditos em fontes*.csv) e desenhos dos sinais da DGV
data/piloto_*/     ruas piloto (fotos seguidas), com a mesma estrutura
```

## 6. Regras (ficha do desafio)

- Fotos Mapillary: **CC BY-SA 4.0** — mostrar sempre autor + link (o catálogo guarda-os).
- **Rostos e matrículas** nunca guardados nem mostrados (a Mapillary já os desfoca; desfocar antes de mostrar recortes).
- Pontos Mapillary e OpenStreetMap (ODbL) servem **só para medir**, nunca entram no catálogo.
- Proibido: Google Street View; código AGPL (YOLOE e restantes modelos Ultralytics, PyMuPDF);
  pesos Mapillary Vistas / Objects365. O YOLO-World (GPL-3.0) foi testado à parte e posto de parte.
- Os PDFs (ficha do desafio, guia da DGV) não estão no repositório: pô-los em `pdfs/`
  (`sinais.py extrair` precisa de `pdfs/SinalizacaoRodoviaria.pdf`).
