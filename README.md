# Ctrl+F Aveiro — catálogo de mobiliário urbano

Hackathon Aveiro Tech City 2026, Desafio #2. A partir de fotos abertas da Mapillary, o
sistema **encontra, classifica e põe no mapa** sinais de trânsito, iluminação pública e
contentores de resíduos de Aveiro, e exporta um catálogo (GeoJSON/CSV).

| Categoria | Classes |
|---|---|
| Sinalização vertical | STOP, cedência, limites de velocidade (com o número), sentido proibido, … |
| Iluminação pública | Poste de luz, Candeeiro |
| Resíduos | Ilha de reciclagem (ecoponto), Contentor indiferenciado, Contentor subterrâneo |

Modelos (todos Apache 2.0): **OWLv2** encontra os objetos; **SigLIP** dá o tipo, lê o
número dos limites e descarta o que não é mobiliário. Classes, frases e regras estão em
[config/taxonomia.yaml](config/taxonomia.yaml).

## 1. Instalar (Windows, PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env        # e colar o teu token da Mapillary no .env
.\.venv\Scripts\python.exe scripts\descarregar_modelos.py
```

O `.env` é pessoal e **nunca** vai para o repositório.

## 2. Descarregar as fotos (não estão no repositório: ~2,9 GB)

```powershell
$env:PYTHONIOENCODING = "utf-8"
# amostra principal (3 050 fotos, ~2,2 GB)
.\.venv\Scripts\python.exe scripts\download_sample.py --ids-from data\meta\sample_images.csv --n 100000 --size original --workers 10
# ruas piloto (fotos seguidas), cada uma na sua pasta
foreach ($p in "piloto_rua", "piloto_sequencia", "piloto_frontal") {
  $env:DATA_DIR = "data\$p"
  .\.venv\Scripts\python.exe scripts\download_sample.py --ids-from data\$p\meta\sample_images.csv --n 100000 --size original
}
Remove-Item Env:DATA_DIR
.\.venv\Scripts\python.exe scripts\fetch_ground_truth.py     # referências Mapillary/OSM (só para medir)
```

## 3. Pipeline (por esta ordem)

```powershell
.\.venv\Scripts\python.exe scripts\run_detector.py        # 1. encontra objetos (OWLv2) ~12 min com GPU
.\.venv\Scripts\python.exe scripts\classify_subtype.py    # 2. tipo de cada objeto (SigLIP) ~25 min
.\.venv\Scripts\python.exe scripts\ler_limites.py         # 3. número dos limites de velocidade
.\.venv\Scripts\python.exe scripts\geolocate.py           # 4. posição no mapa (1 foto)
.\.venv\Scripts\python.exe scripts\build_catalog.py       # 5. catálogo GeoJSON + CSV
.\.venv\Scripts\python.exe scripts\analisa_erros.py       # 6. relatório de erros (data\relatorios)
.\.venv\Scripts\python.exe scripts\visualize.py --n 12    # imagens com as caixas (data\viz)
```

Os resultados já calculados estão em `data/preds`, `data/catalogo` e `data/relatorios`
(dá para começar no passo 2 ou 4 sem correr o detetor).

**Outra pasta de dados** (ruas piloto): `$env:DATA_DIR = "data\piloto_rua"` antes dos comandos.
**Fotos seguidas de uma rua:** `scripts\captura_sequencia.py --lat .. --lon .. --n 100 [--tipo frontal]`,
depois a pipeline + `scripts\triangulacao.py` (posição com várias fotos).
**Tudo de uma vez no tmux (Ubuntu/WSL):** `bash scripts/sessao_tmux.sh`.

## 4. Ferramentas de avaliação

| Script | Para quê |
|---|---|
| `analisa_erros.py` | Deteção, tipo de sinal, limites, posição, repetidos |
| `triangulacao.py` | Posição com várias fotos e repetibilidade entre passagens |
| `escolhe_prompts.py` | Escolher as frases que os modelos entendem melhor (CCAS + medição) |
| `audita_classificacao.py` | Verificação visual do tipo (folhas de imagens + CSV) |
| `folha_catalogo.py` / `guarda_referencias.py` | Ver exemplos do catálogo / guardar fotos de referência |
| `testa_semelhanca.py`, `testa_vocabulario.py` | Testes (código exato dos sinais; vocabulário dos modelos) |
| `lista_pontos_mapillary.py` | Que "points" a Mapillary tem num município |

## 5. Aplicação web (protótipo)

Instala as dependências do projeto e inicia o catálogo interativo:

```powershell
.\.venv\Scripts\python.exe -m pip install "streamlit>=1.49" "folium>=0.15" "streamlit-folium>=0.25"
$env:CTRL_F_ADMIN_PASSWORD = "escolhe-uma-palavra-passe"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

O catálogo pode ser consultado sem autenticação. A área de administração só fica
disponível quando `CTRL_F_ADMIN_PASSWORD` está configurada; permite rever deteções,
corrigir o tipo, a designação e as coordenadas, e confirmar ou rejeitar elementos.
As correções atualizam `data/catalogo/catalogo.geojson` e `catalogo.csv`.
Os visitantes podem filtrar e exportar o catálogo visível em GeoJSON ou CSV.
Este acesso simples destina-se ao protótipo local, não a uma publicação em produção.
Para outra cidade, define também `CTRL_F_MUNICIPALITY` e `CTRL_F_CATALOG_PATH`
(caminho para o respetivo ficheiro GeoJSON); as correções serão gravadas junto desse ficheiro.

## 6. Regras (ficha do desafio)

- Fotos Mapillary: **CC BY-SA 4.0** — mostrar sempre autor + link (o catálogo guarda-os).
- **Rostos e matrículas** nunca guardados nem mostrados (desfocar antes de mostrar recortes).
- Pontos Mapillary e OpenStreetMap (ODbL) servem **só para medir**, nunca entram no catálogo.
- Proibido: Google Street View; YOLOE (AGPL); PyMuPDF (AGPL).
- Fotos de referência: créditos em `data/referencias/**/fontes*.csv`.
