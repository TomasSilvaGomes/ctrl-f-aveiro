# Ctrl+F Aveiro — catálogo de mobiliário urbano

Hackathon Aveiro Tech City 2026, Desafio #2. A partir de fotos abertas da Mapillary, o
sistema **encontra, classifica e põe no mapa** sinais de trânsito, iluminação pública e
contentores de resíduos de Aveiro, e exporta um catálogo (GeoJSON/CSV).

| Categoria | Classes |
|---|---|
| Sinalização vertical | STOP, cedência, limites de velocidade (com o número), sentido proibido, … |
| Iluminação pública | Poste de luz, Candeeiro |
| Resíduos | Ilha de reciclagem (ecoponto), Contentor indiferenciado, Contentor subterrâneo |

O **OWLv2** encontra os objetos, o **SigLIP** classifica-os e lê os limites de
velocidade, e o **SAM 2** delimita o contorno de cada objeto na fotografia.
Classes, frases e regras estão em [config/taxonomia.yaml](config/taxonomia.yaml).

## 1. Instalar (Windows, PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\descarregar_modelos.py
```

## 2. Descarregar as fotos (não estão no repositório: ~2,9 GB)

Para recolher as fotografias usadas na pipeline inicial, copia `.env.example` para
`.env` e configura o teu token do Mapillary aí. O `.env` é pessoal e fica fora do
repositório.

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

Instala as dependências do projeto na secção 1 e inicia o catálogo interativo:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.maxUploadSize 25
```

O catálogo e o envio de fotografias estão disponíveis sem conta.
O botão **Administração** abre a fila de **Pedidos** e o separador **Registo**.
Aprovar publica a fotografia e as respetivas deteções no mapa. O registo guarda a
receção, a decisão, a hora e a nota. O estado dos elementos é visível em Administração.
Quem tiver acesso à app pode abrir este modo e rever pedidos ou corrigir elementos.

A barra de pesquisa sugere localidades portuguesas à medida que escreves e centra o mapa
na seleção. O índice inclui 16 610 entradas do [GeoNames](https://www.geonames.org/export/),
sob a licença CC BY 4.0, e pode ser atualizado com `scripts/preparar_localidades.py PT.zip`.
Os filtros de categoria e subtipo ficam junto à pesquisa. Enquadrar, exportar (GeoJSON
ou CSV) e administração ficam por cima dos controlos de zoom.
O mapa usa uma base simples com cor do OpenFreeMap. A partir do zoom 13, os elementos
aparecem como ícones. Passar o rato mostra o nome. Tocar numa zona com ícones
sobrepostos abre uma grelha de seleção.
[OpenFreeMap](https://openfreemap.org/)

Em **Adicionar fotos**, qualquer visitante pode enviar JPEG, PNG, WebP e HEIC
até 25 MB por ficheiro. A app lê o GPS EXIF ou pede as coordenadas quando faltam.
Ao carregar em **Analisar e enviar**, o OWLv2 encontra os objetos, o SigLIP
classifica cada recorte e o SAM 2 gera a máscara a partir da caixa de deteção.
As deteções partilham a posição da captura. Os ficheiros enviados e as importações
do Mapillary entram na fila de pedidos. Uma fotografia aprovada sem deteções recebe
um marcador de fotografia.

Antes de publicar ficheiros enviados, a app cobre as regiões identificadas como rostos
ou matrículas e guarda apenas os JPEG resultantes, sem os metadados do original.
Se os pesos do EgoBlur Gen1 estiverem instalados, a app usa-os. Guarda
`ego_blur_face_gen1.jit` e `ego_blur_lp_gen1.jit` em `weights/egoblur/` ou indica outra pasta
com `CTRL_F_EGOBLUR_DIR`. Os nomes antigos `ego_blur_face.jit` e `ego_blur_lp.jit` também
são aceites. [Download do EgoBlur](https://www.projectaria.com/tools/egoblur/)

Quando faltam os pesos do EgoBlur, a app usa o YuNet para os rostos
e o detetor de matrículas do Open Image Models. No primeiro uso, descarrega os
pesos públicos para `weights/privacy`. Os modelos correm localmente.
[YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
[Open Image Models](https://github.com/ankandrew/open-image-models#plate-detection)

A ficha mostra primeiro a fotografia e permite escolher outra se houver várias vistas.
Nos pontos do catálogo inicial, a app usa a deteção já calculada pelo OWLv2 e
classificada pelo SigLIP. Ao abrir uma vista com a fotografia disponível em
`data/images` ou nos recortes de referência, o SAM 2 gera a máscara em segundo plano.
A fotografia e a máscara ficam em `data/catalogo/previews/sam2` para as próximas
aberturas. A máscara fica visível na ficha. **Imagem completa** abre a cena em ecrã
inteiro com as máscaras das deteções dessa fotografia. O índice de deteções originais
vem de `data/preds/owlv2_classified.jsonl`.

Quando falta a fotografia local de uma deteção, a app tenta obter essa imagem pela
API do Mapillary e segmentá-la com a caixa já guardada. Configura `MAPILLARY_TOKEN`
no ficheiro `.env` do servidor para ativar esta opção. Em **Adicionar fotos →
Mapillary**, cola o link ou ID de outra fotografia para a analisar com OWLv2, SigLIP e
SAM 2. As deteções novas usam a posição da captura e a ficha inclui o crédito da
fotografia. Esta opção usa as fotografias publicadas com o desfoque do Mapillary.

Para instalar o SAM 2.1 Hiera Tiny em `weights/sam2` e preparar em lote as máscaras
das fotografias locais, corre o seguinte comando. A variável `CTRL_F_SAM2_DIR`
permite indicar outra pasta para os pesos.

```powershell
.\.venv\Scripts\python.exe scripts/preparar_previews.py --descarregar-modelo
```

As contribuições novas guardam uma pré-visualização reduzida e anonimizada da cena.
Os créditos da fotografia estão disponíveis no ícone ©. Os marcadores dos sinais
usam pictogramas do subtipo classificado. O limite de velocidade usa o número
identificado, quando disponível. O subtipo «outro» usa um sinal genérico.

As contribuições e as suas correções ficam em `data/catalogo/contribuicoes.sqlite3`.
As correções do catálogo inicial são guardadas no GeoJSON e no CSV.
Para outra cidade, define também `CTRL_F_MUNICIPALITY` e `CTRL_F_CATALOG_PATH`
(caminho para o respetivo ficheiro GeoJSON) e usa uma pasta por catálogo.

## 6. Regras (ficha do desafio)

- Fotos Mapillary: **CC BY-SA 4.0** — mostrar sempre autor + link (o catálogo guarda-os).
- **Rostos e matrículas** nunca guardados nem mostrados (desfocar antes de mostrar recortes).
- Pontos Mapillary e OpenStreetMap (ODbL) servem **só para medir**, nunca entram no catálogo.
- Proibido: Google Street View; YOLOE (AGPL); PyMuPDF (AGPL).
- Fotos de referência: créditos em `data/referencias/**/fontes*.csv`.
