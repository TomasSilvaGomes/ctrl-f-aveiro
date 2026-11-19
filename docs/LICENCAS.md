# Registo de licenças — dados, modelos e software

> Última verificação: 16/11/2026. Regra da equipa: **nada entra no código sem uma linha nesta tabela.**
> Tratamos o projeto como *potencialmente comercial* (a Câmara pode adotá-lo), por isso
> licenças "não comerciais" (NC) ficam fora do núcleo da aplicação.

## Legenda

| Estado | Significado |
|---|---|
| 🟢 | Pode entrar no núcleo da aplicação (licença permissiva: Apache 2.0, MIT, BSD) |
| 🟡 | Pode entrar, **com obrigações** que temos de cumprir (atribuição, share-alike, copyleft) |
| 🟠 | Só para benchmark/protótipo, declarado; **não** como dependência final sem confirmação |
| 🔴 | Não usar |
| ⚪ | Por confirmar na fonte oficial antes de usar |

## 1. Dados de entrada

| Recurso | Licença | Estado | Obrigações / notas |
|---|---|---|---|
| Imagens **Mapillary** | CC BY-SA 4.0 (imagens); Termos Mapillary (fev/2024) permitem uso comercial, incluindo treino de algoritmos | 🟡 | Atribuição por imagem: autor, link, licença e indicação de alteração ("anonimizada/recortada"). Recortes e imagens que mostramos/exportamos ficam CC BY-SA. Guardar `creator` e `image_id` em cada observação. |
| API Mapillary (`detections`, `map_features`) | Não encontrámos licença própria explícita | 🟠 | **Só para validação** (métricas). Nunca entram no catálogo. |
| Imagens **KartaView** | CC BY-SA 4.0 (imagens); software MIT/proprietário | 🟡 ⚪ | Mesmas obrigações da Mapillary. Confirmar texto atual em kartaview.org/terms. |
| **Vídeos próprios** | Nossos | 🟢 | R1: anonimizar antes de guardar; vídeo original nunca entra na aplicação. |
| Google Street View / Google Maps imagery | Termos Google Maps Platform | 🔴 | Proíbem extrair/guardar conteúdo e criar dados a partir dele. |

## 2. Dados de validação e mapa

| Recurso | Licença | Estado | Obrigações / notas |
|---|---|---|---|
| Dados **OpenStreetMap** (Overpass) | ODbL 1.0 | 🟡 | Usar **só para comparar** (métricas). Não fundir pontos OSM no nosso catálogo — isso tornava o catálogo uma base de dados derivada sujeita a share-alike ODbL. |
| Tiles `tile.openstreetmap.org` | Tile Usage Policy da OSMF | 🟡 | "© OpenStreetMap contributors" visível; User-Agent/Referer corretos; sem pré-download em massa. OK para desenvolvimento e demo; em produção usar outro fornecedor ou servidor próprio. |
| Ortofotos / LiDAR **DGT** | ⚪ | ⚪ | Confirmar licença na página de dados abertos da DGT antes de usar como camada de fundo. |
| Datasets abertos Lisboa / Águeda | CC BY 4.0 (Águeda) | 🟢 | Apenas como referência de formato de exportação. |

## 3. Modelos

| Modelo | Licença | Estado | Notas |
|---|---|---|---|
| **OWLv2** (`google/owlv2-*`) | Apache 2.0 | 🟢 | Deteção zero-shot por texto. |
| **Grounding DINO tiny** (`IDEA-Research/grounding-dino-tiny`) | Apache 2.0 | 🟢 | Deteção zero-shot por texto. (Grounding DINO 1.5/1.6 são só API → excluídos.) |
| **YOLOE** (THU-MIG, via Ultralytics) | AGPL-3.0 | 🟡 | Copyleft forte, incluindo uso em rede: se entrar, **todo o nosso código passa a AGPL-3.0** e o código-fonte tem de estar disponível a quem usa a aplicação. Compatível com "código aberto", mas é uma decisão da equipa. |
| **Mask2Former Swin-L Mapillary Vistas** | Código MIT; pesos HF marcados "other"; treinados no Vistas (CC BY-NC-SA) | 🟠 | Não é claro se os pesos herdam o "NC". Testado a 19/11 como alternativa ao SAM 2 no contorno (`posicao.py contorno --sam mask2former`): mesma qualidade (iluminação até 5 m 63% vs 63%) e tempo igual ao SAM 2 small → **não compensa**; fica só como comparação. |
| **DINOv2** (`facebook/dinov2-base`) | Apache 2.0 (desde ago/2023, código e pesos) | 🟢 | Testado a 17/11 nos desenhos dos sinais (23–38% contra 78% do SigLIP) e **retirado** a 19/11. |
| DINOv3 | Licença própria Meta ("DINOv3 License"), com formulário | 🔴 ⚪ | Evitar; DINOv2 chega. |
| **CLIP** (OpenAI) / **SigLIP** (Google) | MIT / Apache 2.0 | 🟢 ⚪ | Classificação zero-shot de subtipo. Confirmar no model card do checkpoint escolhido. |
| **EgoBlur Gen2** (rostos + matrículas) | Apache 2.0 — código (GitHub) e pesos. O "EgoBlur Model License Agreement" que se aceita no download (projectaria.com/tools/egoblur/license) é o texto da Apache 2.0, sem termos extra; a FAQ diz "Both the Gen 2 face and license plate models are licensed under Apache 2.0" (confirmado a 19/11) | 🟢 | Integrado em `comum.Desfocar` e no `dados.py video` (pesos `weights/ego_blur_*_gen2.jit`, fora do git). Se se redistribuírem os pesos: incluir a licença Apache 2.0; não usar marcas da Meta. |
| **SAM 2.1** (`facebook/sam2.1-hiera-*`) | Apache 2.0 | 🟢 | Contorno dos objetos (`posicao.py contorno`); small ≈ large na qualidade e 2,3× mais rápido. |
| **Depth Anything V2 Small** | Apache 2.0 | 🟢 | Só o Small. **Base/Large/Giant são CC BY-NC 4.0 → 🔴.** |
| EasyOCR / PaddleOCR | Apache 2.0 | 🟢 | **Retirado a 17/11**: o SigLIP lê o número dos limites melhor (97% contra 31%). |
| Mapillary Traffic Sign Dataset (MTSD) | Uso académico / não comercial | 🔴 | Não precisamos (sem treino). |
| Detetores pré-treinados em **Objects365** (ex.: checkpoints D-FINE/RT-DETR "obj365") | Código Apache 2.0; licença do dataset Objects365 por confirmar | 🟠 ⚪ | Têm classes "Street Lights", "Trash bin Can", "Stop Sign", "Speed Limit Sign". Mesma dúvida dos pesos Vistas. |
| Pictogramas oficiais de sinais PT (Wikimedia Commons) | Variável por ficheiro (muitos em domínio público) | ⚪ | Galeria de referência para classificar sinais sem anotar. Confirmar licença de cada ficheiro. |
| Mapillary Vistas (dataset) | CC BY-NC-SA | 🔴 | Não descarregamos nem treinamos com ele. |

## 4. Software

| Software | Licença | Estado | Notas |
|---|---|---|---|
| PyTorch | BSD-3 | 🟢 | RTX 5060 Ti exige build CUDA 12.8 (`cu128`). |
| **OpenSfM** (reconstrução 3D da Mapillary) | BSD-2 no repositório original `mapillary/OpenSfM` | 🟢 | Usado desde 19/11 em `dados.py poses --opensfm` (imagem Docker feita com o `Dockerfile.ubuntu24` do repositório original; dependências de sistema Ceres/OpenCV/Eigen com licenças BSD/Apache/MPL; `fpdf2` LGPL só nos relatórios do próprio OpenSfM, que não usamos). Usar só o original BSD: as cópias OpenDroneMap/WebODM têm alterações **AGPL-3.0** → 🔴. |
| Hugging Face `transformers` | Apache 2.0 | 🟢 | |
| FastAPI | MIT | 🟢 | |
| Leaflet | BSD-2 | 🟢 | |
| PostgreSQL / PostGIS | PostgreSQL License / GPL-2.0 | 🟢 | PostGIS corre como serviço separado; não impõe licença ao nosso código. |
| Shapely, mercantile, requests, mapbox-vector-tile | BSD / MIT / Apache 2.0 | 🟢 | |

## 5. Decisões que daqui resultam

1. **Não treinamos modelos** (zero-shot + k-NN). Isto evita a questão, juridicamente em aberto, de saber se um modelo treinado com imagens CC BY-SA ou NC é obra derivada.
2. **A galeria de recortes do k-NN** é material CC BY-SA adaptado: guardar ao lado um `ATTRIBUTION.csv` (image_id, autor, link, licença).
3. **Catálogo exportado** (GeoJSON/CSV) leva colunas `source`, `image_id`, `author`, `license`.
4. **Licença do nosso repositório**: MIT/Apache 2.0 se o núcleo for só 🟢; **AGPL-3.0** se adotarmos YOLOE. Decidir após o benchmark.
5. **Perguntas para os mentores (18/11)**: (a) pesos treinados em Vistas são aceitáveis? (b) a solução é considerada de uso comercial pela Câmara?

## Fontes

- Mapillary — Termos de Utilização: https://www.mapillary.com/terms
- Mapillary — Licença CC BY-SA: https://help.mapillary.com/hc/en-us/articles/115001770409-CC-BY-SA-license-for-open-data
- Mapillary Vistas: https://www.mapillary.com/dataset/vistas
- KartaView (licença): https://en.wikipedia.org/wiki/KartaView · https://lists.openstreetmap.org/pipermail/tagging/2021-June/061819.html
- OSMF Tile Usage Policy: https://operations.osmfoundation.org/policies/tiles/
- OWLv2: https://huggingface.co/google/owlv2-base-patch16
- Grounding DINO tiny: https://huggingface.co/IDEA-Research/grounding-dino-tiny
- YOLOE: https://roboflow.com/model-licenses/yoloe · https://huggingface.co/jameslahm/yoloe/discussions/3/files · https://ultralytics.com/license
- Mask2Former (MIT): https://awesome.ecosyste.ms/projects/github.com%2Ffacebookresearch%2Fmask2former
- Mask2Former Vistas (HF): https://huggingface.co/facebook/mask2former-swin-large-mapillary-vistas-semantic
- DINOv2 Apache 2.0: https://ai.meta.com/blog/dinov2-facet-computer-vision-fairness-evaluation/
- Depth Anything V2: https://docs.lightly.ai/train/stable/depth_estimation.html
- EgoBlur: https://huggingface.co/projectaria/EgoBlur · https://projectaria.com/tools/egoblur · https://www.projectaria.com/tools/egoblur/license/ · https://github.com/facebookresearch/EgoBlur
