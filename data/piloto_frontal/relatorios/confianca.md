# A partir de que confiança se aceita um objeto?

Fotos com marcações da Mapillary: 51. Encontra = % dos objetos marcados pela Mapillary que o sistema apanha com confiança >= limiar. Precisão = % das deteções >= limiar que batem com uma marcação (valor mínimo: a Mapillary não marca tudo). Deteções na amostra = quantas deteções (todas as fotos) passam o limiar.

## Sinais

Marcações Mapillary contadas: 93 · deteções nas fotos marcadas: 220

| Limiar | Encontra | Precisão | F1 | Deteções na amostra |
|---|---|---|---|---|
| 0.10 | 95% | 41% | 0.57 | 272 |
| 0.15 | 95% | 41% | 0.57 | 270 |
| 0.20 | 95% | 42% | 0.58 | 265 |
| 0.25 | 94% | 43% | 0.59 | 259 |
| 0.30 | 92% | 43% | 0.59 | 249 |
| 0.35 | 90% | 45% | 0.60 | 235 |
| 0.40 **← sugerido** | 88% | 47% | 0.61 | 220 |
| 0.45 | 82% | 48% | 0.61 | 197 |
| 0.50 | 74% | 50% | 0.59 | 166 |
| 0.55 | 72% | 52% | 0.60 | 150 |
| 0.60 | 66% | 54% | 0.59 | 134 |
| 0.65 | 58% | 51% | 0.55 | 119 |
| 0.70 | 53% | 51% | 0.52 | 108 |
| 0.75 | 46% | 53% | 0.49 | 93 |
| 0.80 | 45% | 58% | 0.51 | 83 |
| 0.85 | 34% | 60% | 0.44 | 61 |
| 0.90 | 22% | 65% | 0.32 | 36 |
| 0.95 | 12% | 61% | 0.20 | 22 |

## Iluminação

Marcações Mapillary contadas: 14 · deteções nas fotos marcadas: 321

| Limiar | Encontra | Precisão | F1 | Deteções na amostra |
|---|---|---|---|---|
| 0.10 | 86% | 4% | 0.08 | 402 |
| 0.15 | 86% | 4% | 0.08 | 402 |
| 0.20 | 86% | 4% | 0.08 | 401 |
| 0.25 | 86% | 4% | 0.08 | 392 |
| 0.30 | 86% | 4% | 0.08 | 369 |
| 0.35 | 86% | 5% | 0.09 | 333 |
| 0.40 | 86% | 6% | 0.10 | 298 |
| 0.45 | 86% | 6% | 0.12 | 248 |
| 0.50 | 79% | 6% | 0.12 | 219 |
| 0.55 | 71% | 7% | 0.12 | 190 |
| 0.60 | 71% | 8% | 0.14 | 160 |
| 0.65 | 71% | 10% | 0.18 | 125 |
| 0.70 | 64% | 11% | 0.18 | 104 |
| 0.75 | 64% | 13% | 0.22 | 81 |
| 0.80 | 50% | 13% | 0.21 | 63 |
| 0.85 | 43% | 16% | 0.23 | 45 |
| 0.90 **← sugerido** | 36% | 24% | 0.29 | 24 |
| 0.95 | 21% | 25% | 0.23 | 12 |

## Contentores

Marcações Mapillary contadas: 0 · deteções nas fotos marcadas: 13

| Limiar | Encontra | Precisão | F1 | Deteções na amostra |
|---|---|---|---|---|
| 0.10 **← sugerido** | nan% | 0% | nan | 14 |
| 0.15 | nan% | 0% | nan | 14 |
| 0.20 | nan% | 0% | nan | 14 |
| 0.25 | nan% | 0% | nan | 14 |
| 0.30 | nan% | 0% | nan | 14 |
| 0.35 | nan% | 0% | nan | 13 |
| 0.40 | nan% | 0% | nan | 10 |
| 0.45 | nan% | 0% | nan | 10 |
| 0.50 | nan% | 0% | nan | 10 |
| 0.55 | nan% | 0% | nan | 9 |
| 0.60 | nan% | 0% | nan | 9 |
| 0.65 | nan% | 0% | nan | 7 |
| 0.70 | nan% | 0% | nan | 6 |
| 0.75 | nan% | 0% | nan | 5 |
| 0.80 | nan% | 0% | nan | 5 |
| 0.85 | nan% | 0% | nan | 5 |
| 0.90 | nan% | 0% | nan | 2 |
| 0.95 | nan% | 0% | nan | 1 |

## Limiar sugerido (F1 máximo)

| Categoria | Limiar | Encontra | Precisão |
|---|---|---|---|
| Sinais | 0.40 | 88% | 47% |
| Iluminação | 0.90 | 36% | 24% |
| Contentores | 0.10 | nan% | 0% |

# Por objeto no catálogo (todas as fotos juntas)

Objetos de referência = pontos da Mapillary e do OSM no mapa; quando as duas fontes marcam o mesmo objeto (pontos a <= 2 m) conta uma vez. Encontra = % desses objetos com um elemento do catálogo a <= 8 m. Certos = % dos elementos com um objeto de referência a <= 8 m (valor mínimo: as referências não têm tudo). Registos por objeto: 1,0 = sem repetidos.

## Sinais

| Limiar | Elementos | Encontra | Certos | Registos por objeto | F1 |
|---|---|---|---|---|---|
| 0.00 **← melhor F1** | 154 | 49% | 66% | 1.66 | 0.56 |
| 0.10 | 154 | 49% | 66% | 1.66 | 0.56 |
| 0.20 | 149 | 48% | 66% | 1.63 | 0.56 |
| 0.30 | 138 | 46% | 67% | 1.61 | 0.54 |
| 0.35 | 128 | 45% | 66% | 1.52 | 0.54 |
| 0.40 | 119 | 43% | 70% | 1.51 | 0.54 |
| 0.45 | 112 | 43% | 69% | 1.40 | 0.53 |
| 0.50 | 94 | 38% | 70% | 1.35 | 0.50 |
| 0.55 | 86 | 36% | 69% | 1.31 | 0.47 |
| 0.60 | 76 | 34% | 68% | 1.27 | 0.45 |
| 0.65 | 67 | 31% | 70% | 1.21 | 0.43 |
| 0.70 | 60 | 29% | 70% | 1.20 | 0.41 |
| 0.75 | 50 | 28% | 76% | 1.15 | 0.41 |
| 0.80 | 46 | 26% | 76% | 1.21 | 0.39 |
| 0.85 | 37 | 23% | 81% | 1.15 | 0.36 |

## Iluminação

| Limiar | Elementos | Encontra | Certos | Registos por objeto | F1 |
|---|---|---|---|---|---|
| 0.00 | 160 | 56% | 54% | 1.58 | 0.55 |
| 0.10 | 160 | 56% | 54% | 1.58 | 0.55 |
| 0.20 | 160 | 56% | 54% | 1.58 | 0.55 |
| 0.30 **← melhor F1** | 157 | 56% | 55% | 1.58 | 0.56 |
| 0.35 | 153 | 56% | 55% | 1.53 | 0.55 |
| 0.40 | 141 | 54% | 55% | 1.50 | 0.55 |
| 0.45 | 126 | 52% | 56% | 1.45 | 0.54 |
| 0.50 | 118 | 49% | 56% | 1.47 | 0.52 |
| 0.55 | 105 | 44% | 55% | 1.45 | 0.49 |
| 0.60 | 97 | 44% | 56% | 1.38 | 0.49 |
| 0.65 | 82 | 39% | 55% | 1.29 | 0.46 |
| 0.70 | 70 | 35% | 56% | 1.26 | 0.43 |
| 0.75 | 59 | 32% | 54% | 1.14 | 0.40 |
| 0.80 | 44 | 28% | 57% | 1.04 | 0.37 |
| 0.85 | 33 | 22% | 55% | 1.00 | 0.31 |

## Contentores

| Limiar | Elementos | Encontra | Certos | Registos por objeto | F1 |
|---|---|---|---|---|---|
| 0.00 | 6 | 7% | 50% | 1.00 | 0.12 |
| 0.10 | 6 | 7% | 50% | 1.00 | 0.12 |
| 0.20 | 6 | 7% | 50% | 1.00 | 0.12 |
| 0.30 | 6 | 7% | 50% | 1.00 | 0.12 |
| 0.35 | 6 | 7% | 50% | 1.00 | 0.12 |
| 0.40 | 6 | 7% | 50% | 1.00 | 0.12 |
| 0.45 | 6 | 7% | 50% | 1.00 | 0.12 |
| 0.50 | 6 | 7% | 50% | 1.00 | 0.12 |
| 0.55 | 5 | 7% | 60% | 1.00 | 0.12 |
| 0.60 | 5 | 7% | 60% | 1.00 | 0.12 |
| 0.65 | 4 | 7% | 75% | 1.00 | 0.13 |
| 0.70 | 4 | 7% | 75% | 1.00 | 0.13 |
| 0.75 **← melhor F1** | 3 | 7% | 100% | 1.00 | 0.13 |
| 0.80 | 3 | 7% | 100% | 1.00 | 0.13 |
| 0.85 | 3 | 7% | 100% | 1.00 | 0.13 |

Melhor limiar por objeto: Sinais 0.00 · Iluminação 0.30 · Contentores 0.75

Gráfico: confianca.svg · exemplos por faixa: confianca_<categoria>.jpg