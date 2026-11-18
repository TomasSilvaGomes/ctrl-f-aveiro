# Análise de erros

3 050 fotos analisadas · modelo owlv2.

As referências (Mapillary e OpenStreetMap) servem só para medir. Também têm erros e não estão completas: use estes números sobretudo para comparar versões do sistema.

## 1. Encontra os objetos?

| Categoria | Objetos nas fotos | Encontrados | Taxa |
|---|---|---|---|
| Sinais | 2 590 | 2 399 | 93% |
| Iluminação | 7 140 | 3 411 | 48% |
| Contentores | 1 452 | 1 020 | 70% |

*Objetos nas fotos = marcados pela Mapillary (com pelo menos 24 píxeis). Contentores: só os com >= 0.9 m de largura (a Mapillary marca também papeleiras, que a ficha não pede).*


## 2. Acerta no tipo de sinal?

| Tipo de sinal | Sinais | Acerto |
|---|---|---|
| Passagem para peões | 558 | 100% |
| Sentido proibido | 414 | 100% |
| Obrigação | 292 | 83% |
| Trânsito de sentido único | 169 | 88% |
| Perigo | 161 | 78% |
| Cedência de passagem | 160 | 99% |
| Paragem e estacionamento proibidos | 159 | 99% |
| Outros | 100 | 21% |
| STOP | 99 | 98% |
| Proibido virar / inversão de marcha | 85 | 93% |
| Direção / indicação | 78 | 21% |
| Limite de velocidade | 51 | 100% |
| Estacionamento | 46 | 98% |
| Estacionamento proibido | 27 | 52% |
| **Total** | **2 399** | **88%** |

*Sinais = sinais encontrados pelo sistema que a Mapillary também identificou. Acerto = o sistema deu o mesmo tipo.*

Trocas mais frequentes:

| Era | O sistema disse | Vezes |
|---|---|---|
| Obrigação | Outros | 45 |
| Outros | Passagem para peões | 35 |
| Direção / indicação | Outros | 25 |
| Trânsito de sentido único | Passagem para peões | 20 |
| Direção / indicação | Estacionamento proibido | 17 |


## 3. Lê o número do limite de velocidade?

O número é lido pelo SigLIP num "print" do sinal (vista normal, ampliada 4×), no passo ler_limites.py. Duas verificações:

**3.1 Contra as marcações da Mapillary nas fotos**

| Tamanho do sinal na foto | Sinais | Número certo | Número errado | Não leu |
|---|---|---|---|---|
| Pequeno (< 40 px) | 23 | 100% | 0% | 0% |
| Médio (40–70 px) | 16 | 100% | 0% | 0% |
| Grande (≥ 70 px) | 12 | 100% | 0% | 0% |
| **Total** | 51 | 100% | 0% | 0% |

*Tamanho = largura do sinal em píxeis. Número certo = igual ao da referência.*

**3.2 Contra os pontos de sinais da Mapillary no mapa** (até 12 m)

| Tamanho do sinal na foto | Sinais | Número certo | Número errado | Não leu |
|---|---|---|---|---|
| Pequeno (< 40 px) | 30 | 93% | 7% | 0% |
| Médio (40–70 px) | 16 | 94% | 6% | 0% |
| Grande (≥ 70 px) | 16 | 100% | 0% | 0% |
| **Total** | 62 | 95% | 5% | 0% |

*Tamanho = largura do sinal em píxeis. Número certo = igual ao da referência.*

Números de referência nesta amostra: 20 km/h: 5, 30 km/h: 49, 40 km/h: 4, 50 km/h: 2, 70 km/h: 1, 120 km/h: 1. A maioria das ruas da cidade é de 30 km/h, por isso há poucos exemplos de outros números.


## 4. Acerta no sítio do mapa?

**Como se mede** (para cada objeto):

```text
  P = ponto calculado pelo sistema
  R = ponto de referência mais próximo (Mapillary ou OpenStreetMap)
  erro = distância entre P e R

        . - - - - .
      /     R       \      círculo de 5 m à volta de R
     |    P1          |    P1 dentro  ->  ACERTA (erro até 5 m)
      \             /
        ' - - - - '      P2 fora    ->  FALHA  (erro maior que 5 m)
                     P2
```

### 4.1 Resumo

| Categoria | Objetos medidos | Acertam até 5 m | Acertam até 10 m | Erro típico |
|---|---|---|---|---|
| Sinais | 5 238 | 70% | 89% | 2,9 m |
| Iluminação | 7 597 | 44% | 79% | 5,8 m |
| Contentores | 1 864 | 74% | 89% | 2,2 m |

*Acertam até 5 m = % de objetos a 5 metros ou menos do sítio certo (os restantes falham). Erro típico = metade dos objetos erra menos do que isto.*

### 4.2 Taxa de acerto conforme a margem aceite

| Margem aceite | Sinais | Iluminação | Contentores |
|---|---|---|---|
| até 2 m | 38% | 17% | 47% |
| até 5 m | 70% | 44% | 74% |
| até 10 m | 89% | 79% | 89% |
| até 20 m | 99% | 99% | 99% |

*Cada valor = % de objetos que ficam dentro dessa margem. A taxa de falha é o resto até 100%.*

### 4.3 Perto ou longe da câmara

| Categoria | Objeto perto (até 10 m) | Objeto longe (mais de 10 m) |
|---|---|---|
| Sinais | 79% | 64% |
| Iluminação | 52% | 41% |
| Contentores | 86% | 67% |

*Valores = % que acertam até 5 m. Objetos perto da câmara ficam mais bem posicionados.*

### 4.4 Foto 360° ou foto frontal

| Categoria | Foto 360° | Foto frontal |
|---|---|---|
| Sinais | 73% | 52% |
| Iluminação | 49% | 30% |
| Contentores | 77% | 34% |

*Valores = % que acertam até 5 m. As fotos 360° (recentes) posicionam melhor.*

### 4.5 O resultado é mérito do sistema?

| Categoria | Erro típico do sistema | Erro típico ao acaso | Melhor que o acaso? |
|---|---|---|---|
| Sinais | 2,9 m | 7,4 m | sim |
| Iluminação | 5,8 m | 8,7 m | só um pouco |
| Contentores | 2,2 m | 10,3 m | sim |

*Ao acaso = o mesmo objeto à mesma distância da câmara, mas numa direção sorteada. Se o sistema não for bem melhor do que isto, o número é sorte.*

### 4.6 O que falha: a distância ou a direção?

| Categoria | Erro na distância à câmara | Erro na direção |
|---|---|---|
| Sinais | 2,0 m | 0,5 m |
| Iluminação | 3,5 m | 1,7 m |
| Contentores | 1,6 m | 0,6 m |

*Valores típicos. A direção está bem medida; o que falha é a distância. A triangulação (usar várias fotos) corrige isto.*

Não medidos, por não haver referência a menos de 20 m: sinais 71, iluminação 102, contentores 56.


## 5. Há registos repetidos no catálogo?

| Categoria | Objetos repetidos | Registos por objeto |
|---|---|---|
| Sinais | 48% | 1,7 |
| Iluminação | 38% | 1,5 |
| Contentores | 28% | 1,4 |

*Objetos repetidos = % de objetos reais com mais de um registo no catálogo. Registos por objeto: o ideal é 1,0.*
