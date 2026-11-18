# Análise de erros

100 fotos analisadas · modelo owlv2.

As referências (Mapillary e OpenStreetMap) servem só para medir. Também têm erros e não estão completas: use estes números sobretudo para comparar versões do sistema.

## 1. Encontra os objetos?

| Categoria | Objetos nas fotos | Encontrados | Taxa |
|---|---|---|---|
| Sinais | 88 | 87 | 99% |
| Iluminação | 239 | 61 | 26% |
| Contentores | 112 | 75 | 67% |

*Objetos nas fotos = marcados pela Mapillary (com pelo menos 24 píxeis). Contentores: só os com >= 0.9 m de largura (a Mapillary marca também papeleiras, que a ficha não pede).*


## 2. Acerta no tipo de sinal?

| Tipo de sinal | Sinais | Acerto |
|---|---|---|
| Passagem para peões | 40 | 100% |
| Paragem e estacionamento proibidos | 23 | 100% |
| Outros | 18 | 6% |
| Perigo | 5 | 60% |
| Estacionamento | 1 | 100% |
| **Total** | **87** | **78%** |

*Sinais = sinais encontrados pelo sistema que a Mapillary também identificou. Acerto = o sistema deu o mesmo tipo.*

Trocas mais frequentes:

| Era | O sistema disse | Vezes |
|---|---|---|
| Outros | Passagem para peões | 17 |
| Perigo | Cedência de passagem | 2 |


## 3. Lê o número do limite de velocidade?

O número é lido pelo SigLIP num "print" do sinal (vista normal, ampliada 4×), no passo sinais.py limites. Duas verificações:

**3.1 Contra as marcações da Mapillary nas fotos**

| Tamanho do sinal na foto | Sinais | Número certo | Número errado | Não leu |
|---|---|---|---|---|
| Pequeno (< 40 px) | 0 | — | — | — |
| Médio (40–70 px) | 0 | — | — | — |
| Grande (≥ 70 px) | 0 | — | — | — |
| **Total** | 0 | — | — | — |

*Tamanho = largura do sinal em píxeis. Número certo = igual ao da referência.*

**3.2 Contra os pontos de sinais da Mapillary no mapa** (até 12 m)

| Tamanho do sinal na foto | Sinais | Número certo | Número errado | Não leu |
|---|---|---|---|---|
| Pequeno (< 40 px) | 0 | — | — | — |
| Médio (40–70 px) | 0 | — | — | — |
| Grande (≥ 70 px) | 0 | — | — | — |
| **Total** | 0 | — | — | — |

*Tamanho = largura do sinal em píxeis. Número certo = igual ao da referência.*

Números de referência nesta amostra: . A maioria das ruas da cidade é de 30 km/h, por isso há poucos exemplos de outros números.


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
| Sinais | 203 | 66% | 80% | 0,5 m |
| Iluminação | 523 | 63% | 80% | 2,0 m |
| Contentores | 132 | 94% | 96% | 0,7 m |

*Acertam até 5 m = % de objetos a 5 metros ou menos do sítio certo (os restantes falham). Erro típico = metade dos objetos erra menos do que isto.*

### 4.2 Taxa de acerto conforme a margem aceite

| Margem aceite | Sinais | Iluminação | Contentores |
|---|---|---|---|
| até 2 m | 59% | 50% | 75% |
| até 5 m | 66% | 63% | 94% |
| até 10 m | 80% | 80% | 96% |
| até 20 m | 96% | 98% | 100% |

*Cada valor = % de objetos que ficam dentro dessa margem. A taxa de falha é o resto até 100%.*

### 4.3 Perto ou longe da câmara

| Categoria | Objeto perto (até 10 m) | Objeto longe (mais de 10 m) |
|---|---|---|
| Sinais | 71% | 63% |
| Iluminação | 88% | 59% |
| Contentores | 100% | 92% |

*Valores = % que acertam até 5 m. Objetos perto da câmara ficam mais bem posicionados.*

### 4.4 Foto 360° ou foto frontal

| Categoria | Foto 360° | Foto frontal |
|---|---|---|
| Sinais | 66% | — |
| Iluminação | 63% | — |
| Contentores | 94% | — |

*Valores = % que acertam até 5 m. As fotos 360° (recentes) posicionam melhor.*

### 4.5 O resultado é mérito do sistema?

| Categoria | Erro típico do sistema | Erro típico ao acaso | Melhor que o acaso? |
|---|---|---|---|
| Sinais | 0,5 m | 10,9 m | sim |
| Iluminação | 2,0 m | 12,2 m | sim |
| Contentores | 0,7 m | 10,1 m | sim |

*Ao acaso = o mesmo objeto à mesma distância da câmara, mas numa direção sorteada. Se o sistema não for bem melhor do que isto, o número é sorte.*

### 4.6 O que falha: a distância ou a direção?

| Categoria | Erro na distância à câmara | Erro na direção |
|---|---|---|
| Sinais | 0,3 m | 0,4 m |
| Iluminação | 1,2 m | 0,5 m |
| Contentores | 0,6 m | 0,3 m |

*Valores típicos. A direção está bem medida; o que falha é a distância. A triangulação (usar várias fotos) corrige isto.*

Não medidos, por não haver referência a menos de 20 m: sinais 4, iluminação 33, contentores 1.


## 5. Há registos repetidos no catálogo?

| Categoria | Objetos repetidos | Registos por objeto |
|---|---|---|
| Sinais | 40% | 1,5 |
| Iluminação | 48% | 1,7 |
| Contentores | 16% | 1,2 |

*Objetos repetidos = % de objetos reais com mais de um registo no catálogo. Registos por objeto: o ideal é 1,0.*
