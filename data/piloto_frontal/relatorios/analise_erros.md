# Análise de erros

64 fotos analisadas · modelo owlv2.

As referências (Mapillary e OpenStreetMap) servem só para medir. Também têm erros e não estão completas: use estes números sobretudo para comparar versões do sistema.

## 1. Encontra os objetos?

| Categoria | Objetos nas fotos | Encontrados | Taxa |
|---|---|---|---|
| Sinais | 93 | 88 | 95% |
| Iluminação | 14 | 12 | 86% |
| Contentores | 0 | 0 | — |

*Objetos nas fotos = marcados pela Mapillary (com pelo menos 24 píxeis). Contentores: só os com >= 0.9 m de largura (a Mapillary marca também papeleiras, que a ficha não pede).*


## 2. Acerta no tipo de sinal?

| Tipo de sinal | Sinais | Acerto |
|---|---|---|
| Estacionamento | 27 | 100% |
| Obrigação | 14 | 64% |
| Passagem para peões | 12 | 100% |
| Trânsito de sentido único | 10 | 90% |
| Paragem e estacionamento proibidos | 10 | 100% |
| Cedência de passagem | 7 | 100% |
| Direção / indicação | 3 | 100% |
| Sentido proibido | 2 | 100% |
| Perigo | 2 | 50% |
| Outros | 1 | 0% |
| **Total** | **88** | **91%** |

*Sinais = sinais encontrados pelo sistema que a Mapillary também identificou. Acerto = o sistema deu o mesmo tipo.*

Trocas mais frequentes:

| Era | O sistema disse | Vezes |
|---|---|---|
| Obrigação | Outros | 3 |
| Obrigação | Trânsito de sentido único | 2 |
| Outros | Estacionamento proibido | 1 |
| Trânsito de sentido único | Obrigação | 1 |
| Perigo | Outros | 1 |


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
| Sinais | 225 | 54% | 81% | 4,7 m |
| Iluminação | 390 | 29% | 73% | 7,3 m |
| Contentores | 10 | 60% | 100% | 2,7 m |

*Acertam até 5 m = % de objetos a 5 metros ou menos do sítio certo (os restantes falham). Erro típico = metade dos objetos erra menos do que isto.*

### 4.2 Taxa de acerto conforme a margem aceite

| Margem aceite | Sinais | Iluminação | Contentores |
|---|---|---|---|
| até 2 m | 14% | 4% | 40% |
| até 5 m | 54% | 29% | 60% |
| até 10 m | 81% | 73% | 100% |
| até 20 m | 100% | 100% | 100% |

*Cada valor = % de objetos que ficam dentro dessa margem. A taxa de falha é o resto até 100%.*

### 4.3 Perto ou longe da câmara

| Categoria | Objeto perto (até 10 m) | Objeto longe (mais de 10 m) |
|---|---|---|
| Sinais | 83% | 53% |
| Iluminação | 0% | 29% |
| Contentores | — | 60% |

*Valores = % que acertam até 5 m. Objetos perto da câmara ficam mais bem posicionados.*

### 4.4 Foto 360° ou foto frontal

| Categoria | Foto 360° | Foto frontal |
|---|---|---|
| Sinais | — | 54% |
| Iluminação | — | 29% |
| Contentores | — | 60% |

*Valores = % que acertam até 5 m. As fotos 360° (recentes) posicionam melhor.*

### 4.5 O resultado é mérito do sistema?

| Categoria | Erro típico do sistema | Erro típico ao acaso | Melhor que o acaso? |
|---|---|---|---|
| Sinais | 4,7 m | 10,9 m | sim |
| Iluminação | 7,3 m | 10,9 m | só um pouco |
| Contentores | 2,7 m | 11,0 m | sim |

*Ao acaso = o mesmo objeto à mesma distância da câmara, mas numa direção sorteada. Se o sistema não for bem melhor do que isto, o número é sorte.*

### 4.6 O que falha: a distância ou a direção?

| Categoria | Erro na distância à câmara | Erro na direção |
|---|---|---|
| Sinais | 2,7 m | 2,3 m |
| Iluminação | 5,5 m | 2,9 m |
| Contentores | 0,9 m | 1,6 m |

*Valores típicos. A direção está bem medida; o que falha é a distância. A triangulação (usar várias fotos) corrige isto.*

Não medidos, por não haver referência a menos de 20 m: sinais 1.


## 5. Há registos repetidos no catálogo?

| Categoria | Objetos repetidos | Registos por objeto |
|---|---|---|
| Sinais | 46% | 1,6 |
| Iluminação | 40% | 1,5 |
| Contentores | 0% | 1,0 |

*Objetos repetidos = % de objetos reais com mais de um registo no catálogo. Registos por objeto: o ideal é 1,0.*
