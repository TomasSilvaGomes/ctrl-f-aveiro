# Análise de erros

457 fotos analisadas · modelo owlv2.

As referências (Mapillary e OpenStreetMap) servem só para medir. Também têm erros e não estão completas: use estes números sobretudo para comparar versões do sistema.

## 1. Encontra os objetos?

| Categoria | Objetos nas fotos | Encontrados | Taxa |
|---|---|---|---|
| Sinais | 1 238 | 1 129 | 91% |
| Iluminação | 1 733 | 166 | 10% |
| Contentores | 107 | 46 | 43% |

*Objetos nas fotos = marcados pela Mapillary (com pelo menos 24 píxeis). Contentores: só os com >= 0.9 m de largura (a Mapillary marca também papeleiras, que a ficha não pede).*


## 2. Acerta no tipo de sinal?

| Tipo de sinal | Sinais | Acerto |
|---|---|---|
| Obrigação | 317 | 52% |
| Passagem para peões | 273 | 100% |
| Paragem e estacionamento proibidos | 193 | 100% |
| Perigo | 164 | 76% |
| Limite de velocidade | 90 | 100% |
| Cedência de passagem | 74 | 100% |
| Outros | 16 | 75% |
| Estacionamento | 1 | 100% |
| Proibido virar / inversão de marcha | 1 | 0% |
| **Total** | **1 129** | **83%** |

*Sinais = sinais encontrados pelo sistema que a Mapillary também identificou. Acerto = o sistema deu o mesmo tipo.*

Trocas mais frequentes:

| Era | O sistema disse | Vezes |
|---|---|---|
| Obrigação | Outros | 153 |
| Perigo | Cedência de passagem | 25 |
| Perigo | Outros | 14 |
| Outros | Passagem para peões | 3 |
| Proibido virar / inversão de marcha | Obrigação | 1 |


## 3. Lê o número do limite de velocidade?

O número é lido pelo SigLIP num "print" do sinal (vista normal, ampliada 4×), no passo sinais.py limites. Duas verificações:

**3.1 Contra as marcações da Mapillary nas fotos**

| Tamanho do sinal na foto | Sinais | Número certo | Número errado | Não leu |
|---|---|---|---|---|
| Pequeno (< 40 px) | 56 | 100% | 0% | 0% |
| Médio (40–70 px) | 31 | 100% | 0% | 0% |
| Grande (≥ 70 px) | 3 | 100% | 0% | 0% |
| **Total** | 90 | 100% | 0% | 0% |

*Tamanho = largura do sinal em píxeis. Número certo = igual ao da referência.*

**3.2 Contra os pontos de sinais da Mapillary no mapa** (até 12 m)

| Tamanho do sinal na foto | Sinais | Número certo | Número errado | Não leu |
|---|---|---|---|---|
| Pequeno (< 40 px) | 70 | 97% | 3% | 0% |
| Médio (40–70 px) | 35 | 100% | 0% | 0% |
| Grande (≥ 70 px) | 12 | 100% | 0% | 0% |
| **Total** | 117 | 98% | 2% | 0% |

*Tamanho = largura do sinal em píxeis. Número certo = igual ao da referência.*

Números de referência nesta amostra: 30 km/h: 117. A maioria das ruas da cidade é de 30 km/h, por isso há poucos exemplos de outros números.


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
| Sinais | 2 351 | 71% | 92% | 0,9 m |
| Iluminação | 917 | 56% | 88% | 3,7 m |
| Contentores | 359 | 57% | 63% | 2,4 m |

*Acertam até 5 m = % de objetos a 5 metros ou menos do sítio certo (os restantes falham). Erro típico = metade dos objetos erra menos do que isto.*

### 4.2 Taxa de acerto conforme a margem aceite

| Margem aceite | Sinais | Iluminação | Contentores |
|---|---|---|---|
| até 2 m | 55% | 43% | 48% |
| até 5 m | 71% | 56% | 57% |
| até 10 m | 92% | 88% | 63% |
| até 20 m | 100% | 100% | 79% |

*Cada valor = % de objetos que ficam dentro dessa margem. A taxa de falha é o resto até 100%.*

### 4.3 Perto ou longe da câmara

| Categoria | Objeto perto (até 10 m) | Objeto longe (mais de 10 m) |
|---|---|---|
| Sinais | 75% | 70% |
| Iluminação | 75% | 53% |
| Contentores | 73% | 52% |

*Valores = % que acertam até 5 m. Objetos perto da câmara ficam mais bem posicionados.*

### 4.4 Foto 360° ou foto frontal

| Categoria | Foto 360° | Foto frontal |
|---|---|---|
| Sinais | 71% | — |
| Iluminação | 56% | — |
| Contentores | 57% | — |

*Valores = % que acertam até 5 m. As fotos 360° (recentes) posicionam melhor.*

### 4.5 O resultado é mérito do sistema?

| Categoria | Erro típico do sistema | Erro típico ao acaso | Melhor que o acaso? |
|---|---|---|---|
| Sinais | 0,9 m | 6,1 m | sim |
| Iluminação | 3,7 m | 8,7 m | sim |
| Contentores | 2,4 m | 11,2 m | sim |

*Ao acaso = o mesmo objeto à mesma distância da câmara, mas numa direção sorteada. Se o sistema não for bem melhor do que isto, o número é sorte.*

### 4.6 O que falha: a distância ou a direção?

| Categoria | Erro na distância à câmara | Erro na direção |
|---|---|---|
| Sinais | 0,7 m | 0,3 m |
| Iluminação | 1,6 m | 1,0 m |
| Contentores | 1,4 m | 0,7 m |

*Valores típicos. A direção está bem medida; o que falha é a distância. A triangulação (usar várias fotos) corrige isto.*

Não medidos, por não haver referência a menos de 20 m: sinais 9, iluminação 6, contentores 42.


## 5. Há registos repetidos no catálogo?

| Categoria | Objetos repetidos | Registos por objeto |
|---|---|---|
| Sinais | 87% | 4,0 |
| Iluminação | 35% | 1,5 |
| Contentores | 36% | 1,5 |

*Objetos repetidos = % de objetos reais com mais de um registo no catálogo. Registos por objeto: o ideal é 1,0.*
