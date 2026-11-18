# Escolha das frases (fase 1: contraste nas fotos de referência)

Contraste = parecida com a sua classe menos a mais parecida das outras. Escolhidas = as melhores entre as que também são bem parecidas com a sua classe.

## Classificador — iluminacao / poste_iluminacao

| Frase | Parecida (sua classe) | Contraste | Escolhida |
|---|---|---|---|
| a modern street light on a tall grey metal pole | 0.162 | +0.047 | ✅ |
| a utility pole | 0.109 | +0.045 |  |
| a tall street light pole | 0.165 | +0.042 | ✅ |
| a tall slim metal street light pole with a lamp head at the top | 0.171 | +0.038 | ✅ |
| a street light pole with an arm | 0.165 | +0.032 |  |
| a lighting pole | 0.163 | +0.023 |  |
| a light pole | 0.158 | +0.021 |  |
| a street lamp on a tall metal post | 0.159 | +0.010 |  |
| a street light | 0.155 | +0.005 |  |
| a lamppost | 0.153 | -0.006 |  |

## Classificador — iluminacao / candeeiro

| Frase | Parecida (sua classe) | Contraste | Escolhida |
|---|---|---|---|
| a classic lantern on a wrought iron bracket | 0.133 | +0.073 |  |
| two lanterns on a wall bracket | 0.128 | +0.064 |  |
| an ornamental cast iron lamp post with lanterns | 0.159 | +0.061 | ✅ |
| a wrought iron lantern | 0.155 | +0.060 | ✅ |
| a vintage street lamp | 0.153 | +0.047 |  |
| a wall-mounted lantern | 0.131 | +0.041 |  |
| an ornamental street lamp | 0.171 | +0.040 | ✅ |
| a decorative street lamp | 0.168 | +0.027 |  |
| a lamp | 0.136 | +0.020 |  |
| a street lantern | 0.164 | +0.015 |  |

## Classificador — residuos / ilha_reciclagem

| Frase | Parecida (sua classe) | Contraste | Escolhida |
|---|---|---|---|
| a large dark blue plastic recycling bell container with a coloured opening | 0.142 | +0.060 |  |
| blue, yellow and green recycling containers | 0.156 | +0.056 | ✅ |
| brown recycling containers with blue, yellow and green openings | 0.173 | +0.040 | ✅ |
| a recycling igloo | 0.128 | +0.035 |  |
| recycling containers with coloured openings | 0.180 | +0.029 | ✅ |
| a recycling bank | 0.163 | +0.024 |  |
| a row of recycling containers | 0.152 | +0.022 |  |
| a glass recycling container | 0.127 | +0.015 |  |
| a recycling bin | 0.121 | +0.007 |  |
| a recycling station | 0.156 | +0.003 |  |

## Classificador — residuos / indiferenciado

| Frase | Parecida (sua classe) | Contraste | Escolhida |
|---|---|---|---|
| a green wheeled garbage container | 0.150 | +0.067 | ✅ |
| a large wheeled garbage container | 0.141 | +0.046 |  |
| a four-wheeled waste container | 0.156 | +0.034 | ✅ |
| a large plastic garbage container with a lid on wheels | 0.138 | +0.033 |  |
| a wheelie bin | 0.144 | +0.031 | ✅ |
| a black or dark grey wheeled plastic garbage container with a lid | 0.111 | +0.026 |  |
| a dumpster | 0.131 | +0.021 |  |
| a garbage container | 0.145 | +0.013 |  |
| a municipal waste container | 0.143 | +0.002 |  |

## Classificador — residuos / subterraneo

| Frase | Parecida (sua classe) | Contraste | Escolhida |
|---|---|---|---|
| stainless steel waste inlets on a paved platform | 0.124 | +0.064 |  |
| a row of stainless steel bins with coloured stripes | 0.143 | +0.035 | ✅ |
| underground container inlets on the sidewalk | 0.147 | +0.024 | ✅ |
| a grey cylinder with a black dome lid | 0.055 | +0.024 |  |
| a waste deposit column | 0.156 | +0.020 | ✅ |
| a large round semi-underground waste container with wooden slat cladding and a black dome lid with a lifting ring | 0.081 | +0.016 |  |
| an underground waste container | 0.147 | +0.008 |  |
| a semi-underground waste container | 0.128 | +0.002 |  |
| a row of dark metal underground waste container inlets with yellow, blue or green openings on a metal platform | 0.142 | +0.000 |  |

## Detetor — iluminacao

| Frase | Pontuação (sua categoria) | Contraste | Escolhida |
|---|---|---|---|
| a decorative street lamp | 55% | +53% | ✅ |
| a wall-mounted street lamp | 54% | +52% | ✅ |
| an ornamental street lamp | 53% | +52% | ✅ |
| a lamppost | 53% | +51% | ✅ |
| a lamp post | 51% | +50% |  |
| a street lamp | 51% | +49% |  |
| a streetlight | 50% | +48% |  |
| a street light | 49% | +47% |  |
| a street light on a tall pole | 47% | +45% |  |
| a tall street light pole | 43% | +42% |  |
| a lighting pole | 38% | +36% |  |
| a light pole | 37% | +35% |  |
| a street light pole with an arm | 30% | +28% |  |
| a street lantern | 20% | +17% |  |
| a lamp | 17% | +16% |  |
| a utility pole | 17% | +16% |  |

## Detetor — residuos

| Frase | Pontuação (sua categoria) | Contraste | Escolhida |
|---|---|---|---|
| a recycling container | 54% | +51% | ✅ |
| a garbage container | 55% | +50% | ✅ |
| a four-wheeled waste container | 54% | +49% | ✅ |
| a waste container | 55% | +48% | ✅ |
| a recycling bin | 53% | +48% |  |
| a garbage can | 57% | +48% |  |
| an underground waste container | 55% | +48% |  |
| a semi-underground waste container | 54% | +48% |  |
| a large wheeled garbage container | 50% | +46% |  |
| a trash bin | 48% | +41% |  |
| a wheelie bin | 44% | +38% |  |
| a row of recycling containers | 34% | +32% |  |
| a dumpster | 35% | +32% |  |
| a waste deposit column | 27% | +21% |  |
| a recycling igloo | 13% | +11% |  |
