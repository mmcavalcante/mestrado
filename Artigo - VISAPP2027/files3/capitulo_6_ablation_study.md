# 6. Ablation Study

Esta seção isola a contribuição individual de cada componente de projeto do
RadarCover — multicobertura de regimes críticos, pesos de importância
meteorológica, uso de métricas meteorológicas na definição da matriz de
cobertura, sensibilidade ao custo computacional na otimização e o esquema de
fusão do ensemble podado — através de seis configurações controladas,
mantendo fixos o pool de modelos candidatos, os regimes construídos (Etapa 4)
e a tabela de profiling (Etapa 5) entre todas as variantes. Cada variante
altera exatamente **um** componente da formulação do Weighted Set Multicover
(Etapa 7) ou do esquema de fusão (Etapa 8), permitindo atribuição causal do
efeito observado.

## 6.1 Protocolo experimental

Partindo da execução de referência (Tabela 6.1, linha A), cinco variantes
foram derivadas:

| Variante | Componente alterado | Configuração |
|---|---|---|
| **A — RadarCover completo** | — (referência) | multicobertura + pesos de regime + custo + fusão ponderada |
| **B — Sem multicobertura** | requisito de cobertura | $r_u = 1$ para todos os regimes, inclusive os críticos (em vez de $r_u=2$) |
| **C — Sem pesos de regime** | severidade | $w_u = 1$ uniforme para todos os regimes |
| **D — Sem métricas meteorológicas** | matriz de cobertura | $\alpha_{u,j}$ definido apenas pelo limiar de PSNR, ignorando FSS e CSI |
| **E — Cost-unaware** | função objetivo | custo $\text{cost}_j$ uniforme (a otimização passa a minimizar apenas a cardinalidade do ensemble) |
| **F — Fusão simples** | Etapa 8 | mesma seleção de A, mas fusão por média aritmética em vez de fusão ponderada por PSNR |

Todas as variantes foram resolvidas com o mesmo solver (ILP exato via
PuLP/CBC, com fallback para a heurística gulosa) e avaliadas sobre o mesmo
conjunto de patches de teste, garantindo comparabilidade direta.

## 6.2 Resultados

**Tabela 6.1** — Resultados do Ablation Study (execução de demonstração:
90 frames amostrados do IPMet Radar Dataset, 16 regimes, orçamento máximo de
5 modelos).

| Variante | Nº modelos | Regimes não atendidos | Parâmetros totais | Latência (ms) | PSNR | SSIM | FSS | CSI | Recall |
|---|---|---|---|---|---|---|---|---|---|
| A — RadarCover completo | 2 | 8 | 9.442 | 10,16 | **18,89** | **0,682** | 0,935 | 0,752 | 0,998 |
| B — Sem multicobertura | 1 | 7 | 4.721 | 4,38 | 18,82 | 0,611 | 0,936 | 0,755 | 0,997 |
| C — Sem pesos de regime | 2 | 8 | 9.442 | 9,36 | 18,89 | 0,682 | 0,935 | 0,752 | 0,998 |
| D — Sem métricas meteorológicas | 2 | 8 | 9.442 | 9,23 | 18,89 | 0,682 | 0,935 | 0,752 | 0,998 |
| E — Cost-unaware | 2 | 8 | 9.442 | 8,90 | 18,89 | 0,682 | 0,935 | 0,752 | 0,998 |
| F — Fusão simples | 2 | 8 | 9.442 | 11,91 | 18,89 | 0,682 | 0,935 | 0,751 | 0,998 |

Modelos selecionados em A, C, D, E e F: `{SRCNN#0, SRCNN#1}`. Em B:
`{SRCNN#1}`.

## 6.3 Discussão por componente

**Multicobertura (B vs. A).** É o componente com efeito mais evidente e
mensurável. Ao remover o requisito de dupla cobertura para regimes críticos
($r_u{=}1$ em vez de $r_u{=}2$), o solver considera satisfeita a cobertura
com um único modelo, reduzindo o ensemble de 2 para 1 membro. Isso corta a
latência pela metade e os parâmetros totais em 50%, mas reduz o SSIM em
**10,4 pontos percentuais relativos** (0,682 → 0,611) e o número de regimes
não atendidos cai de 8 para 7 — um efeito colateral esperado, já que exigir
$r_u{=}1$ é uma restrição estritamente mais fraca que $r_u{=}2$, logo mais
fácil de satisfazer com menos recursos. Este resultado evidencia
concretamente o trade-off que a multicobertura foi desenhada para gerenciar:
redundância em regimes críticos tem custo computacional diretamente
proporcional (aqui, +100% de parâmetros e latência), mas produz ganho de
robustez perceptual (SSIM) que a restrição sem redundância não captura.

**Pesos de regime, métricas meteorológicas e custo (C, D, E vs. A).** Nas
três variantes, a seleção final coincide exatamente com a configuração de
referência, produzindo métricas idênticas até a terceira casa decimal
(pequenas variações em latência são ruído de medição de tempo de CPU, não
diferenças de seleção). Isso **não** indica que esses componentes sejam
irrelevantes na formulação — mas sim uma limitação da escala de
demonstração usada nesta seção: com apenas 12 modelos candidatos e 16
regimes, a matriz de cobertura é suficientemente esparsa (taxa média de
cobertura ≈ 27%, Etapa 6) para que exista um único par de modelos
dominante (`SRCNN#0`, `SRCNN#1`) que satisfaz a maior parte dos requisitos
de cobertura a baixo custo, deixando pouca margem de decisão para pesos de
severidade, o tipo de métrica usada no threshold, ou a forma exata da função
de custo alterarem o resultado ótimo. Espera-se que essas três variantes se
diferenciem de forma mais clara na escala completa do artigo (4.014 frames,
30–50 regimes, pool de modelos mais diverso — ver Capítulo 7, Limitações),
onde múltiplos pares de modelos passam a ser factíveis e a escolha entre
eles volta a depender de severidade, tipo de métrica de cobertura e custo.
Este é registrado como direcionamento explícito para a versão final do
estudo, não como conclusão nula sobre a utilidade desses componentes.

**Fusão ponderada vs. simples (F vs. A).** Com apenas dois modelos
selecionados de desempenho médio semelhante (`SRCNN#0` e `SRCNN#1`, mesma
arquitetura e hiperparâmetros, seeds diferentes), o efeito da fusão ponderada
por PSNR é pequeno em magnitude, porém consistente na direção esperada: a
fusão ponderada obtém PSNR e SSIM marginalmente superiores à média simples
(18,891 vs. 18,885; 0,68202 vs. 0,68204 — diferença de SSIM dentro do ruído,
mas PSNR favorece a fusão ponderada), ao custo de um Recall ligeiramente
menor (0,9977 vs. 0,9978). Isso é coerente com o papel pretendido da fusão
ponderada: enviesar a saída final em direção ao modelo historicamente mais
preciso, o que tende a favorecer métricas de fidelidade média (PSNR/SSIM) em
detrimento da sensibilidade a picos de eco isolados que contribuem para o
Recall. Com um pool de modelos mais heterogêneo em desempenho — como ocorre
na escala completa, onde arquiteturas mais profundas (EDSR-Lite,
SwinIR-Light-Lite) tendem a superar SRCNN por margem maior — esse efeito
deve se acentuar.

## 6.4 Síntese

O Ablation Study confirma que a **multicobertura de regimes críticos** é o
componente de maior impacto isolado na configuração testada, com um
trade-off custo–qualidade claramente quantificável. Os demais componentes
(pesos de severidade, métricas meteorológicas na cobertura, sensibilidade a
custo e fusão ponderada) têm papel projetado para ganhar relevância à medida
que o espaço de modelos candidatos e regimes cresce — algo que a execução em
escala de demonstração, por construção, não tem amplitude suficiente para
evidenciar plenamente. A versão final deste capítulo, a ser executada com a
configuração completa (`n_frames: 4014`, `n_regimes: 30–50`, pool de
modelos treinados em GPU), deve reportar as mesmas seis variantes com maior
poder de diferenciação estatística, incluindo testes de significância
(ver Capítulo 4 — Metodologia Experimental) sobre as diferenças observadas.
