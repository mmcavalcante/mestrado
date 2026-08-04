# 4. Dataset and Experimental Setup

## 4.1 Conjunto de dados: IPMet Radar Dataset

Os experimentos utilizam o **IPMet Radar Dataset** [4], disponibilizado
publicamente em `github.com/rafaepires/IPMet-Radar-Dataset` sob licença CC
BY 4.0. O subconjunto público contém **4.014 imagens** de refletividade de
radar meteorológico coletadas pelo radar do Centro de Pesquisas
Meteorológicas de Bauru (IPMet/UNESP), projeção CAPPI a 3,5 km de altura,
varredura PPI a 0,3° de elevação, grade cartesiana de **640×640 pixels**.

Uma ressalva metodológica relevante, identificada por inspeção direta dos
arquivos (e não documentada no README do repositório): os frames
disponíveis no subconjunto público cobrem o período de **31/12/2023 a
31/01/2024** (um mês), não o intervalo completo de janeiro/2022 a
janeiro/2024 mencionado como origem do dataset completo (67.424 imagens).
Isso é declarado explicitamente como característica do subconjunto usado, e
retomado na Seção 5.6 (Final Remarks) quanto à generalização
temporal/sazonal dos resultados.

As imagens são arquivos PNG **RGBA**, não grayscale de 8 bits como sugere o
README — os pixels codificam refletividade através de uma paleta de cores
meteorológica, com o canal alfa indicando presença/ausência de eco
(alfa = 0 fora do alcance de detecção). A intensidade de refletividade usada
neste trabalho é obtida por $I = L \cdot \mathbb{1}[\alpha>0]$, onde $L$ é a
luminância ponderada ($0{,}299R+0{,}587G+0{,}114B$) normalizada em $[0,1]$
— uma proxy documentada da refletividade em dBZ, não uma decodificação
exata da paleta de cores oficial do produto IPMet.

## 4.2 Divisão dos dados

Divisão **cronológica** (não aleatória), evitando vazamento de informação
entre quadros de radar temporalmente adjacentes e altamente correlacionados
— cada imagem representa o estado da atmosfera a cada ~5–8 minutos, com
forte autocorrelação temporal em eventos meteorológicos. Proporções: 60%
treino / 20% validação / 20% teste, respeitando a ordem cronológica dos
frames (o conjunto de teste corresponde sempre ao período mais recente da
amostra).

## 4.3 Protocolo de degradação

As entradas de baixa resolução (LR) são sintetizadas a partir da imagem de
alta resolução (HR) real por: (i) suavização gaussiana ($\sigma=1{,}0$),
simulando perda de nitidez óptica/eletrônica do sensor; (ii) subamostragem
por *average pooling* no fator de escala $s\in\{2,4\}$ (este trabalho reporta
$s=2$); (iii) ruído gaussiano aditivo ($\sigma_{\text{ruído}}=0{,}02$),
simulando ruído eletrônico de aquisição. Os patches de treino/avaliação
($64\times64$ pixels) são amostrados priorizando regiões com eco de
precipitação (média de intensidade do patch), já que a grande maioria da
área de cobertura do radar não contém eco em um dado instante (fração média
de pixels com eco > 0,05 observada nos frames: tipicamente < 15%), o que
tornaria o treinamento dominado por patches triviais (fundo vazio) se a
amostragem fosse uniforme.

## 4.4 Pool de modelos candidatos e treinamento

Seis arquiteturas de restauração (Seção 3.3), cada uma instanciada com 2
seeds aleatórias distintas, totalizando **12 modelos candidatos**. Por
restrição do ambiente de execução (CPU única, sem GPU disponível, disco
insuficiente para as dependências CUDA do PyTorch), os modelos foram
implementados sobre um motor de convolução 2D com retropropagação manual em
NumPy puro (`nn_core.py`), preservando a topologia de cada arquitetura
(conexões residuais, sub-pixel convolution, mistura convolucional-recorrente)
em largura/profundidade reduzida. Otimizador Adam ($\text{lr}=3\times10^{-3}$),
função de perda MSE, 60 iterações por modelo sobre patches amostrados
aleatoriamente do conjunto de treino — configuração de **demonstração**; a
configuração de escala plena (`ESCALAR PARA PAPER` em `configs/default.yaml`)
prevê treino em GPU por milhares de iterações sobre o dataset completo.

**Tabela 4.1** — Visão geral do pool de modelos candidatos (métricas médias
sobre todos os regimes perfilados, execução de demonstração).

| Modelo | Parâmetros | PSNR médio (dB) | SSIM médio | FSS médio | CSI médio | Recall médio | Latência (ms) |
|---|---|---|---|---|---|---|---|
| SRCNN#1 | 4.721 | 18,61 | 0,602 | 0,914 | 0,698 | 0,996 | 27,44 |
| SRCNN#0 | 4.721 | 17,96 | 0,638 | 0,899 | 0,668 | 0,996 | 27,51 |
| ESPCN#1 | 3.316 | 17,52 | 0,614 | 0,895 | 0,659 | 0,991 | 3,38 |
| ESPCN#0 | 3.316 | 17,50 | 0,624 | 0,894 | 0,664 | 0,992 | 3,40 |
| SwinIR-Light-Lite#0 | 4.825 | 17,47 | 0,368 | 0,901 | 0,639 | 0,987 | 20,62 |
| EDSR-Lite#1 | 8.077 | 16,53 | 0,387 | 0,878 | 0,610 | 0,979 | 32,08 |
| RLFN-Lite#1 | 1.396 | 16,17 | 0,441 | 0,889 | 0,616 | 0,964 | 1,59 |
| EDSR-Lite#0 | 8.077 | 15,39 | 0,333 | 0,896 | 0,602 | 0,935 | 35,92 |
| CRMN-Lite#0 | 1.374 | 14,55 | 0,170 | 0,838 | 0,532 | 0,929 | 3,23 |
| RLFN-Lite#0 | 1.396 | 14,33 | 0,307 | 0,865 | 0,542 | 0,886 | 1,64 |
| CRMN-Lite#1 | 1.374 | 14,03 | 0,170 | 0,844 | 0,525 | 0,905 | 3,64 |
| SwinIR-Light-Lite#1 | 4.825 | 13,98 | 0,138 | 0,861 | 0,505 | 0,878 | 22,72 |

## 4.5 Construção de regimes e profiling

Regimes meteorológicos construídos via K-Means sobre 8 descritores por
patch (intensidade média, desvio-padrão, percentil 90, fração de eco,
entropia, densidade local de eco, número de componentes conectados,
distância normalizada ao centro do radar — Seção 3.4), com **16 regimes**
na execução de demonstração (30–50 na escala plena; 9 deles classificados
como críticos). Cada modelo é perfilado em cada regime com PSNR, SSIM, FSS
(limiar de excedência 0,10; janela 8×8), CSI e Recall sobre pixels de eco
crítico (mesmo limiar).

## 4.6 Baselines

Sete métodos de comparação (Seção 3): melhor modelo isolado (*Best-Single*,
sempre 1 modelo), ensemble completo sem poda (*Full-Ensemble*, sempre os 12
modelos do pool), fronteira de Pareto custo×qualidade (*Pareto-Pruning*,
tamanho definido pela própria fronteira, não por um orçamento externo) —
esses três, análogos ao papel do *Stacking* nas Tabelas 1–3 do artigo-
modelo [16], são constantes entre as diferentes configurações de orçamento
(Seção 4.7) — e quatro métodos que recebem o mesmo orçamento $K$ do
RadarCover a cada configuração: os $K$ modelos de maior PSNR médio
ponderado (*Top-K*), poda aleatória de $K$ modelos (*Random-Pruning*), poda
por diversidade via clustering de perfis regime-a-regime em $K$ grupos
(*Diversity-Based*), e Set Cover tradicional restrito a $K$ modelos
(*Traditional-Set-Cover*: $r_u{=}1$ para todos os regimes, sem ponderação
por severidade — caso particular não ponderado do método proposto, Seção
2.4).

## 4.7 Desenho experimental multi-orçamento

Inspirado no protocolo do artigo-modelo [16] — que compara os métodos de
poda sob três tamanhos de pool (10, 30 e 50 classificadores, Tabelas 1–3) —,
este trabalho avalia o RadarCover e os quatro baselines sensíveis a
orçamento sob três **orçamentos de ensemble podado**: $K \in \{2, 4, 6\}$
modelos. Diferentemente do artigo-modelo, onde $K$ controla o tamanho do
*pool inicial* antes da poda, aqui $K$ é um **teto superior** imposto à
otimização (`max_models` na Etapa 7): o RadarCover-Multicover pode
selecionar *menos* que $K$ modelos se a cobertura já estiver satisfeita a
menor custo — comportamento explorado em detalhe na Seção 5.2, onde se
observa que o método propõe consistentemente 2 modelos independentemente do
teto oferecido, enquanto Top-K, Random-Pruning e Diversity-Based sempre
consomem o orçamento inteiro por definição.

## 4.8 Teste de significância estatística: Friedman + Nemenyi

Diferentemente da versão anterior deste estudo — que declarava a ausência
de testes de significância como limitação por potência estatística
insuficiente —, esta reestruturação adota o mesmo protocolo do artigo-
modelo [16] (Figura 6): teste de **Friedman** seguido do post-hoc de
**Nemenyi** com diagrama de diferença crítica (CD), conforme Demšar (2006)
[19]. A adaptação necessária foi identificar a unidade de comparação
correta: o artigo-modelo ranqueia os métodos em cada um de $N{=}15$–$20$
*datasets*; aqui, cada um dos **16 regimes meteorológicos** desempenha o
mesmo papel — uma unidade de comparação independente sobre a qual cada um
dos 8 métodos (RadarCover + 7 baselines) produz uma pontuação de PSNR,
repetida por $n_{\text{seeds}}{=}3$ realizações aleatórias do ruído de
degradação (Seção 4.3) para estabilizar a estimativa por regime.

Formalmente, com $k{=}8$ métodos e $N{=}16$ blocos (regimes), a estatística
de Friedman é

$$\chi^2_F = \frac{12N}{k(k+1)}\left[\sum_j R_j^2 - \frac{k(k+1)^2}{4}\right],$$

onde $R_j$ é o rank médio do método $j$ (rank 1 = melhor PSNR naquele
regime, médias em caso de empate). Sob rejeição da hipótese nula (diferença
global significativa), aplica-se o post-hoc de Nemenyi: dois métodos
diferem significativamente se seus ranks médios diferem por pelo menos a
diferença crítica

$$CD = q_\alpha\sqrt{\frac{k(k+1)}{6N}},$$

com $q_{0{,}05}(k{=}8) = 3{,}031$ (Tabela 5(a) de Demšar [19], conferida na
fonte primária), resultando em $CD \approx 2{,}625$ ranks para $N{=}16$.
Complementarmente, calcula-se a estatística $F$ de Iman-Davenport — menos
conservadora que a $\chi^2_F$ pura — para checagem cruzada. Os resultados
completos, por orçamento $K$, estão na Seção 5.3.

## 4.9 Hardware

Execução de demonstração: 1 vCPU, ~3,9 GB RAM, sem GPU. Tempo total de
execução: pipeline principal ≈50s; ablation study ≈43s; estudo
multi-orçamento completo ($K\in\{2,4,6\}$, 3 seeds, 8 métodos, 16 regimes)
≈116s.
