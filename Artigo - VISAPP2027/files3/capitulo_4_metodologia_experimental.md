# 4. Experimental Methodology

## 4.1 Conjunto de dados: IPMet Radar Dataset

Os experimentos utilizam o **IPMet Radar Dataset** [Pires et al., ICASSP
2025], disponibilizado publicamente em
`github.com/rafaepires/IPMet-Radar-Dataset` sob licença CC BY 4.0. O
subconjunto público contém **4.014 imagens** de refletividade de radar
meteorológico coletadas pelo radar do Centro de Pesquisas Meteorológicas de
Bauru (IPMet/UNESP), projeção CAPPI a 3,5 km de altura, varredura PPI a
0,3° de elevação, grade cartesiana de **640×640 pixels**.

Uma ressalva metodológica relevante, identificada por inspeção direta dos
arquivos (e não documentada no README do repositório): os frames
disponíveis no subconjunto público cobrem o período de **31/12/2023 a
31/01/2024** (um mês), não o intervalo completo de janeiro/2022 a
janeiro/2024 mencionado como origem do dataset completo (67.424 imagens).
Isso deve ser declarado explicitamente na versão final do artigo como
característica do subconjunto usado, e é retomado no Capítulo 7
(Limitações) quanto à generalização temporal/sazonal dos resultados.

As imagens são arquivos PNG **RGBA**, não grayscale de 8 bits como sugere o
README — os pixels codificam refletividade através de uma paleta de cores
meteorológica, com o canal alfa indicando presença/ausência de eco
(alfa = 0 fora do alcance de detecção). A intensidade de refletividade usada
neste trabalho é obtida por $I = L \cdot \mathbb{1}[\alpha>0]$, onde $L$ é a
luminância ponderada ($0{,}299R+0{,}587G+0{,}114B$) normalizada em $[0,1]$.
Essa é uma proxy documentada da refletividade em dBZ, não uma decodificação
exata da paleta de cores oficial do produto IPMet (Capítulo 7).

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

Seis arquiteturas de restauração (SRCNN, ESPCN, EDSR-Lite, RLFN-Lite,
SwinIR-Light-Lite, CRMN-Lite), cada uma instanciada com 2 seeds aleatórias
distintas, totalizando **12 modelos candidatos**. Por restrição do ambiente
de execução (CPU única, sem GPU disponível, disco insuficiente para as
dependências CUDA do PyTorch), os modelos foram implementados sobre um motor
de convolução 2D com retropropagação manual em NumPy puro
(`nn_core.py`), preservando a topologia de cada arquitetura (conexões
residuais, sub-pixel convolution, mistura convolucional-recorrente) em
largura/profundidade reduzida. Otimizador Adam ($\text{lr}=3\times10^{-3}$),
função de perda MSE, 60 iterações por modelo sobre patches amostrados
aleatoriamente do conjunto de treino — configuração de **demonstração**;
a configuração de escala plena (`ESCALAR PARA PAPER` em `configs/default.yaml`)
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
distância normalizada ao centro do radar — Capítulo 3), com 16 regimes na
execução de demonstração (30–50 na escala plena). Cada modelo é perfilado em
cada regime com PSNR, SSIM, FSS (limiar de excedência 0,10; janela 8×8),
CSI e Recall sobre pixels de eco crítico (mesmo limiar).

## 4.6 Baselines

Sete métodos de comparação (Capítulo 3): melhor modelo isolado
(*Best-Single*), ensemble completo sem poda (*Full-Ensemble*), poda
aleatória (*Random-Pruning*, mesma cardinalidade da solução do RadarCover),
os $k$ modelos de maior PSNR médio ponderado (*Top-K*), poda por diversidade
via clustering de perfis regime-a-regime (*Diversity-Based*), fronteira de
Pareto custo×qualidade (*Pareto-Pruning*) e Set Cover tradicional
(*Traditional-Set-Cover*: $r_u{=}1$ para todos os regimes, sem ponderação
por severidade — caso particular não ponderado do método proposto).

## 4.7 Hardware e testes estatísticos

Execução de demonstração: 1 vCPU, ~3,9 GB RAM, sem GPU (tempo total de
execução do pipeline completo: ≈50s; ablation study: ≈43s). Dada a escala
reduzida de amostragem (90 frames, poucos patches por regime — mediana de
poucas amostras por célula da tabela de profiling), testes de significância
estatística formal (ex.: teste $t$ pareado ou Wilcoxon signed-rank entre
RadarCover e cada baseline, por regime) **não são reportados nesta versão**
por potência estatística insuficiente; ver Capítulo 7. A versão de escala
plena, sobre os 4.014 frames completos, deve reportar intervalos de
confiança bootstrap (1.000 reamostragens) e testes pareados por regime,
com correção de Bonferroni para comparações múltiplas entre os 7 baselines.
