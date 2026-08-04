# 3. Proposed Method

O RadarCover é organizado em nove etapas sequenciais, ilustradas na Figura
3.1 (diagrama de metodologia original do projeto) e implementadas de ponta
a ponta neste trabalho (Capítulos 4–6). Esta seção descreve cada etapa
formalmente; a implementação de referência está disponível nos módulos
`src/*.py` do pipeline de código que acompanha este artigo.

## 3.1 Visão geral do pipeline

Dado um conjunto de imagens de radar de alta resolução (HR), o RadarCover
(i) sintetiza pares degradados (LR, HR) e extrai patches de treino/teste;
(ii) treina um pool heterogêneo de $M$ modelos candidatos de restauração;
(iii) particiona o espaço de patches em $U$ regimes meteorológicos via
clustering não supervisionado sobre descritores de eco; (iv) perfila cada
um dos $M$ modelos em cada um dos $U$ regimes; (v) constrói uma matriz de
cobertura binária $\alpha \in \{0,1\}^{U\times M}$ a partir de thresholds de
qualidade sobre o profiling; (vi) resolve um problema de **Weighted Set
Multicover** sobre essa matriz para selecionar o subconjunto de modelos de
menor custo que satisfaz os requisitos de cobertura de todos os regimes;
(vii) monta um ensemble podado que combina apenas os modelos selecionados
via fusão ponderada; e (viii) avalia o resultado contra sete baselines de
poda (Capítulo 4, Seção 4.6).

## 3.2 Pré-processamento de imagem (Etapa 2)

Pares (LR, HR) são sintetizados a partir de cada imagem HR real via um
operador de degradação $D_s(\cdot)$ com fator de escala $s$: suavização
gaussiana ($\sigma_{\text{blur}}$), subamostragem por *average pooling* de
fator $s$, e ruído gaussiano aditivo ($\sigma_{\text{ruído}}$). Patches de
tamanho fixo ($64\times64$ px na configuração de referência) são extraídos
de cada imagem, com amostragem priorizada por intensidade média do patch
para evitar que a esparsidade natural de eco de precipitação (tipicamente
< 15% da área de cobertura do radar em um instante, Capítulo 4) domine o
conjunto de treino com patches triviais de fundo vazio.

## 3.3 Pool de modelos candidatos (Etapa 3)

O pool reúne $M$ modelos de restauração/super-resolução heterogêneos em
arquitetura e custo computacional — seis famílias que cobrem um espectro
representativo da evolução arquitetural de super-resolução de imagem única
(SISR): SRCNN [1] (três convoluções sobre entrada já upsampled, a mais
rasa e barata do pool), ESPCN [5] (extrai features em baixa resolução e
faz upsample só na última camada via sub-pixel convolution, reduzindo
custo para o mesmo fator de escala), EDSR-Lite (blocos residuais sem
normalização em lote, inspirado em [6]), RLFN-Lite (blocos de "feature
local residual" com gargalo 1×1→3×3→1×1, inspirado em [7], desenhado para
poucos parâmetros/FLOPs), SwinIR-Light-Lite (aproximação sem atenção real
de [8], por inviabilidade de self-attention em NumPy puro — Seção 4.4) e
CRMN-Lite (mistura convolucional-recorrente, inspirado no modelo original
dos próprios autores do dataset [4]). Cada arquitetura é instanciada com
múltiplas seeds aleatórias de inicialização, conforme indicado no diagrama
da metodologia ("*Multiple random seeds*"). A heterogeneidade do pool é
condição necessária para que a etapa de Weighted Set Multicover (Seção 2)
tenha espaço de decisão não trivial: um pool homogêneo em desempenho
tornaria a cobertura de regimes redundante entre quaisquer $k$ modelos
escolhidos.

## 3.4 Construção de regimes (Etapa 4)

Cada patch $p$ é mapeado a um vetor de descritores
$\mathbf{f}(p) \in \mathbb{R}^8$: intensidade média, desvio-padrão,
percentil 90, fração de eco (*echo ratio*), entropia de Shannon do
histograma de intensidade, densidade local de eco (fração de pixels acima
de um limiar suavizada por filtro de média), número de componentes
conectados (proxy de número de células de precipitação distintas no
patch), e distância normalizada ao centro do radar. Os vetores
$\{\mathbf{f}(p)\}$ do conjunto de teste são padronizados e agrupados via
K-Means em $U$ regimes ($U \in [30,50]$ na configuração de escala plena;
$U{=}16$ na execução de demonstração, Capítulo 4). A cada regime $u$
associam-se dois atributos derivados:

- **Peso de severidade** $w_u = 1 + 3\cdot\overline{I}_u + 0{,}05\cdot\overline{C}_u$
  (normalizado para média 1 sobre todos os regimes), onde
  $\overline{I}_u$ é a intensidade média e $\overline{C}_u$ o número médio
  de componentes conectados dos patches do regime $u$ — regimes de maior
  intensidade e mais núcleos convectivos recebem peso maior.
- **Indicador de criticidade** $\text{crit}_u \in \{0,1\}$, definido como
  $\text{crit}_u{=}1$ se a fração média de eco do regime excede um limiar
  operacional (0,15 na configuração de referência).

## 3.5 Profiling por regime (Etapa 5)

Cada modelo candidato $j$ é avaliado sobre todos os patches de cada regime
$u$, produzindo cinco métricas médias: PSNR e SSIM (qualidade visual); FSS
e CSI (preservação meteorológica — FSS mede excedência de limiar em
janelas espaciais locais [17]; CSI mede acertos sobre falsos alarmes e
omissões em pixels de eco crítico [18]); e Recall sobre
pixels de eco crítico. Adicionalmente, mede-se a latência de inferência
$\ell_j$ (ms/patch) e o número de parâmetros $\pi_j$ de cada modelo. O
resultado é uma tabela de profiling $(j, u) \mapsto$
(PSNR$_{ju}$, SSIM$_{ju}$, FSS$_{ju}$, CSI$_{ju}$, Recall$_{ju}$,
$\ell_j$, $\pi_j$).

## 3.6 Matriz de cobertura (Etapa 6)

A partir da tabela de profiling, define-se

$$\alpha_{u,j} = \mathbb{1}\left[\text{PSNR}_{ju} \geq \tau_{\text{PSNR}} \;\wedge\; \text{FSS}_{ju} \geq \tau_{\text{FSS}} \;\wedge\; \text{CSI}_{ju} \geq \tau_{\text{CSI}}\right],$$

isto é, o modelo $j$ "cobre" o regime $u$ se e somente se atende
simultaneamente aos três limiares de qualidade mínima. O custo de cada
modelo é definido como

$$\text{cost}_j = 0{,}5\cdot\frac{\ell_j}{\max_k \ell_k} + 0{,}5\cdot\frac{\pi_j}{\max_k \pi_k},$$

uma combinação normalizada de latência relativa e tamanho relativo do
modelo. O requisito de cobertura de cada regime é
$r_u = 2$ se $\text{crit}_u{=}1$, e $r_u = 1$ caso contrário — a
formalização do requisito de "multicoverage" para ecos críticos indicado
no diagrama da metodologia.

## 3.7 Otimização Weighted Set Multicover (Etapa 7)

A seleção do ensemble podado é formulada como o programa linear inteiro:

$$
\begin{aligned}
\text{minimizar} \quad & \sum_{j=1}^{M} \text{cost}_j \cdot x_j \\
\text{sujeito a} \quad & \sum_{j=1}^{M} \alpha_{u,j}\, x_j \;\geq\; r_u, \quad \forall u \in \{1,\dots,U\} \\
& x_j \in \{0,1\}, \quad \forall j
\end{aligned}
$$

onde $x_j{=}1$ indica que o modelo $j$ é selecionado. Esta é a formulação
de **Weighted Set Multicover**: "weighted" porque tanto o custo por modelo
($\text{cost}_j$) quanto — na função objetivo relaxada, usada quando a
instância é infactível dentro de um orçamento máximo de modelos — o peso
de severidade $w_u$ penalizam a solução; "multicover" porque $r_u$ pode
exceder 1, exigindo redundância explícita em regimes críticos. O problema é
resolvido de forma exata via programação linear inteira (solver CBC, através
da biblioteca PuLP), com uma heurística gulosa (razão custo-benefício
marginal ponderada por severidade) como alternativa de menor custo
computacional para instâncias maiores. Um orçamento opcional
$\sum_j x_j \leq K$ pode limitar o tamanho máximo do ensemble.

## 3.8 Ensemble podado e inferência (Etapa 8)

O conjunto selecionado $S = \{j : x_j{=}1\}$ define o ensemble final,
cuja predição sobre uma nova entrada LR é a combinação ponderada

$$\hat{y} = \sum_{j \in S} \beta_j \cdot f_j(\text{LR}), \qquad \beta_j = \frac{\overline{\text{PSNR}}_j}{\sum_{k \in S} \overline{\text{PSNR}}_k},$$

onde $\overline{\text{PSNR}}_j$ é o PSNR médio do modelo $j$ sobre todos os
regimes perfilados — os modelos historicamente mais precisos recebem peso
maior na fusão final (Capítulo 6 avalia esta escolha de desenho frente à
alternativa de fusão por média aritmética simples).

## 3.9 Avaliação (Etapa 9)

O ensemble podado é comparado, sobre um conjunto de teste disjunto do
treino (Capítulo 4, Seção 4.2), contra sete baselines de poda (Capítulo 4,
Seção 4.6) em cinco eixos: qualidade visual (PSNR, SSIM), preservação
meteorológica (FSS, CSI, Recall), eficiência computacional (latência,
parâmetros totais, número de modelos), trade-off qualidade–latência, e
inspeção qualitativa de exemplos individuais de restauração. Os resultados
completos desta avaliação são apresentados no Capítulo 5.
