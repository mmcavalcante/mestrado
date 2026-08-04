# 1. Introduction

## 1.1 Contexto e problema

Radares meteorológicos de varredura PPI (*Plan Position Indicator*) são a
principal fonte de observação de curto prazo de precipitação em escala
local, sustentando aplicações que vão do alerta de eventos convectivos
severos à assimilação de dados em modelos de previsão numérica. A
resolução espacial efetiva dessas observações, no entanto, degrada com a
distância ao radar devido ao alargamento do feixe e à projeção de uma
varredura polar sobre uma grade cartesiana — uma limitação física do
sensor, não corrigível apenas por melhor instrumentação em prazo curto.
Redes neurais convolucionais de super-resolução de imagem única (SISR)
demonstraram, em benchmarks de visão computacional, a capacidade de
reconstruir detalhes de alta frequência a partir de entradas degradadas
[1], e essa capacidade já foi transportada com sucesso para dados de radar
meteorológico real, tanto em abordagens baseadas em CNNs simples aplicadas
a varreduras NEXRAD [2] quanto em arquiteturas adversariais aplicadas a
ecos de radar chinês CINRAD [3], e mais recentemente em arquiteturas
híbridas conv-recorrentes desenhadas especificamente para o radar do IPMet
(Bauru, SP), fonte de dados deste trabalho [4].

## 1.2 Motivação

A literatura recente de super-resolução de imagem oferece dezenas de
arquiteturas candidatas, de modelos rasos e baratos como SRCNN [1] e ESPCN
[5] a arquiteturas profundas com dezenas de blocos residuais como EDSR [6],
redes eficientes desenhadas para orçamentos restritos de FLOPs como RLFN
[7], e arquiteturas baseadas em transformers como SwinIR [8]. Cada uma
ocupa um ponto distinto no espaço custo-qualidade, e — como confirmado
empiricamente neste próprio trabalho (Capítulo 5) — nenhuma domina
uniformemente todas as demais em todos os regimes de operação: um modelo
pode restaurar melhor núcleos convectivos intensos e isolados, enquanto
outro é mais fiel para precipitação estratiforme difusa e de baixa
intensidade. Esse padrão é análogo ao problema clássico de *ensemble
learning*, no qual combinar múltiplos modelos fracos tipicamente supera
qualquer modelo individual [9], mas ao custo de somar a latência e a
demanda computacional de todos os membros — um custo proibitivo para
sistemas operacionais de meteorologia, que frequentemente processam
varreduras em intervalos de poucos minutos sob restrição de tempo real.

## 1.3 Lacuna de pesquisa

A literatura de **poda de ensemble** (*ensemble pruning*) trata exatamente
desse trade-off, buscando subconjuntos de um pool de modelos que preservem
(ou até melhorem) o desempenho do ensemble completo usando menos membros
[10]–[13]. Os métodos de poda mais estudados seguem majoritariamente duas
linhas: (i) seleção gulosa por desempenho agregado [13], poda por
diversidade de erro [12], ou busca por fronteira de Pareto custo-acurácia
[10], todos avaliando e selecionando modelos com uma **métrica global
única** por modelo, computada sobre todo o conjunto de avaliação; e (ii)
**clustering não supervisionado dos modelos** por similaridade de
predições, escolhendo um representante (protótipo) por cluster — a linha
seguida pelo trabalho mais diretamente relacionado a este, o OPFsembleR
[16], que usa Optimum-Path Forest não supervisionado para agrupar modelos
de classificação e regressão heterogêneos e demonstra, em 30 datasets
públicos, que o subconjunto podado frequentemente iguala ou supera o
ensemble completo (*stacking*) — achado replicado de forma independente
neste trabalho (Seção 5) para um domínio, modelos e mecanismo de poda
completamente distintos.

Nenhuma dessas duas linhas, no entanto, expressa nativamente que, em
domínios com heterogeneidade espacial e de regime como a meteorologia
radar, o desempenho relativo dos modelos **varia sistematicamente por tipo
de condição observada** (célula convectiva isolada, precipitação
estratiforme, céu limpo, etc.) — informação que tanto uma métrica agregada
quanto um cluster de similaridade *global* de predições apagam por
construção, já que ambos operam sobre o comportamento médio do modelo, não
sobre seu comportamento condicionado ao regime. Adicionalmente, a
formulação clássica de **cobertura de conjuntos** (*Set Cover*), problema
combinatório NP-difícil bem caracterizado na literatura de otimização [14],
[15] — uma terceira linha, de natureza combinatória em vez de estatística
ou baseada em clustering —, trata todo elemento do universo (aqui, cada
regime meteorológico) com igual importância e exige apenas cobertura
simples ($r_u{=}1$), ignorando tanto a severidade meteorológica
diferenciada de cada regime quanto a necessidade de redundância em
condições operacionalmente críticas (ex.: núcleos convectivos intensos,
onde a falha de um único modelo não
deveria comprometer a restauração).

Não foi identificado, na literatura consultada, um método que combine (i)
profiling de modelos de restauração de imagem de radar **por regime
meteorológico** em vez de por métrica agregada global, e (ii) uma
formulação de **Set Multicover ponderado** que permita exigir cobertura
redundante especificamente em regimes críticos, ponderada por severidade
meteorológica — a lacuna que este trabalho busca preencher.

## 1.4 Solução proposta

Este artigo propõe o **RadarCover**, um pipeline de nove etapas (Capítulo
3) que: (1) constrói um pool heterogêneo de modelos candidatos de
restauração/super-resolução de imagem de radar; (2) particiona o espaço de
entrada em regimes meteorológicos via clustering não supervisionado sobre
descritores de eco; (3) perfila cada modelo candidato em cada regime com
métricas de qualidade visual (PSNR, SSIM) e de preservação meteorológica
(FSS [17], CSI [18]); (4) formula a seleção do ensemble podado como um
problema de **Weighted Set Multicover** — generalização do Set Cover
clássico que pondera regimes por severidade e permite exigir multicobertura
em regimes críticos —, resolvido de forma exata via programação linear
inteira; e (5) monta um ensemble compacto que combina apenas os modelos
selecionados via fusão ponderada.

## 1.5 Perguntas de pesquisa

- **RQ1.** Um ensemble selecionado por Weighted Set Multicover consegue
  igualar ou superar a qualidade de restauração do ensemble completo
  (sem poda) usando significativamente menos modelos e menor custo
  computacional?
- **RQ2.** A exigência explícita de multicobertura em regimes críticos e a
  ponderação por severidade meteorológica produzem um ensemble
  qualitativamente melhor do que a formulação de Set Cover tradicional
  (não ponderada, sem redundância)?
- **RQ3.** A seleção informada por profiling por regime supera baselines
  que ignoram essa informação (poda aleatória, poda por diversidade pura)?

## 1.6 Contribuições

1. Uma formulação de **Weighted Set Multicover** aplicada à poda de
   ensembles de restauração de imagem, incorporando pesos de severidade por
   regime e requisitos de multicobertura para condições críticas —
   extensão não encontrada na literatura de poda de ensemble consultada.
2. Um protocolo de **construção de regimes meteorológicos** via descritores
   de eco (intensidade, entropia, densidade, conectividade, distância ao
   radar) e clustering não supervisionado, permitindo profiling de modelos
   por condição de operação em vez de média global.
3. Uma **implementação executável de ponta a ponta** do pipeline completo
   (Capítulo 3), validada sobre dados reais do IPMet Radar Dataset [4],
   incluindo sete baselines de comparação (Capítulo 5) e um estudo de
   ablação isolando a contribuição de cada componente de projeto (Capítulo
   6) — com resultados empíricos que já corroboram RQ1 e RQ2 mesmo em
   escala de demonstração reduzida (Capítulo 7).
4. Uma discussão explícita e fundamentada em evidência das limitações de
   escala, generalização e validade estatística do estudo (Capítulo 7),
   incluindo um achado metodológico relevante sobre a composição temporal
   real do subconjunto público do dataset utilizado.

## Referências (Capítulos 1–2)

[1] C. Dong, C. C. Loy, K. He, and X. Tang, "Learning a deep convolutional
network for image super-resolution," in *Proc. ECCV*, 2014, pp. 184–199.

[2] A. Geiss and J. W. Hardin, "Radar super resolution using a deep
convolutional neural network," *J. Atmos. Ocean. Technol.*, vol. 37,
no. 12, pp. 2197–2207, 2020.

[3] H. Chen, X. Zhang, Y. Liu, and Q. Zeng, "Generative adversarial networks
capabilities for super-resolution reconstruction of weather radar echo
images," *Atmosphere*, vol. 10, no. 9, art. 555, 2019.

[4] R. G. Pires, D. F. S. Santos, R. V. Calheiros, J. P. Papa, I. H. Lee,
S. Bakshi, and K. Muhammad, "A convolutional recurrent mixer network for
radar meteorological image super-resolution," in *Proc. IEEE ICASSP*,
Hyderabad, India, 2025, pp. 1–5.

[5] W. Shi, J. Caballero, F. Huszár, J. Totz, A. P. Aitken, R. Bishop,
D. Rueckert, and Z. Wang, "Real-time single image and video
super-resolution using an efficient sub-pixel convolutional neural
network," in *Proc. IEEE CVPR*, 2016, pp. 1874–1883.

[6] B. Lim, S. Son, H. Kim, S. Nah, and K. M. Lee, "Enhanced deep residual
networks for single image super-resolution," in *Proc. IEEE CVPRW*, 2017,
pp. 136–144.

[7] F. Kong, M. Li, S. Liu, D. Liu, J. He, Y. Bai, F. Chen, and L. Fu,
"Residual local feature network for efficient super-resolution," in *Proc.
IEEE CVPRW (NTIRE)*, 2022.

[8] J. Liang, J. Cao, G. Sun, K. Zhang, L. Van Gool, and R. Timofte,
"SwinIR: Image restoration using Swin Transformer," in *Proc. IEEE ICCVW*,
2021, pp. 1833–1844.

[9] Z.-H. Zhou, J. Wu, and W. Tang, "Ensembling neural networks: Many
could be better than all," *Artif. Intell.*, vol. 137, no. 1–2, pp.
239–263, 2002.

[10] R. Hu, S. Zhou, Y. Liu, and Z. Tang, "Margin-based Pareto ensemble
pruning: An ensemble pruning algorithm that learns to search optimized
ensembles," *Comput. Intell. Neurosci.*, vol. 2019, art. 7560872, 2019.

[11] D. D. Margineantu and T. G. Dietterich, "Pruning adaptive boosting,"
in *Proc. 14th Int. Conf. Mach. Learn. (ICML)*, 1997, pp. 211–218.

[12] G. Martínez-Muñoz, D. Hernández-Lobato, and A. Suárez, "An analysis
of ensemble pruning techniques based on ordered aggregation," *IEEE
Trans. Pattern Anal. Mach. Intell.*, vol. 31, no. 2, pp. 245–259, 2009.

[13] R. Caruana, A. Niculescu-Mizil, G. Crew, and A. Ksikes, "Ensemble
selection from libraries of models," in *Proc. 21st Int. Conf. Mach.
Learn. (ICML)*, 2004.

[14] R. M. Karp, "Reducibility among combinatorial problems," in
*Complexity of Computer Computations*, R. E. Miller and J. W. Thatcher,
Eds. New York: Plenum, 1972, pp. 85–103.

[15] V. Chvátal, "A greedy heuristic for the set-covering problem," *Math.
Oper. Res.*, vol. 4, no. 3, pp. 233–235, 1979.

[16] D. S. Jodas, L. A. Passos, D. Rodrigues, K. A. P. da Costa, and J. P.
Papa, "OPFsembleR: An Optimum-Path Forest-based Framework for Ensemble
Pruning," submitted to the Int. Conf. on Pattern Recognition (ICPR), 2026.

[17] N. M. Roberts and H. W. Lean, "Scale-selective verification of
rainfall accumulations from high-resolution forecasts of convective
events," *Mon. Weather Rev.*, vol. 136, no. 1, pp. 78–97, 2008.

[18] J. T. Schaefer, "The critical success index as an indicator of
warning skill," *Weather Forecast.*, vol. 5, no. 4, pp. 570–575, 1990.

[19] J. Demšar, "Statistical Comparisons of Classifiers over Multiple Data
Sets," *J. Mach. Learn. Res.*, vol. 7, pp. 1–30, 2006.
