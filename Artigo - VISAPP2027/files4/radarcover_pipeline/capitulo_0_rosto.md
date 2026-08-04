# RadarCover: Ensembles Compactos e Conscientes de Custo para Restauração de Imagens de Radar Meteorológico via Weighted Set Multicover

*Estrutura e protocolo experimental (múltiplos orçamentos de ensemble,
teste de Friedman + post-hoc de Nemenyi) inspirados em OPFsembleR: An
Optimum-Path Forest-based Framework for Ensemble Pruning (Jodas et al.,
submetido ao ICPR 2026).*

## Resumo

A restauração e super-resolução de imagens de radar meteorológico via
aprendizado profundo admite múltiplas arquiteturas candidatas, cada uma
ocupando um ponto distinto do espaço custo–qualidade e com desempenho
heterogêneo entre diferentes regimes meteorológicos. Métodos tradicionais
de poda de ensemble selecionam subconjuntos de modelos a partir de uma
métrica de desempenho agregada única, ignorando essa heterogeneidade por
regime. Este trabalho propõe o RadarCover, um pipeline de nove etapas que
particiona o espaço de entrada em regimes meteorológicos via clustering
sobre descritores de eco, perfila cada modelo candidato por regime, e
formula a seleção do ensemble podado como um problema de Weighted Set
Multicover — generalização do Set Cover clássico que pondera regimes por
severidade e exige cobertura redundante em condições críticas. O pipeline
foi validado sobre dados reais do IPMet Radar Dataset (radar de Bauru, SP)
sob três orçamentos de ensemble ($K{=}2,4,6$), com teste de Friedman e
post-hoc de Nemenyi (diagramas de diferença crítica) usando os 16 regimes
meteorológicos como unidade de comparação — protocolo estatístico ausente
na versão anterior deste estudo, adotado seguindo o artigo-modelo. Em
todos os três orçamentos, a diferença global entre os 8 métodos comparados
é estatisticamente significativa ($p<0{,}0001$), e o RadarCover permanece
estatisticamente indistinguível do melhor baseline usando de 2× a 6× menos
modelos, propondo consistentemente apenas 2 modelos independentemente do
teto oferecido. Um estudo de ablação isola a multicobertura de regimes
críticos como o componente de maior impacto individual. Limitações de
escala, generalização e o alcance específico da ponderação por severidade
nesta configuração são discutidas em detalhe.

**Palavras-chave:** super-resolução de imagem; radar meteorológico; poda
de ensemble; cobertura de conjuntos ponderada; teste de Friedman; diagrama
de diferença crítica; aprendizado profundo.

## Abstract

Deep learning-based restoration and super-resolution of weather radar
imagery admits multiple candidate architectures, each occupying a distinct
point in the cost–quality space with performance that varies across
meteorological regimes. Traditional ensemble pruning methods select model
subsets from a single aggregate performance metric, ignoring this
regime-level heterogeneity. This work proposes RadarCover, a nine-stage
pipeline that partitions the input space into meteorological regimes via
clustering over echo descriptors, profiles each candidate model per
regime, and formulates pruned-ensemble selection as a Weighted Set
Multicover problem — a generalization of classical Set Cover that weighs
regimes by severity and requires redundant coverage under critical
conditions. The pipeline was validated on real data from the IPMet Radar
Dataset (Bauru, Brazil) under three ensemble budgets ($K{=}2,4,6$), with a
Friedman test and Nemenyi post-hoc (critical-difference diagrams) using the
16 meteorological regimes as the comparison unit — a statistical protocol
absent from the prior version of this study, adopted following the model
paper. Across all three budgets, the global difference among the 8
compared methods is statistically significant ($p<0.0001$), and RadarCover
remains statistically indistinguishable from the best baseline while using
2× to 6× fewer models, consistently proposing only 2 models regardless of
the offered ceiling. An ablation study isolates multicoverage of critical
regimes as the highest-impact individual component. Scale, generalization,
and the specific scope of severity weighting at this configuration are
discussed in detail.

**Keywords:** image super-resolution; weather radar; ensemble pruning;
weighted set cover; Friedman test; critical difference diagram; deep
learning.

---

## Sumário

1. Introduction
2. Background: Weighted Set Cover and Multicover
3. Proposed Method
4. Dataset and Experimental Setup
5. Results
6. Conclusions

Referências
