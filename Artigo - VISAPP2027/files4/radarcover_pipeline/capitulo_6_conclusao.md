# 6. Conclusions

Este trabalho propôs o **RadarCover**, um pipeline de nove etapas para
construção de ensembles compactos e conscientes de custo para restauração
de imagens de radar meteorológico, cujo núcleo é uma formulação de
**Weighted Set Multicover** (Seção 2). A reestruturação apresentada aqui,
tomando como modelo o protocolo experimental do OPFsembleR [16], substituiu
uma única comparação agregada — sem suporte estatístico — por uma avaliação
sob três orçamentos de ensemble ($K{=}2,4,6$) com teste de Friedman e
post-hoc de Nemenyi, usando os 16 regimes meteorológicos como unidade de
comparação independente, no mesmo papel que os $N$ datasets desempenham no
artigo-modelo. Os resultados foram obtidos processando a totalidade dos
**4.014 frames** disponíveis no subconjunto público do IPMet Radar Dataset.

Os resultados respondem às três perguntas de pesquisa do Capítulo 1 com
suporte estatístico real, não apenas estimativas pontuais:

- **RQ1** (RadarCover vs. Full-Ensemble): **confirmada, com significância
  estatística.** Em todos os três orçamentos testados, o RadarCover
  permanece no mesmo grupo estatístico do Full-Ensemble (Seção 5.3) usando
  entre 1/6 e 1/2 do número de modelos.
- **RQ2** (multicobertura e ponderação vs. Set Cover tradicional):
  **parcialmente confirmada.** A multicobertura de regimes críticos
  produziu, isoladamente, o maior efeito mensurável do estudo de ablação
  (Seção 5.6) — sua remoção derruba o SSIM em >10%. Já a ponderação por
  severidade e o uso de métricas meteorológicas na cobertura não se
  diferenciaram nesta escala, resultado atribuído à esparsidade da matriz
  de cobertura com apenas 12 candidatos, não a uma refutação desses
  componentes — permanece como questão em aberto para a escala plena.
- **RQ3** (seleção informada vs. baselines não informados): **confirmada,
  com significância estatística.** Random-Pruning e Diversity-Based
  ocupam consistentemente os grupos estatísticos de pior desempenho nos
  três diagramas de diferença crítica (Figuras 5.1–5.3).

Um achado adicional, não antecipado nas perguntas de pesquisa originais mas
central ao valor prático do método, emergiu da comparação multi-orçamento
(Seção 5.2–5.3): o RadarCover-Multicover **não consome o orçamento de
modelos oferecido além do necessário para satisfazer a cobertura** — ele
propõe consistentemente 2 modelos independentemente de o teto ser 2, 4 ou 6
—, enquanto o baseline Top-K sempre usa o orçamento inteiro e, a partir de
$K{=}4$, assume a liderança *numérica* de rank médio. Ainda assim, essa
liderança nunca atinge significância estatística frente ao RadarCover (CD
de Nemenyi $=2{,}625$; diferença observada máxima $=1{,}50$, em $K{=}6$).
Esse é o mesmo tipo de nuance que o próprio artigo-modelo reporta — nenhum
método de poda domina uniformemente em toda configuração de tamanho de
ensemble [16] —, e reforça que a contribuição do RadarCover deve ser
entendida como **otimalidade de custo sob restrição de cobertura**, não
como maximização incondicional de qualidade sob um orçamento de modelos.

## Limitações centrais a superar

Embora a avaliação tenha processado a totalidade dos 4.014 frames
disponíveis, o treinamento dos 12 modelos candidatos permanece em escala de
demonstração (60 iterações em motor de NumPy puro, CPU) — ver Seção 5.7
(Final Remarks) para a discussão completa. Essa limitação é de escala de
treinamento, não de desenho metodológico: o código do pipeline (Seções 2–4)
e a infraestrutura de teste estatístico (Seção 4.8) foram construídos para
que a transição à escala plena de treinamento exija apenas a porta dos
modelos candidatos para um framework de deep learning acelerado por GPU e
o aumento do número de iterações.

## Trabalhos futuros

**Roteamento dinâmico de modelos.** Classificar cada novo quadro de radar
em um regime (reaproveitando o classificador de clustering da Etapa 4) e
invocar em tempo de inferência apenas o subconjunto de modelos que cobrem
aquele regime específico, reduzindo ainda mais o custo médio frente à
execução do ensemble podado completo a cada quadro.

**Sequências temporais de radar.** Incorporar contexto temporal — como
entrada adicional aos modelos candidatos ou como sinal na construção de
regimes — dada a alta taxa de amostragem do IPMet e a forte autocorrelação
de sistemas de precipitação.

**Cobertura consciente de incerteza.** Substituir a cobertura binária
($\alpha_{u,j}\in\{0,1\}$) por uma formulação probabilística que propague
a incerteza de estimação — naturalmente maior em regimes com poucas
amostras — para a própria decisão de seleção do Multicover.

**Validação com treinamento em escala plena e mesmo protocolo estatístico.**
O passo mais direto: repetir exatamente o desenho experimental desta seção
(múltiplos orçamentos, Friedman + Nemenyi sobre regimes) com modelos
treinados em GPU por milhares de iterações e, opcionalmente, 30–50 regimes
— o que deve, por si só, esclarecer se a ponderação por severidade e as
métricas meteorológicas na cobertura (RQ2) se diferenciam estatisticamente
numa matriz de cobertura menos esparsa, e se a vantagem de custo do
RadarCover sobre o Top-K se mantém, cresce ou se inverte à medida que o
pool de candidatos cresce — replicando, para o domínio de radar
meteorológico, a mesma pergunta que [16] respondeu para classificação e
regressão tabular: a vantagem da poda estruturada sobre o ensemble completo
escala com o tamanho do pool, ou é um efeito específico de pools pequenos?

**Datasets adicionais.** Replicar o pipeline sobre radares de
características distintas e sobre o período completo do IPMet Radar
Dataset é o passo mais direto para testar a robustez das conclusões diante
da diversidade sazonal e geográfica do problema.
