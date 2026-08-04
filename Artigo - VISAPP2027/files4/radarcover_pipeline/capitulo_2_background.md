# 2. Background: Weighted Set Cover and Multicover

Esta seção apresenta o arcabouço teórico sobre o qual o RadarCover é
construído — o problema de Cobertura de Conjuntos e sua variante ponderada
com multicobertura —, no mesmo papel que a exposição do Optimum-Path Forest
cumpre no artigo que serviu de modelo estrutural para esta reescrita,
OPFsembleR [16]: fundamentar, antes de descrever o método propriamente dito
(Seção 3), o mecanismo combinatório que decide quais modelos compõem o
ensemble final.

## 2.1 O problema de Cobertura de Conjuntos

Dado um universo finito de elementos $\mathcal{U} = \{1, \dots, U\}$ e uma
coleção de subconjuntos $\mathcal{S} = \{S_1, \dots, S_M\}$, com
$S_j \subseteq \mathcal{U}$, o problema de **Set Cover** busca a menor
subcoleção $\mathcal{S}' \subseteq \mathcal{S}$ tal que
$\bigcup_{S_j \in \mathcal{S}'} S_j = \mathcal{U}$. Karp o incluiu entre os
21 problemas originalmente provados NP-completos, o que significa que
nenhum algoritmo eficiente (tempo polinomial) é conhecido para encontrar a
cobertura ótima exata em instâncias arbitrariamente grandes.

Na sua variante **ponderada** (*Weighted Set Cover*), cada subconjunto
$S_j$ tem um custo $c_j > 0$, e o objetivo passa a ser minimizar o custo
total da subcoleção escolhida, não sua cardinalidade. Chvátal demonstrou
que o algoritmo guloso — a cada passo, escolher o subconjunto que minimiza
o custo por elemento ainda não coberto — garante uma solução com custo no
máximo $H(d)$ vezes o ótimo, onde $H(d) = \sum_{i=1}^{d} 1/i$ é o número
harmônico e $d$ é o tamanho do maior subconjunto; esse é o melhor fator de
aproximação possível para o problema em geral (sob hipóteses usuais de
complexidade computacional).

## 2.2 De Set Cover a Set Multicover

O **Set Multicover** generaliza o Set Cover ao substituir o requisito
"cada elemento coberto por pelo menos um subconjunto" por "cada elemento
$u \in \mathcal{U}$ coberto por pelo menos $r_u \geq 1$ subconjuntos", onde
$r_u$ é um requisito de cobertura específico do elemento. O Set Cover
clássico é o caso particular em que $r_u = 1$ para todo $u$. O mesmo
algoritmo guloso de Chvátal se estende naturalmente ao caso multicover,
mantendo garantia de aproximação logarítmica: a cada passo, escolhe-se o
subconjunto que maximiza a razão entre cobertura marginal ainda necessária
(min entre o que falta e o que o subconjunto oferece) e seu custo.

## 2.3 Cobertura de conjuntos como seleção de modelos

A conexão entre Set Cover e seleção/poda de ensemble é direta quando se
interpreta:

- **elementos do universo** ($\mathcal{U}$) como *condições, subdomínios ou
  segmentos do espaço de entrada* que um sistema de aprendizado precisa
  atender bem (no RadarCover: regimes meteorológicos; em outros domínios,
  poderiam ser classes, clusters de instâncias, ou datasets, como no
  artigo-modelo);
- **subconjuntos** ($S_j$) como *o conjunto de condições em que o modelo
  candidato $j$ tem desempenho satisfatório* (uma linha da matriz de
  cobertura $\alpha$, Seção 3.6);
- **custo** ($c_j$) como o *custo computacional* de incluir o modelo $j$ no
  ensemble final (latência, número de parâmetros).

Sob essa lente, minimizar o Set Cover ponderado é exatamente **podar um
ensemble ao menor subconjunto de modelos que ainda atende toda a
diversidade de condições observadas** — a formalização matemática do
objetivo central de qualquer método de *ensemble pruning* [11], [13]. A
generalização para Multicover formaliza, adicionalmente, um requisito que
os métodos clássicos de poda por métrica agregada (Top-K/Pareto-
Pruning/Diversity-Based neste trabalho, e a maioria dos métodos revisados
por [16]) não expressam nativamente: **redundância obrigatória em
condições críticas** — aqui, exigir $r_u \geq 2$ para regimes
meteorológicos de alta severidade, para que a falha ou baixo desempenho de
um único modelo nessas condições não comprometa a restauração.

## 2.4 Formulação como Programa Linear Inteiro

A instância de Weighted Set Multicover usada no RadarCover (detalhada como
Etapa 7 na Seção 3.7) é resolvida de duas formas complementares:

1. **Exata**, via Programação Linear Inteira (solver CBC, biblioteca PuLP)
   — viável porque o pool de modelos candidatos ($M \leq 50$ tipicamente) e
   o número de regimes ($U \in [16, 50]$) mantêm a instância pequena o
   suficiente para resolução exata em tempo prático;
2. **Heurística gulosa**, extensão do algoritmo de Chvátal para o caso
   multicover ponderado (Seção 2.2), usada como *fallback* caso o solver
   exato não esteja disponível ou a instância cresça além do que é exato
   praticamente viável.

Diferente de trabalhos anteriores que aplicam cobertura de conjuntos à
seleção de modelos com pesos apenas nos subconjuntos (custo por modelo), a
formulação aqui pondera também os **elementos do universo** — os regimes —
por severidade meteorológica (Seção 3.4), e permite $r_u > 1$. O Capítulo 5
(Resultados) isola quantitativamente, via um estudo de ablação, o efeito
específico de cada uma dessas duas extensões frente ao caso clássico não
ponderado ($r_u{=}1$ para todos, $w_u{=}1$ uniforme — o baseline
*Traditional-Set-Cover*).
