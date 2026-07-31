# 2. Related Work

## 2.1 Restauração de imagem de radar

A aplicação de aprendizado profundo à restauração e super-resolução de
imagens de radar meteorológico é uma linha de pesquisa relativamente
recente, mas já com precedentes diretos. Geiss e Hardin [2] treinaram uma
rede convolucional para reverter a degradação artificial de varreduras PPI
do radar NEXRAD (radar meteorológico operacional dos EUA), demonstrando
que a abordagem preserva bordas nítidas e variabilidade de pequena escala
melhor que interpolação convencional — protocolo de degradação artificial
(par LR/HR sintetizado a partir de um único produto de alta resolução)
adotado também neste trabalho (Capítulo 4), por ausência de pares
LR/HR medidos independentemente. Chen et al. [3] propuseram uma abordagem
baseada em redes adversariais generativas (GAN) para o mesmo problema sobre
radares chineses CINRAD, reportando ganhos de PSNR/SSIM sobre métodos de
representação esparsa não local, e observando explicitamente que ecos de
radar têm textura e informação de borda ricas — o que motiva o uso de
perdas perceptuais/adversariais além do erro quadrático médio (MSE) puro.
Mais diretamente relacionado a este trabalho, Pires et al. [4] propuseram a
Convolutional Recurrent Mixer Network (CRMN), uma arquitetura híbrida
conv-recorrente, para super-resolução de imagens do próprio radar do IPMet
(Bauru, SP) — os mesmos autores que disponibilizaram publicamente o IPMet
Radar Dataset utilizado neste artigo (Capítulo 4). A CRMN é incluída no
pool de modelos candidatos deste trabalho (`CRMN-Lite`, Capítulo 3), ainda
que em versão simplificada por restrição do ambiente de execução (Capítulo
7).

## 2.2 Modelos de super-resolução de imagem

O pool de modelos candidatos do RadarCover (Capítulo 3) cobre um espectro
representativo da evolução arquitetural de super-resolução de imagem única
(SISR) em visão computacional geral, cuja transferência para o domínio de
radar meteorológico segue o precedente estabelecido pela Seção 2.1.

**SRCNN** [1], proposta por Dong et al., foi a primeira rede convolucional
aplicada a SISR: três camadas convolucionais operando sobre a entrada já
upsampled por interpolação bicúbica, realizando extração de patches,
mapeamento não linear e reconstrução. É rasa, barata e ainda hoje usada
como baseline de referência na literatura.

**ESPCN** [5], de Shi et al., introduziu a *sub-pixel convolution*
(rearranjo de canais para upsample, aqui implementado como `PixelShuffle`
no motor `nn_core.py` — Capítulo 3): em vez de fazer upsample da entrada
antes de processá-la (como SRCNN), a rede extrai features na resolução
baixa e faz upsample apenas na última camada, reduzindo drasticamente o
custo computacional para o mesmo fator de escala — refletido neste
trabalho na latência de ESPCN (Tabela 4.1), consistentemente mais baixa
que a de SRCNN apesar de arquitetura de profundidade comparável.

**EDSR** [6], de Lim et al., removeu a normalização em lote (*batch
normalization*) dos blocos residuais usados em arquiteturas anteriores
(argumentando que ela restringe a faixa de valores de ativação, prejudicial
em tarefas de regressão como SR) e empilhou dezenas de blocos residuais
para aumentar a capacidade representacional — a arquitetura mais profunda
do pool original, aqui reduzida (`EDSR-Lite`) a poucos blocos por restrição
de treino em CPU (Capítulo 7).

**RLFN** [7], de Kong et al. (equipe ByteDance, vencedora do desafio NTIRE
2022 de super-resolução eficiente), propôs o bloco de "feature local
residual" (RLFB): três convoluções por bloco com agregação por soma em vez
de concatenação (usada em arquiteturas anteriores como RFDN), reduzindo
fragmentação de operações e favorecendo tempo de execução em hardware real
sem sacrificar PSNR/SSIM — motivação de desenho diretamente alinhada ao
objetivo de eficiência do RadarCover, e refletida nos resultados deste
trabalho: `RLFN-Lite` obteve a menor latência de todo o pool mesmo com
poucos parâmetros (Capítulo 5).

**SwinIR** [8], de Liang et al., foi a primeira arquitetura de restauração
de imagem baseada em Swin Transformer amplamente adotada, combinando
blocos de auto-atenção em janelas locais (RSTB — *Residual Swin Transformer
Block*) com uma conexão residual rasa, obtendo estado da arte em SR,
denoising e redução de artefatos de compressão com menos parâmetros que
CNNs equivalentes. Por inviabilidade de implementar atenção eficientemente
em NumPy puro sobre imagens completas (restrição do ambiente de execução,
Capítulo 3), este trabalho usa uma aproximação sem atenção real
(`SwinIR-Light-Lite`), limitação explicitamente registrada no Capítulo 7.

## 2.3 Ensemble learning

A combinação de múltiplos modelos preditivos para superar o desempenho de
qualquer modelo individual é um resultado bem estabelecido em aprendizado
de máquina, com Zhou et al. [9] demonstrando formalmente, através da
decomposição viés-variância do erro, que um subconjunto cuidadosamente
selecionado de uma coleção de redes neurais pode generalizar melhor que a
coleção completa ("*many could be better than all*") — resultado que
motiva diretamente a hipótese central deste trabalho, testada
empiricamente no Capítulo 5 (RadarCover vs. Full-Ensemble). A literatura de
ensemble para restauração/super-resolução de imagem, especificamente, tende
a usar fusão simples (média ou média ponderada de saídas, como no esquema
de fusão do RadarCover — Capítulo 3, Etapa 8) sobre um pequeno número de
modelos previamente escolhidos por especialistas, sem um critério
sistemático e quantitativo de seleção — a lacuna que a poda de ensemble
(Seção 2.4) e este trabalho especificamente buscam preencher.

## 2.4 Poda de ensemble (Ensemble Pruning)

O problema de reduzir um ensemble treinado a um subconjunto menor, sem
perda relevante de desempenho, foi formalizado inicialmente por Margineantu
e Dietterich [11], que propuseram o método *Kappa pruning* para reduzir
ensembles de AdaBoost preservando acurácia. Trabalhos subsequentes
propuseram critérios alternativos de seleção: Caruana et al. [13]
introduziram a seleção gulosa a partir de uma biblioteca de modelos
pré-treinados heterogêneos (adicionando iterativamente o modelo que mais
melhora o desempenho do subconjunto atual — estratégia comparável ao
baseline *Top-K* usado neste trabalho, Capítulo 4); Martínez-Muñoz et al.
[12] analisaram sistematicamente técnicas de poda baseadas em ordenação por
agregação, mostrando que a ordem em que os membros são incluídos no
ensemble afeta fortemente o resultado final; e Hu et al. [10] propuseram
poda por fronteira de Pareto (custo × acurácia), formulação usada
diretamente como um dos sete baselines de comparação deste trabalho
(*Pareto-Pruning*, Capítulo 4). Um traço comum a esses métodos — e a lacuna
explorada no Capítulo 1 — é a dependência de uma métrica de desempenho
única e agregada por modelo, sem mecanismo para capturar variação de
desempenho por subdomínio ou regime de entrada, o que este trabalho aborda
via profiling por regime meteorológico (Capítulo 3, Etapa 5).

## 2.5 Otimização baseada em Set Covering

O problema de Cobertura de Conjuntos (*Set Cover*) — dado um universo de
elementos e uma coleção de subconjuntos, encontrar o menor número de
subconjuntos cuja união cobre todo o universo — está entre os 21 problemas
originalmente provados NP-completos por Karp [14]. Sua variante
ponderada (*Weighted Set Cover*, onde cada subconjunto tem um custo e
busca-se minimizar o custo total da cobertura) admite um algoritmo guloso
clássico com garantia de aproximação logarítmica $H(d)$, onde $d$ é o
tamanho do maior subconjunto, devido a Chvátal [15] — algoritmo usado como
base do solver heurístico deste trabalho (`solve_greedy`, Capítulo 3). A
formulação de cobertura de conjuntos tem sido aplicada a problemas de
seleção de modelos em sistemas de múltiplos classificadores, tipicamente
como forma de garantir que cada categoria ou subconjunto de instâncias do
problema seja atendido por pelo menos um membro do ensemble/comitê
selecionado. Este trabalho estende esse arcabouço em duas direções não
usuais na literatura de poda de ensemble consultada: (i) generalização para
**Set *Multi*cover** — exigindo $r_u \geq 2$ especificamente em regimes
meteorológicos classificados como críticos (Capítulo 3, Etapa 6), em vez do
requisito uniforme $r_u{=}1$ da formulação clássica; e (ii) ponderação dos
próprios elementos do universo (os regimes) por severidade meteorológica,
não apenas dos subconjuntos (os modelos) por custo — uma dupla ponderação
(custo do modelo + peso do regime) ausente tanto na formulação clássica de
Chvátal [15] quanto nas aplicações de Set Cover a seleção de modelos
revisadas nesta seção. O Capítulo 6 (Ablation Study) isola empiricamente a
contribuição de cada uma dessas duas extensões frente ao caso particular
não ponderado (*Traditional-Set-Cover*, um dos sete baselines de
comparação).

## 2.6 Verificação meteorológica espacial

Métricas de erro pixel a pixel como PSNR e SSIM, padrão em visão
computacional, penalizam duplamente pequenos erros de deslocamento espacial
de um eco de precipitação (o chamado *double penalty problem*), mesmo
quando a estrutura espacial prevista é qualitativamente correta — um
problema bem documentado na literatura de verificação de previsão
meteorológica. O Fractions Skill Score (FSS), introduzido por Roberts e
Lean [17], mitiga esse problema comparando frações de excedência de um
limiar de intensidade dentro de janelas espaciais locais, em vez de
comparação ponto a ponto — motivo pelo qual este trabalho o inclui como
métrica de "preservação meteorológica" (Capítulo 3, Etapa 5), complementar
a PSNR/SSIM. O Critical Success Index (CSI), formalizado por Schaefer [18]
no contexto de avaliação de alertas meteorológicos do National Weather
Service dos EUA, mede a razão entre acertos e a soma de acertos, falsos
alarmes e omissões sobre um evento binarizado por limiar — adotado neste
trabalho para quantificar especificamente a preservação de núcleos de eco
crítico (Capítulo 3, Etapa 5), com a ressalva, já discutida por Schaefer
[18], de que o CSI não é uma medida não enviesada de habilidade preditiva,
sendo proporcional à frequência de ocorrência do evento — um cuidado
metodológico relevante dado que os regimes meteorológicos deste trabalho
têm frequência de ocorrência (representação na amostra) desigual entre si
(Capítulo 7, Seção 7.3).

---

*A lista de referências numeradas [1]–[18] citadas neste capítulo encontra-se
consolidada ao final do Capítulo 1.*
