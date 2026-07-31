# 7. Limitations

Esta seção discute as limitações metodológicas do RadarCover identificadas
tanto por desenho quanto — de forma mais concreta — pelas decisões de
implementação e pelos resultados obtidos nos Capítulos 4 a 6. Cada
limitação é acompanhada, quando aplicável, da evidência empírica que a
motivou e do encaminhamento previsto para a versão de escala plena do
trabalho.

## 7.1 Degradação simulada

O par (LR, HR) usado em treino e avaliação é sintetizado artificialmente a
partir da imagem HR real via desfoque gaussiano + subamostragem + ruído
gaussiano aditivo (Capítulo 4, Seção 4.3), e não a partir de pares
LR/HR medidos independentemente por instrumentos de resolução distinta.
Essa é uma limitação estrutural comum a praticamente toda a literatura de
super-resolução de imagens de radar (a ausência de radares de resolução
mais alta operando simultaneamente sobre a mesma área torna pares reais
inviáveis), mas implica que:

- o modelo aprende a reverter especificamente o operador de degradação
  escolhido (blur gaussiano + ruído gaussiano), podendo generalizar mal
  para degradações reais do sensor de origem, que envolvem efeitos não
  modelados aqui (atenuação por precipitação intensa entre o radar e o
  alvo, *ground clutter*, *beam blockage* topográfico, e artefatos de
  reamostragem da projeção polar→cartesiana mencionados no próprio README
  do dataset);
- os parâmetros de degradação ($\sigma_{\text{blur}}=1{,}0$,
  $\sigma_{\text{ruído}}=0{,}02$) foram escolhidos por inspeção visual, não
  calibrados contra uma caracterização estatística do sensor real do IPMet.

## 7.2 Seleção de thresholds

A matriz de cobertura (Etapa 6) depende de três limiares definidos
manualmente — PSNR ≥ 18 dB, FSS ≥ 0,30, CSI ≥ 0,15 (Capítulo 4) — que
determinam quais pares (modelo, regime) são considerados "cobertos". Esses
valores não foram calibrados por validação cruzada nem justificados por
requisitos operacionais externos (ex.: limiares de aceitação usados em
sistemas de alerta meteorológico real); foram escolhidos para produzir uma
matriz de cobertura com esparsidade razoável (~27% de células cobertas,
Etapa 6) na escala de demonstração. O Capítulo 6 mostrou concretamente a
consequência prática dessa escolha: com apenas 12 candidatos, os thresholds
atuais deixam pouca margem de decisão, e três dos cinco componentes
avaliados no Ablation Study (pesos de regime, uso de métricas
meteorológicas, sensibilidade a custo) não alteraram a seleção final — um
sintoma direto de que a combinação (nº de candidatos, thresholds) está perto
de um regime degenerado, onde poucas soluções são factíveis. Uma análise de
sensibilidade sistemática dos thresholds (variando cada um independentemente
e observando o tamanho do conjunto factível) é necessária antes da versão
final e não foi realizada aqui.

## 7.3 Qualidade do clustering (construção de regimes)

Os regimes meteorológicos são construídos via K-Means sobre 8 descritores
escalares (Capítulo 3), com o número de clusters fixado a priori (16 na
demonstração, 30–50 planejado para a versão final) em vez de determinado
por um critério de qualidade de clustering (ex.: método do cotovelo,
coeficiente de silhueta, Gap Statistic). K-Means também assume clusters
aproximadamente esféricos em distância euclidiana no espaço padronizado de
features, uma suposição não verificada para descritores como "número de
componentes conectados", que é uma contagem inteira de cauda longa, e
"distância ao centro do radar", que tem uma relação não linear conhecida
com a qualidade do sinal (a resolução espacial do radar degrada com a
distância, conforme observado no próprio README do dataset). A execução de
demonstração obteve uma distribuição de regimes visivelmente desbalanceada
(o maior regime concentrou ~28% dos patches em uma clusterização exploratória
preliminar, Seção de testes do Capítulo 3), o que pode enviesar os pesos de
severidade calculados por regime (Etapa 4) em favor de padrões majoritários
comuns, exigindo validação adicional da estabilidade do clustering (ex.:
K-Means com múltiplas inicializações e medição de consistência entre
execuções) antes de resultados definitivos.

## 7.4 Generalização do dataset

Três restrições de generalização identificadas diretamente na Seção 4.1:

1. **Escopo geográfico único** — todos os dados vêm de um único radar
   (Bauru, SP), portanto conclusões sobre desempenho relativo dos modelos e
   regimes construídos podem não se transferir a radares com características
   distintas de terreno, clima ou hardware.
2. **Escopo temporal restrito** — o subconjunto público efetivamente
   utilizado cobre um único mês (31/12/2023–31/01/2024), não o intervalo de
   dois anos (jan/2022–jan/2024) do dataset completo mencionado na origem.
   Isso significa que a diversidade de regimes meteorológicos observável
   (ex.: eventos convectivos de verão vs. sistemas frontais de inverno) está
   sub-representada, e os "regimes" construídos na Etapa 4 refletem a
   variabilidade de um único período sazonal (verão no hemisfério sul).
3. **Amostragem reduzida na demonstração** — apenas 90 dos 4.014 frames
   disponíveis foram usados nas execuções deste artigo (Capítulos 4-6), por
   restrição de tempo/recursos computacionais do ambiente de execução, não
   por uma decisão de desenho experimental. A escala plena deve utilizar o
   dataset completo.

## 7.5 Custo de treinamento

Por restrição do ambiente de execução (1 vCPU, sem GPU, disco insuficiente
para as dependências CUDA do PyTorch — Capítulo 4, Seção 4.4), os 12
modelos candidatos foram treinados por apenas 60 iterações cada sobre um
motor de convolução implementado em NumPy puro, e não sobre um framework de
deep learning com aceleração por GPU e um orçamento de treino realista
(tipicamente milhares a dezenas de milhares de iterações na literatura de
super-resolução). O Capítulo 5 já discutiu a consequência direta: as
arquiteturas mais profundas do pool (EDSR-Lite, SwinIR-Light-Lite,
CRMN-Lite) apresentaram SSIM sistematicamente inferior às mais rasas
(SRCNN, ESPCN), um padrão mais consistente com sub-treinamento do que com
mérito arquitetural real. Adicionalmente, o SwinIR-Light-Lite implementado
aqui **não contém mecanismo de atenção real** — foi aproximado por blocos
convolucionais residuais mais profundos, uma simplificação declarada no
Capítulo 3 por inviabilidade de implementar self-attention eficientemente em
NumPy puro sobre imagens completas. Isso significa que os resultados
associados a essa arquitetura específica não devem ser interpretados como
representativos do SwinIR-Light da literatura.

## 7.6 Ausência de testes de significância estatística

Como consequência direta das limitações de escala (7.4, 7.5), os resultados
comparativos do Capítulo 5 e do Ablation Study do Capítulo 6 são reportados
como médias pontuais, sem intervalos de confiança ou testes de hipótese
(Capítulo 4, Seção 4.7). Com poucas amostras por célula da tabela de
profiling (regime × modelo), testes pareados teriam potência estatística
insuficiente para distinguir diferenças pequenas (ex.: RadarCover vs. Top-K,
que empataram exatamente na configuração testada) de ruído amostral. Essa é
considerada a limitação mais significativa para a validade das conclusões
comparativas específicas (não da arquitetura do método), e sua resolução
depende diretamente da execução em escala plena.

## 7.7 Síntese

As limitações acima compartilham uma causa raiz comum: as execuções
reportadas neste artigo usam uma escala de demonstração (90 frames, 16
regimes, 12 candidatos treinados por 60 iterações em CPU) deliberadamente
reduzida para caber nas restrições do ambiente de execução disponível, e
não a escala pretendida do estudo completo (4.014 frames, 30–50 regimes,
treino em GPU). O pipeline de código (Capítulos 3-6) foi construído para que
essa transição de escala não exija mudança estrutural — apenas alteração de
parâmetros de configuração (`configs/default.yaml`, comentários
`ESCALAR PARA PAPER`) e, no caso do treinamento dos modelos, a porta da
interface `BaseSRModel` do motor NumPy para uma implementação equivalente em
PyTorch. As demais limitações (degradação simulada, decodificação de
refletividade por proxy de luminância em vez da paleta de cores oficial do
IPMet — Capítulo 3 — e escopo geográfico de radar único) são estruturais ao
problema e ao dataset disponível, devendo ser reconhecidas explicitamente
como fronteiras de validade do estudo, e não resolvidas apenas por mais
poder computacional.
