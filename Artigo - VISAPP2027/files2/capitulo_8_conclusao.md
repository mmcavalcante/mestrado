# 8. Conclusion and Future Work

## 8.1 Síntese

Este trabalho propôs o **RadarCover**, um pipeline de nove etapas para
construção de ensembles compactos e conscientes de custo para restauração
de imagens de radar meteorológico, cujo núcleo é uma formulação de
**Weighted Set Multicover** — generalização do problema clássico de
cobertura de conjuntos que pondera regimes meteorológicos por severidade e
permite exigir cobertura redundante especificamente em condições críticas.
A hipótese central, de que a seleção de ensemble informada por profiling
por regime meteorológico (em vez de uma métrica de desempenho agregada
única) produz ensembles mais eficientes sem perda de qualidade, foi testada
empiricamente sobre dados reais do IPMet Radar Dataset [4], com resultados
que respondem diretamente às três perguntas de pesquisa formuladas no
Capítulo 1:

- **RQ1** (RadarCover vs. Full-Ensemble): confirmada. O ensemble
  selecionado igualou/superou a qualidade do ensemble completo (PSNR +0,06
  dB, SSIM +0,034) usando 5× menos modelos, 5× menos parâmetros e 3,8×
  menos latência (Capítulo 5).
- **RQ2** (multicobertura e ponderação vs. Set Cover tradicional):
  confirmada em qualidade — o RadarCover superou seu próprio caso
  particular não ponderado em todas as métricas de qualidade e
  meteorológicas (Capítulo 5), embora o Ablation Study (Capítulo 6) tenha
  mostrado que, na escala de demonstração testada, apenas o componente de
  multicobertura produziu diferenciação robusta isolada — pesos de regime,
  métricas meteorológicas na cobertura e sensibilidade a custo não
  diferenciaram a seleção final nessa escala reduzida, um resultado
  atribuído à esparsidade da matriz de cobertura com apenas 12 candidatos
  (Capítulo 7), e não a uma refutação desses componentes.
- **RQ3** (seleção informada vs. baselines não informados): confirmada. Os
  dois baselines que ignoram desempenho absoluto na seleção (poda aleatória
  e poda por diversidade pura) apresentaram a maior queda de qualidade de
  toda a comparação (Capítulo 5).

## 8.2 Contribuições confirmadas

Revisitando as contribuições listadas no Capítulo 1: a formulação de
Weighted Set Multicover (Contribuição 1) e o protocolo de construção de
regimes meteorológicos (Contribuição 2) foram implementados e validados
empiricamente; o pipeline executável de ponta a ponta (Contribuição 3)
rodou sobre dados reais do dataset público em ≈50 segundos (execução
principal) e ≈43 segundos (estudo de ablação) em hardware modesto (1 vCPU,
sem GPU), demonstrando viabilidade prática mesmo fora de um ambiente de
pesquisa com recursos de GPU dedicados; e a discussão de limitações
fundamentada em evidência (Contribuição 4) identificou, entre outros
achados, uma discrepância metodológica relevante entre a documentação
pública do dataset e sua composição temporal real (Capítulo 4, Seção 4.1) —
um achado com valor para qualquer trabalho futuro que venha a utilizar o
mesmo subconjunto de dados.

## 8.3 Limitações centrais a superar

Como detalhado no Capítulo 7, os resultados reportados foram obtidos em
escala de demonstração (90 de 4.014 frames disponíveis, 16 de 30–50 regimes
planejados, 12 modelos treinados por 60 iterações em um motor de CNN em
NumPy puro por restrição de hardware) e não incluem testes de significância
estatística formal. Essas limitações são de escala de execução, não de
desenho metodológico — o código do pipeline (Capítulos 3–6) foi construído
para que a transição à escala plena exija apenas alteração de parâmetros de
configuração e a porta dos modelos candidatos para um framework de deep
learning acelerado por GPU, sem mudança estrutural na formulação de
regimes, cobertura ou otimização.

## 8.4 Trabalhos futuros

**Roteamento dinâmico de modelos.** A formulação atual seleciona um
ensemble fixo, aplicado uniformemente a toda inferência subsequente. Uma
extensão natural é o roteamento dinâmico: classificar cada nova imagem de
radar em um regime (usando o mesmo classificador de clustering da Etapa 4)
e invocar, em tempo de inferência, apenas o subconjunto de modelos
selecionados que cobrem aquele regime específico — reduzindo ainda mais o
custo computacional médio em relação à execução do ensemble podado completo
a cada quadro.

**Sequências temporais de radar.** Este trabalho trata cada quadro de
radar como uma imagem estática independente. Given a alta taxa de
amostragem temporal do IPMet (varreduras a cada 5–8 minutos) e a forte
autocorrelação temporal de sistemas de precipitação, uma extensão relevante
é incorporar contexto temporal — seja como entrada adicional aos modelos
candidatos (restauração espaço-temporal), seja como sinal adicional na
construção de regimes (ex.: regime definido também pela trajetória de
evolução do sistema, não apenas pelo estado instantâneo).

**Cobertura consciente de incerteza.** A matriz de cobertura atual
(Etapa 6) é binária, definida por thresholds fixos de qualidade. Uma
extensão natural é substituir a cobertura binária por uma formulação
probabilística — ex.: exigir que a probabilidade estimada de um modelo
atender ao threshold de qualidade em um regime exceda um nível de
confiança, propagando incerteza de estimação (naturalmente alta em regimes
com poucas amostras, como discutido no Capítulo 7) para a própria decisão
de seleção do Multicover, em vez de tratá-la como um valor determinístico.

**Datasets adicionais.** A validação em um único radar (Bauru, SP) e um
único mês de observações (Capítulo 7) limita a generalização das
conclusões específicas sobre desempenho relativo dos modelos e regimes.
Replicar o pipeline sobre radares de características distintas (diferente
terreno, clima, banda de frequência do radar) e sobre um período mais longo
do IPMet Radar Dataset completo (os 67.424 frames de janeiro/2022 a
janeiro/2024, dos quais o subconjunto público usado neste trabalho é uma
fração) é o passo mais direto para testar a robustez das conclusões diante
da diversidade sazonal e geográfica do problema.

## 8.5 Considerações finais

O RadarCover demonstra, com evidência empírica obtida sobre dados reais
ainda que em escala reduzida, que tratar a poda de ensemble como um
problema de cobertura de conjuntos ponderado e sensível a regime —
em vez de um ranqueamento por métrica agregada única — é uma direção
metodologicamente sólida e computacionalmente vantajosa para a restauração
de imagens de radar meteorológico. A confirmação definitiva de sua
vantagem sobre os métodos de poda estabelecidos na literatura (Capítulo 2)
depende da execução em escala plena delineada nas seções anteriores, para a
qual este trabalho entrega tanto os resultados preliminares quanto a
infraestrutura de código pronta para reprodução e extensão.
