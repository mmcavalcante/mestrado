# 5. Results and Discussion

## 5.1 Resultados individuais dos modelos candidatos

A Tabela 4.1 (Capítulo 4) mostra dispersão relevante de qualidade entre os
12 candidatos, mesmo em escala reduzida: PSNR médio varia de 13,98 dB
(SwinIR-Light-Lite#1) a 18,61 dB (SRCNN#1), um intervalo de 4,6 dB. Os
modelos rasos e mais simples (SRCNN, ESPCN) superam consistentemente as
variantes mais profundas (EDSR-Lite, SwinIR-Light-Lite, CRMN-Lite) neste
regime de treinamento — um padrão esperado dado o orçamento de treino muito
reduzido da execução de demonstração (60 iterações): arquiteturas mais
profundas têm mais parâmetros a ajustar e tipicamente exigem
significativamente mais iterações para convergir a partir de inicialização
aleatória, o que penaliza especificamente EDSR-Lite e SwinIR-Light-Lite
neste cenário. Isso é consistente com o SSIM notavelmente baixo dessas
arquiteturas (0,33–0,39 contra 0,60–0,64 de SRCNN) — sintoma característico
de sub-treinamento (a saída ainda não capturou textura fina, apenas a
tendência de baixa frequência herdada do upsample bicúbico residual). Esse
comportamento é revisitado no Capítulo 7 como limitação direta do orçamento
computacional da demonstração, e não deve ser interpretado como veredito
sobre o mérito relativo das arquiteturas em si.

RLFN-Lite, por outro lado, já demonstra na escala reduzida sua proposta de
desenho — mesmo com apenas 1.396 parâmetros (a arquitetura mais leve do
pool, empatada com CRMN-Lite), RLFN-Lite#1 atinge FSS de 0,889 e CSI de
0,616, superando modelos com 2× a 6× mais parâmetros (EDSR-Lite, CRMN-Lite),
e o faz com latência de apenas 1,59 ms — a menor do pool inteiro.

## 5.2 Desempenho da poda (RadarCover vs. baselines)

A Tabela 5.1 reproduz a comparação final da Etapa 9 (arquivo
`final_comparison.csv`), ordenada por PSNR ponderado por severidade de
regime.

**Tabela 5.1** — Comparação final entre o RadarCover (Weighted Set
Multicover) e os 7 baselines.

| Método | Nº modelos | Parâmetros totais | Latência (ms) | PSNR | SSIM | FSS | CSI | Recall |
|---|---|---|---|---|---|---|---|---|
| **RadarCover-Multicover** | **2** | **9.442** | **13,72** | **18,89** | **0,682** | 0,935 | 0,752 | 0,998 |
| Top-K | 2 | 9.442 | 13,89 | 18,89 | 0,682 | 0,935 | 0,752 | 0,998 |
| Full-Ensemble | 12 | 47.418 | 51,65 | 18,84 | 0,648 | 0,928 | 0,737 | 0,999 |
| Best-Single | 1 | 4.721 | 6,03 | 18,82 | 0,611 | 0,936 | 0,755 | 0,997 |
| Pareto-Pruning | 4 | 10.829 | 9,28 | 18,28 | 0,666 | 0,923 | 0,723 | 0,996 |
| Traditional-Set-Cover | 2 | 6.117 | 7,40 | 18,23 | 0,606 | 0,929 | 0,733 | 0,994 |
| Random-Pruning | 2 | 9.546 | 13,00 | 17,01 | 0,317 | 0,920 | 0,698 | 0,980 |
| Diversity-Based | 2 | 4.712 | 1,77 | 16,89 | 0,548 | 0,908 | 0,684 | 0,983 |

O RadarCover empata estatisticamente com Top-K nesta execução (mesma
seleção de modelos, `{SRCNN#0, SRCNN#1}` — resultado já discutido no
Capítulo 6: em pools pequenos e matrizes de cobertura esparsas, a
formulação de cobertura converge para a mesma dupla dominante que a simples
classificação por PSNR). O resultado mais relevante para a tese central do
artigo é a comparação **RadarCover vs. Full-Ensemble**: qualidade
equivalente ou ligeiramente superior (PSNR +0,06 dB, SSIM +0,034) usando
**5× menos modelos, 5× menos parâmetros e 3,8× menos latência**. Essa razão
de compressão é a evidência empírica direta da motivação do artigo
("compact ensemble... reducing inference cost", conforme o diagrama da
metodologia).

O RadarCover também supera o Traditional-Set-Cover (seu próprio caso
particular sem multicobertura nem ponderação por severidade) em todas as
métricas de qualidade — PSNR +0,66 dB, SSIM +0,076, FSS +0,006, CSI +0,019
— usando a mesma quantidade de modelos (2), porém com mais parâmetros
totais (9.442 vs. 6.117), pois a exigência de multicobertura em regimes
críticos empurra a seleção para modelos com melhor cobertura meteorológica,
não necessariamente os mais baratos. Isto quantifica diretamente o valor da
formulação ponderada e multicover proposta frente ao Set Cover clássico.

Random-Pruning e Diversity-Based, os dois baselines que **não usam
informação de desempenho** na seleção (aleatória e por clustering de perfil,
respectivamente), apresentam a maior queda de qualidade (PSNR 1,9–2,0 dB
abaixo do RadarCover; SSIM até 0,365 abaixo) — confirmando que a seleção
informada por profiling (Etapa 5) é indispensável, e que diversidade por si
só, sem considerar desempenho absoluto, não é um critério suficiente de
poda neste domínio.

## 5.3 Preservação meteorológica

As métricas meteorológicas (FSS, CSI, Recall) mostram um padrão distinto do
PSNR/SSIM: o Best-Single (um único modelo, SRCNN#1) obtém o **maior FSS e
CSI** de toda a tabela (0,936 e 0,755, respectivamente) apesar de ter o pior
SSIM entre os métodos de qualidade competitiva. Isso sugere que, para a
tarefa específica de preservar a *localização espacial* de ecos acima de um
limiar operacional (o que FSS e CSI medem), um único modelo bem calibrado
pode superar até ensembles maiores — a fusão de múltiplos modelos (mesmo
poucos) tende a suavizar bordas de eco, o que penaliza levemente CSI/FSS
mesmo quando melhora PSNR/SSIM médio. O RadarCover mantém FSS (0,935) e CSI
(0,752) muito próximos do Best-Single (diferença de apenas 0,001 e 0,003
respectivamente) — ou seja, incorpora o modelo com melhor comportamento
meteorológico (SRCNN#1 está entre os 2 selecionados) sem sacrificar
significativamente essa propriedade, enquanto ainda ganha em SSIM (+0,071)
sobre o modelo isolado.

## 5.4 Eficiência computacional e trade-off qualidade–latência

A Figura 5.1 posiciona todos os 8 métodos no plano latência×PSNR.
RadarCover e Top-K ocupam a região de melhor compromisso (alto PSNR, baixa
latência); Full-Ensemble está isolado no extremo de alta latência sem
ganho de qualidade proporcional — evidência visual direta de que a poda é
"gratuita" em termos de qualidade neste cenário, mas custa 3,8× menos para
executar.

![Trade-off qualidade x latência](outputs/quality_latency_tradeoff.png)

**Figura 5.1** — Trade-off latência (ms) × PSNR (dB) entre RadarCover e os
7 baselines. Fonte: `outputs/quality_latency_tradeoff.png`, gerado a partir
de `final_comparison.csv`.

## 5.5 Exemplo qualitativo

A Figura 5.2 ilustra a restauração de um patch de teste real (frame de
23/01/2024, escolhido por conter fração de eco acima da mediana da amostra,
12,7%), comparando entrada em baixa resolução (upsample bicúbico), a saída
do RadarCover (2 modelos), do Full-Ensemble (12 modelos) e do Best-Single
(1 modelo) contra o alvo de alta resolução real.

![Exemplo qualitativo de restauração](outputs/qualitative_example.png)

**Figura 5.2** — Exemplo de restauração em um patch de teste real do IPMet
Radar Dataset (64×64 px, fator de escala 2×). PSNR/SSIM medidos neste
exemplo específico: RadarCover 20,42 dB / 0,685; Full-Ensemble 20,15 dB /
0,658; Best-Single 20,40 dB / 0,690.

Neste exemplo pontual, o Best-Single obtém o maior SSIM, mas o RadarCover
apresenta o maior PSNR entre os três, reforçando o padrão observado na
agregação da Tabela 5.1: pequenos ensembles bem selecionados competem
diretamente com modelos únicos otimizados, com a vantagem estrutural de que
a seleção do RadarCover é derivada de um critério de cobertura por regime
(portanto potencialmente mais robusta a variação de condições
meteorológicas fora da amostra de teste específica), enquanto o Best-Single
é escolhido por uma única métrica agregada global.

## 5.6 Síntese

Os resultados de demonstração, embora obtidos em escala reduzida (Capítulo
7 detalha as limitações associadas), já reproduzem o padrão central proposto
pela metodologia: (i) modelos candidatos têm desempenho heterogêneo e
complementar entre regimes (5.1); (ii) a seleção via Weighted Set
Multicover iguala ou supera o ensemble completo com fração dos recursos
computacionais (5.2, 5.4); (iii) a formulação ponderada e com multicobertura
supera seu próprio caso particular não ponderado, o Set Cover tradicional
(5.2); e (iv) a preservação de propriedades meteorológicas (FSS/CSI) é
mantida próxima ao melhor modelo individual mesmo com a fusão de múltiplos
modelos (5.3). A confirmação estatisticamente robusta desses padrões, com
testes de significância formais, é reservada para a execução em escala
plena (ver Capítulo 4, Seção 4.7).
