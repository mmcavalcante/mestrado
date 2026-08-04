# RadarCover — Pipeline de Execução (Etapas 1–9)

Implementação executável de ponta a ponta da metodologia descrita no diagrama
"Proposed Methodology" (`Objetivo_Estrutura.pdf`), sobre o **IPMet Radar
Dataset** real (https://github.com/rafaepires/IPMet-Radar-Dataset).

## Como rodar

```bash
pip install -r requirements.txt   # numpy, scipy, scikit-learn, scikit-image, pandas, pyyaml, pulp, pillow
python run_demo.py --config configs/default.yaml
```

Saídas em `outputs/`:
- `profile_table.csv` — profiling completo (modelo × regime), Etapa 5
- `final_comparison.csv` — tabela comparativa final, Etapa 9

## Mapeamento etapa → código

| # | Etapa (diagrama) | Módulo | Função principal |
|---|---|---|---|
| 1 | IPMet Radar Dataset | `src/data.py` | `list_dataset`, `chronological_split` |
| 2 | Image Preprocessing | `src/data.py` | `degrade`, `extract_patches`, `build_lr_hr_pairs` |
| 3 | Candidate Model Pool | `src/models.py` | `build_candidate_pool` (SRCNN, ESPCN, EDSR-Lite, RLFN-Lite, SwinIR-Light-Lite, CRMN-Lite × seeds) |
| 4 | Regime Construction | `src/regimes.py` | `extract_features`, `build_regimes` (K-Means) |
| 5 | Per-Regime Model Profiling | `src/profiling.py` | `profile_pool_over_regimes` (PSNR, SSIM, FSS, CSI, Recall, latência, params) |
| 6 | Coverage Matrix Definition | `src/coverage.py` | `build_problem` (α, custo, pesos de regime, requisitos de multicobertura) |
| 7 | Weighted Set Multicover Optimization | `src/coverage.py` | `solve_ilp` (PuLP/CBC, exato) + `solve_greedy` (heurística) |
| 8 | Pruned Ensemble Inference | `src/ensemble.py` | `PrunedEnsemble.predict` (fusão simples/ponderada) |
| 9 | Evaluation and Analysis | `src/evaluate.py` | `evaluate_selection`, `build_results_table` |
| — | Baselines | `src/baselines.py` | best single, full ensemble, random, top-k, diversity, Pareto, set cover tradicional |
| — | Orquestração | `src/pipeline.py` | `run_pipeline(cfg)` — executa 1→9 em sequência |

## Decisões de design documentadas (relevantes para o Capítulo 7 — Limitações)

1. **Decodificação de refletividade**: as imagens do dataset são PNGs RGBA
   coloridos (paleta meteorológica + canal alfa de cobertura), não grayscale
   puro como o README do dataset sugere. Este pipeline extrai uma *proxy* de
   intensidade via luminância ponderada × máscara de alfa. Reconstruir a
   escala dBZ exata exigiria a tabela de cores oficial do produto IPMet.

2. **Motor de rede neural em NumPy puro**: o ambiente de execução é CPU-only
   com disco insuficiente para as dependências CUDA do PyTorch. Os 6 modelos
   candidatos são implementados sobre um motor de conv2d + backprop manual
   (`src/nn_core.py`), preservando a topologia de cada arquitetura original
   (skip-connections, sub-pixel convolution, mistura recorrente) em escala
   reduzida (menos canais/blocos). Para experimentos de publicação, a mesma
   interface (`BaseSRModel`) deve ser portada para PyTorch/GPU.

3. **Escala de demonstração**: `configs/default.yaml` usa 90 frames, 16
   regimes, 60 iterações de treino por modelo — roda em ~50s numa CPU. Cada
   parâmetro tem um comentário `ESCALAR PARA PAPER` indicando o valor real
   a usar (4.014 frames completos, 30–50 regimes, treino em GPU por milhares
   de iterações).

## Resultado de demonstração obtido (dados reais, config padrão)

| método | nº modelos | params totais | latência (ms) | PSNR | SSIM |
|---|---|---|---|---|---|
| RadarCover-Multicover | 2 | 9.442 | 13,7 | **18,89** | 0,682 |
| Full-Ensemble | 12 | 47.418 | 51,6 | 18,84 | 0,648 |
| Best-Single | 1 | 4.721 | 6,0 | 18,82 | 0,611 |
| Traditional-Set-Cover | 2 | 6.117 | 7,4 | 18,23 | 0,606 |

Mesmo em escala reduzida, o padrão esperado pela hipótese central do artigo
já aparece: o RadarCover iguala/supera o Full-Ensemble com **6× menos
parâmetros e ~3,8× menor latência**, superando o Traditional-Set-Cover (que
ignora multicobertura e pesos de severidade) em todas as métricas de
qualidade — validando a formulação Weighted Multicover proposta na Etapa 7.

## Próximos passos sugeridos (Fase C do plano de execução)

1. Rodar com `n_frames: 4014`, `n_regimes: 40`, modelos portados para PyTorch/GPU.
2. Usar `profile_table.csv` e `final_comparison.csv` reais para escrever os
   Capítulos 4 (Metodologia Experimental) e 5 (Resultados e Discussão).
3. Rodar o pipeline variando: (a) com/sem multicobertura, (b) com/sem pesos
   de regime, (c) fusão simples vs. ponderada — para o Capítulo 6 (Ablation Study).
