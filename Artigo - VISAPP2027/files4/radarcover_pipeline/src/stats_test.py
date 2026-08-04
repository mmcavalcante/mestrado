"""
stats_test.py — Teste de Friedman + post-hoc de Nemenyi + diagrama de
diferença crítica (CD diagram), seguindo Demšar, J. (2006) "Statistical
Comparisons of Classifiers over Multiple Data Sets", JMLR 7, 1-30.

Papel no estudo (Seção 5, restruturação inspirada no OPFsembleR)
-------------------------------------------------------------------
OPFsembleR ranqueia os métodos de poda em CADA um dos N datasets e aplica
Friedman+Nemenyi sobre esses ranks para decidir quais métodos diferem
estatisticamente (Figura 6 do artigo-modelo). O RadarCover tem um único
dataset de origem (IPMet), mas U=16 regimes meteorológicos independentes
construídos na Etapa 4 — cada regime desempenha aqui o mesmo papel que um
"dataset" desempenha no artigo-modelo: uma unidade de comparação sobre a
qual cada método produz uma pontuação, e o conjunto de U pontuações por
método é o que o teste de Friedman analisa.

Valores críticos q_alpha (Tabela 5(a) de Demšar 2006, conferidos na fonte
primária antes de uso):
    k :  2      3      4      5      6      7      8      9      10
 q0.05: 1.960  2.343  2.569  2.728  2.850  2.949  3.031  3.102  3.164
 q0.10: 1.645  2.052  2.291  2.459  2.589  2.693  2.780  2.855  2.920
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

Q_ALPHA_005 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
               8: 3.031, 9: 3.102, 10: 3.164}
Q_ALPHA_010 = {2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589, 7: 2.693,
               8: 2.780, 9: 2.855, 10: 2.920}


@dataclass
class FriedmanNemenyiResult:
    k: int                      # número de métodos comparados
    n: int                      # número de blocos (regimes)
    avg_ranks: pd.Series        # rank médio por método (1 = melhor)
    chi2_stat: float
    chi2_p: float
    iman_davenport_F: float
    iman_davenport_p: float
    cd_005: float
    cd_010: float
    significant_at_005: bool


def friedman_nemenyi(matrix: pd.DataFrame, higher_is_better: bool = True) -> FriedmanNemenyiResult:
    """
    matrix: DataFrame com linhas = blocos (regimes) e colunas = métodos,
    valores = métrica de desempenho (ex.: PSNR) daquele método naquele
    regime. Sem valores ausentes (todo método deve ter sido avaliado em
    todo regime).
    """
    if matrix.isnull().values.any():
        raise ValueError("A matriz regime x método não pode conter valores ausentes para o teste de Friedman.")

    k = matrix.shape[1]
    n = matrix.shape[0]
    if k not in Q_ALPHA_005:
        raise ValueError(f"Tabela de valores críticos de Nemenyi não cobre k={k} métodos (suportado: 2-10).")

    # Ranks por linha (regime): rank 1 = melhor. scipy.stats.rankdata rank1=menor valor,
    # por isso rankeamos o negativo quando maior-é-melhor.
    vals = matrix.values if higher_is_better else -matrix.values
    ranks = np.apply_along_axis(lambda row: stats.rankdata(-row, method="average"), 1, vals)
    ranks_df = pd.DataFrame(ranks, index=matrix.index, columns=matrix.columns)
    avg_ranks = ranks_df.mean(axis=0).sort_values()

    # Estatística de Friedman (qui-quadrado) e teste embutido do scipy (equivalente)
    chi2_stat, chi2_p = stats.friedmanchisquare(*[matrix[c].values for c in matrix.columns])

    # Estatística F de Iman-Davenport (menos conservadora que a qui-quadrado pura)
    if (n * (k - 1) - chi2_stat) != 0:
        F_ID = (n - 1) * chi2_stat / (n * (k - 1) - chi2_stat)
        df1, df2 = k - 1, (k - 1) * (n - 1)
        p_ID = 1 - stats.f.cdf(F_ID, df1, df2)
    else:
        F_ID, p_ID = float("inf"), 0.0

    cd_005 = Q_ALPHA_005[k] * np.sqrt(k * (k + 1) / (6.0 * n))
    cd_010 = Q_ALPHA_010[k] * np.sqrt(k * (k + 1) / (6.0 * n))

    return FriedmanNemenyiResult(
        k=k, n=n, avg_ranks=avg_ranks, chi2_stat=float(chi2_stat), chi2_p=float(chi2_p),
        iman_davenport_F=float(F_ID), iman_davenport_p=float(p_ID),
        cd_005=float(cd_005), cd_010=float(cd_010),
        significant_at_005=bool(chi2_p < 0.05),
    )


def cliques_within_cd(avg_ranks: pd.Series, cd: float) -> List[List[str]]:
    """
    Agrupa métodos em 'cliques' de não-diferença estatística: cadeias
    maximais de métodos ordenados por rank médio em que membros
    consecutivos (e portanto todos os pares dentro da cadeia) distam menos
    que CD — mesma lógica das barras horizontais conectando métodos na
    Figura 6 do artigo-modelo.
    """
    ordered = avg_ranks.sort_values()
    names = list(ordered.index)
    values = list(ordered.values)
    cliques = []
    i = 0
    n = len(names)
    while i < n:
        j = i
        while j + 1 < n and values[j + 1] - values[i] < cd:
            j += 1
        if j > i:
            cliques.append(names[i:j + 1])
        i += 1
    # remove cliques que são subconjuntos de outra clique maior
    cliques = [c for c in cliques if not any(set(c) < set(other) for other in cliques)]
    # deduplica
    seen = set()
    uniq = []
    for c in cliques:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


def format_result_text(res: FriedmanNemenyiResult) -> str:
    sig = "significativa" if res.significant_at_005 else "NÃO significativa"
    lines = [
        f"Friedman χ²_F = {res.chi2_stat:.3f} (p = {res.chi2_p:.4f}); "
        f"Iman-Davenport F_F = {res.iman_davenport_F:.3f} (p = {res.iman_davenport_p:.4f}) "
        f"— diferença global {sig} entre os {res.k} métodos (N = {res.n} regimes).",
        f"CD (Nemenyi, α=0.05) = {res.cd_005:.3f} rank | CD (α=0.10) = {res.cd_010:.3f} rank.",
        "Ranks médios (1 = melhor): " + ", ".join(f"{m}={r:.2f}" for m, r in res.avg_ranks.items()),
    ]
    return "\n".join(lines)
