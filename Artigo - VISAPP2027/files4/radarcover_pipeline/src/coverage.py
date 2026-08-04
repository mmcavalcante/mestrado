"""
coverage.py — Etapas 6 e 7: Coverage Matrix Definition + Weighted Set
Multicover Optimization.

Etapa 6 — Matriz de cobertura
------------------------------
alpha[u, j] = 1 se o modelo j "cobre satisfatoriamente" o regime u, isto é,
se suas métricas de profiling (Etapa 5) excedem thresholds mínimos de
qualidade E de preservação meteorológica simultaneamente. Caso contrário 0.

Etapa 7 — Weighted Set Multicover (núcleo da contribuição RadarCover)
-----------------------------------------------------------------------
Formulação (ILP):

    minimize    sum_j  cost_j * x_j
    subject to  sum_j  alpha[u,j] * x_j  >=  r_u        para todo regime u
                x_j in {0,1}

onde:
  - x_j = 1 se o modelo j é selecionado para o ensemble podado
  - cost_j = custo computacional do modelo j (ex.: combinação normalizada de
    latência e nº de parâmetros)
  - r_u = requisito de cobertura do regime u (>=2 se crítico, senão >=1)

Diferente do Set Cover clássico (r_u = 1 para todos), o Multicover permite
exigir redundância (multi-cobertura) em regimes críticos — ecos intensos que
não podem depender de um único modelo (ver Etapa 6: "Critical echoes may
require multicoverage").

O termo "Weighted" refere-se tanto ao custo por modelo quanto ao peso de
importância meteorológica w_u aplicado sobre a folga de cobertura em regimes
não plenamente cobertos (usado na relaxação/penalização quando a solução
exata não é viável dado o orçamento de modelos).

Dois solvers são fornecidos:
  1. `solve_ilp`   — exato, via PuLP/CBC (recomendado para poucos modelos x
                      poucos regimes, como no pool de 12 candidatos x
                      30-50 regimes deste pipeline).
  2. `solve_greedy`— heurística gulosa clássica para Set Multicover
                      (aproximação com fator de garantia H(max r_u), útil
                      como baseline e para escalar caso o pool cresça).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class CoverageProblem:
    model_names: List[str]
    regime_ids: List[int]
    alpha: np.ndarray          # shape (U, M) binário
    cost: np.ndarray           # shape (M,)
    coverage_req: np.ndarray   # shape (U,) inteiro >=1
    regime_weight: np.ndarray  # shape (U,) float (peso de severidade)


def build_coverage_matrix(profile_entries, model_names: List[str], regime_ids: List[int],
                           psnr_thresh: float = 20.0, fss_thresh: float = 0.5,
                           csi_thresh: float = 0.3) -> np.ndarray:
    """Constrói alpha[u,j] a partir das entradas de profiling (Etapa 5)."""
    lookup = {(e.model, e.regime_id): e for e in profile_entries}
    U, M = len(regime_ids), len(model_names)
    alpha = np.zeros((U, M), dtype=int)
    for ui, u in enumerate(regime_ids):
        for mi, m in enumerate(model_names):
            e = lookup.get((m, u))
            if e is None:
                continue
            ok = (e.psnr >= psnr_thresh) and (e.fss >= fss_thresh) and (e.csi >= csi_thresh)
            alpha[ui, mi] = int(ok)
    return alpha


def compute_model_cost(profile_entries, model_names: List[str]) -> np.ndarray:
    """Custo por modelo = média normalizada de (latência, nº parâmetros)
    entre todos os regimes onde foi perfilado — usado como 'cost_j' no
    Multicover (Etapa 7: 'Minimize cost and ensemble size')."""
    by_model_lat = {}
    by_model_par = {}
    for e in profile_entries:
        by_model_lat.setdefault(e.model, []).append(e.latency_ms)
        by_model_par.setdefault(e.model, []).append(e.n_params)
    lat = np.array([np.mean(by_model_lat.get(m, [1.0])) for m in model_names])
    par = np.array([np.mean(by_model_par.get(m, [1.0])) for m in model_names])
    lat_n = lat / (lat.max() + 1e-9)
    par_n = par / (par.max() + 1e-9)
    cost = 0.5 * lat_n + 0.5 * par_n
    return cost


def build_problem(profile_entries, model_names: List[str], regime_ids: List[int],
                   regime_weight: np.ndarray, critical_mask: np.ndarray,
                   psnr_thresh=20.0, fss_thresh=0.5, csi_thresh=0.3) -> CoverageProblem:
    alpha = build_coverage_matrix(profile_entries, model_names, regime_ids,
                                   psnr_thresh, fss_thresh, csi_thresh)
    cost = compute_model_cost(profile_entries, model_names)
    coverage_req = np.where(critical_mask, 2, 1).astype(int)
    return CoverageProblem(model_names, regime_ids, alpha, cost, coverage_req, regime_weight)


# --------------------------------------------------------------------------- #
# Solver 1: ILP exato (PuLP)
# --------------------------------------------------------------------------- #

def solve_ilp(problem: CoverageProblem, max_models: int | None = None,
              allow_relaxation: bool = True, penalty_scale: float = 10.0):
    import pulp

    U, M = problem.alpha.shape
    prob = pulp.LpProblem("WeightedSetMulticover", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{j}", cat="Binary") for j in range(M)]
    slack = [pulp.LpVariable(f"slack_{u}", lowBound=0, cat="Integer") for u in range(U)] \
        if allow_relaxation else None

    cost_term = pulp.lpSum(problem.cost[j] * x[j] for j in range(M))
    if allow_relaxation:
        penalty_term = pulp.lpSum(penalty_scale * problem.regime_weight[u] * slack[u] for u in range(U))
        prob += cost_term + penalty_term
    else:
        prob += cost_term

    for u in range(U):
        covered = pulp.lpSum(int(problem.alpha[u, j]) * x[j] for j in range(M))
        if allow_relaxation:
            prob += covered + slack[u] >= int(problem.coverage_req[u])
        else:
            prob += covered >= int(problem.coverage_req[u])

    if max_models is not None:
        prob += pulp.lpSum(x) <= max_models

    solver = pulp.PULP_CBC_CMD(msg=0)
    prob.solve(solver)

    selected = [problem.model_names[j] for j in range(M) if pulp.value(x[j]) > 0.5]
    status = pulp.LpStatus[prob.status]
    unmet = None
    if allow_relaxation:
        unmet = {problem.regime_ids[u]: pulp.value(slack[u]) for u in range(U) if pulp.value(slack[u]) > 0.5}
    return {"selected": selected, "status": status, "unmet_regimes": unmet,
            "total_cost": sum(problem.cost[problem.model_names.index(m)] for m in selected)}


# --------------------------------------------------------------------------- #
# Solver 2: heurística gulosa (Weighted Greedy Set Multicover)
# --------------------------------------------------------------------------- #

def solve_greedy(problem: CoverageProblem, max_models: int | None = None):
    """
    Algoritmo guloso clássico adaptado para multicover ponderado: a cada
    iteração seleciona o modelo que maximiza
        (soma ponderada de cobertura ainda faltante que ele resolve) / custo
    até que todos os requisitos de cobertura sejam satisfeitos ou o
    orçamento de modelos (`max_models`) se esgote.
    """
    U, M = problem.alpha.shape
    remaining = problem.coverage_req.copy().astype(float)
    selected = []
    available = set(range(M))

    while remaining.max() > 0 and available:
        if max_models is not None and len(selected) >= max_models:
            break
        best_j, best_score = None, -1.0
        for j in available:
            gain = np.minimum(problem.alpha[:, j], np.ceil(remaining)).clip(min=0)
            weighted_gain = float((gain * problem.regime_weight).sum())
            score = weighted_gain / (problem.cost[j] + 1e-6)
            if score > best_score:
                best_score, best_j = score, j
        if best_j is None or best_score <= 0:
            break
        selected.append(problem.model_names[best_j])
        remaining -= problem.alpha[:, best_j]
        remaining = remaining.clip(min=0)
        available.discard(best_j)

    unmet = {problem.regime_ids[u]: remaining[u] for u in range(U) if remaining[u] > 0}
    total_cost = sum(problem.cost[problem.model_names.index(m)] for m in selected)
    return {"selected": selected, "status": "Greedy-Heuristic", "unmet_regimes": unmet,
            "total_cost": total_cost}
