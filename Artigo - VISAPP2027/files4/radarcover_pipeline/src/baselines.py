"""
baselines.py — conjunto de baselines para comparação (Etapa 9 / Seção
"Comparison with baselines" do artigo), conforme listado no diagrama:

  1. Best single model     — modelo isolado com maior PSNR médio ponderado
  2. Full ensemble         — todos os modelos do pool (custo máximo)
  3. Random pruning        — subconjunto aleatório do mesmo tamanho da
                              solução do Multicover
  4. Top-k                 — os k modelos com melhor PSNR médio (sem
                              considerar cobertura de regimes)
  5. Diversity-based pruning — k modelos escolhidos para maximizar
                              diversidade de perfis regime-a-regime
                              (evita redundância de "pontos fortes")
  6. Pareto pruning        — fronteira de Pareto custo x qualidade; seleciona
                              os modelos não-dominados
  7. Traditional Set Cover — Multicover com r_u=1 para todos os regimes
                              (sem multi-cobertura de regimes críticos) e
                              sem ponderação por severidade — é o caso
                              particular "não ponderado" do método proposto
"""

from __future__ import annotations
from typing import Dict, List
import numpy as np

from coverage import CoverageProblem, solve_greedy


def best_single_model(profile_entries, model_names: List[str], regime_weight_by_id: Dict[int, float]):
    scores = {}
    for m in model_names:
        vals, ws = [], []
        for e in profile_entries:
            if e.model == m:
                vals.append(e.psnr)
                ws.append(regime_weight_by_id.get(e.regime_id, 1.0))
        scores[m] = float(np.average(vals, weights=ws)) if vals else -1e9
    best = max(scores, key=scores.get)
    return {"selected": [best], "status": "Best-Single", "unmet_regimes": None, "total_cost": None}


def full_ensemble(model_names: List[str]):
    return {"selected": list(model_names), "status": "Full-Ensemble", "unmet_regimes": None, "total_cost": None}


def random_pruning(model_names: List[str], k: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    selected = [str(m) for m in rng.choice(model_names, size=min(k, len(model_names)), replace=False)]
    return {"selected": selected, "status": "Random-Pruning", "unmet_regimes": None, "total_cost": None}


def top_k(profile_entries, model_names: List[str], regime_weight_by_id: Dict[int, float], k: int):
    scores = {}
    for m in model_names:
        vals, ws = [], []
        for e in profile_entries:
            if e.model == m:
                vals.append(e.psnr)
                ws.append(regime_weight_by_id.get(e.regime_id, 1.0))
        scores[m] = float(np.average(vals, weights=ws)) if vals else -1e9
    ranked = sorted(scores, key=scores.get, reverse=True)
    return {"selected": ranked[:k], "status": "Top-K", "unmet_regimes": None, "total_cost": None}


def diversity_based_pruning(profile_entries, model_names: List[str], regime_ids: List[int], k: int,
                             seed: int = 0):
    """Constrói o vetor de perfil PSNR-por-regime de cada modelo e agrupa em
    k clusters (KMeans); seleciona o modelo mais central de cada cluster —
    maximiza diversidade de comportamento entre regimes."""
    from sklearn.cluster import KMeans

    lookup = {(e.model, e.regime_id): e.psnr for e in profile_entries}
    X = np.array([[lookup.get((m, u), 0.0) for u in regime_ids] for m in model_names])
    k_eff = min(k, len(model_names))
    km = KMeans(n_clusters=k_eff, random_state=seed, n_init=10).fit(X)
    selected = []
    for c in range(k_eff):
        idx_in_cluster = np.where(km.labels_ == c)[0]
        center = km.cluster_centers_[c]
        dists = np.linalg.norm(X[idx_in_cluster] - center, axis=1)
        chosen = idx_in_cluster[np.argmin(dists)]
        selected.append(model_names[chosen])
    return {"selected": selected, "status": "Diversity-Based", "unmet_regimes": None, "total_cost": None}


def pareto_pruning(profile_entries, model_names: List[str], cost: np.ndarray,
                    regime_weight_by_id: Dict[int, float]):
    """Seleciona modelos não-dominados no plano (custo, -qualidade)."""
    scores = {}
    for m in model_names:
        vals, ws = [], []
        for e in profile_entries:
            if e.model == m:
                vals.append(e.psnr)
                ws.append(regime_weight_by_id.get(e.regime_id, 1.0))
        scores[m] = float(np.average(vals, weights=ws)) if vals else -1e9

    pts = [(cost[i], -scores[m]) for i, m in enumerate(model_names)]  # minimizar ambos
    selected = []
    for i, (c_i, q_i) in enumerate(pts):
        dominated = False
        for j, (c_j, q_j) in enumerate(pts):
            if j == i:
                continue
            if c_j <= c_i and q_j <= q_i and (c_j < c_i or q_j < q_i):
                dominated = True
                break
        if not dominated:
            selected.append(model_names[i])
    return {"selected": selected, "status": "Pareto-Pruning", "unmet_regimes": None, "total_cost": None}


def traditional_set_cover(problem: CoverageProblem, max_models: int | None = None):
    """Caso particular do método proposto: r_u=1 para todos os regimes
    (sem multicoverage) e pesos de regime uniformes (sem ponderação por
    severidade) — isola a contribuição específica dessas duas escolhas de
    design (usado também no Ablation Study, Capítulo 6)."""
    unweighted = CoverageProblem(
        model_names=problem.model_names,
        regime_ids=problem.regime_ids,
        alpha=problem.alpha,
        cost=problem.cost,
        coverage_req=np.ones_like(problem.coverage_req),
        regime_weight=np.ones_like(problem.regime_weight),
    )
    result = solve_greedy(unweighted, max_models=max_models)
    result["status"] = "Traditional-Set-Cover"
    return result
