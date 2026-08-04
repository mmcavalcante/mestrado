"""
multi_budget.py — Reestruturação do estudo RadarCover tomando como modelo
o OPFsembleR (ICPR 2026): em vez de uma única tabela de comparação, roda o
método proposto e os baselines sob múltiplos ORÇAMENTOS de ensemble
(K = 2, 4, 6 modelos), produzindo uma tabela por orçamento (análogas às
Tabelas 1/2/3 do artigo-modelo para 10/30/50 classificadores) e um teste
estatístico de Friedman + post-hoc de Nemenyi com diagrama de diferença
crítica por orçamento (análogo à Figura 6 do artigo-modelo), tratando cada
um dos U regimes meteorológicos como uma unidade de comparação independente
(o papel que cada "dataset" desempenha no artigo-modelo).

Métodos que variam com o orçamento K: RadarCover-Multicover (max_models=K),
Top-K (k=K), Random-Pruning (k=K), Diversity-Based (k=K),
Traditional-Set-Cover (max_models=K).

Métodos constantes entre orçamentos (calculados uma única vez e
reaproveitados): Full-Ensemble (todos os modelos), Best-Single (sempre 1
modelo), Pareto-Pruning (definido pela própria fronteira de Pareto, não por
um orçamento externo) — mesmo papel que "Stacking" desempenha nas Tabelas
1/2/3 do artigo-modelo: uma referência que usa o pool inteiro, constante
entre as colunas de diferentes tamanhos de ensemble podado.
"""

from __future__ import annotations
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from pipeline import prepare_pipeline, log
from coverage import build_problem, solve_ilp, solve_greedy
from baselines import (best_single_model, full_ensemble, random_pruning, top_k,
                        diversity_based_pruning, pareto_pruning, traditional_set_cover)
from evaluate import evaluate_selection_per_regime, evaluate_selection
from stats_test import friedman_nemenyi, format_result_text
from cd_diagram import plot_cd_diagram


def _solve(problem, max_models):
    try:
        res = solve_ilp(problem, max_models=max_models)
    except Exception:
        res = solve_greedy(problem, max_models=max_models)
    return res


def _per_regime_to_matrix(per_regime: dict, metric: str, regime_ids) -> dict:
    return {u: per_regime.get(u, {}).get(metric, (np.nan, np.nan))[0] for u in regime_ids}


def run_multi_budget_study(cfg: dict, budgets=(2, 4, 6), n_seeds=(0, 1, 2)) -> dict:
    t0 = time.time()
    prep = prepare_pipeline(cfg)
    pool, model_names = prep["pool"], prep["model_names"]
    patches_by_regime, profile_entries = prep["patches_by_regime"], prep["profile_entries"]
    regime_ids, regime_weight_by_id = prep["regime_ids"], prep["regime_weight_by_id"]
    critical_mask_by_regime = prep["critical_mask_by_regime"]
    scale, blur_sigma, noise_std = prep["scale"], prep["blur_sigma"], prep["noise_std"]
    model_mean_psnr = prep["model_mean_psnr"]
    out_dir = prep["out_dir"]

    regime_weight_arr = np.array([regime_weight_by_id[u] for u in regime_ids])

    log("=== Estudo multi-orçamento (modelo: OPFsembleR / ICPR 2026) ===")

    def eval_per_regime(name, selected, fusion_mode="weighted"):
        pr = evaluate_selection_per_regime(
            name, selected, pool, patches_by_regime, scale, blur_sigma, noise_std,
            model_mean_psnr, seeds=n_seeds, fusion_mode=fusion_mode,
        )
        return pr

    # ---- Métodos CONSTANTES entre orçamentos: calculados uma única vez ---- #
    log("Avaliando métodos constantes entre orçamentos (Full-Ensemble, Best-Single, Pareto-Pruning)...")
    const_selections = {
        "Full-Ensemble": full_ensemble(model_names)["selected"],
        "Best-Single": best_single_model(profile_entries, model_names, regime_weight_by_id)["selected"],
    }
    problem_ref = build_problem(
        profile_entries, model_names, regime_ids, regime_weight=regime_weight_arr,
        critical_mask=critical_mask_by_regime, psnr_thresh=cfg["psnr_thresh"],
        fss_thresh=cfg["fss_thresh"], csi_thresh=cfg["csi_thresh"],
    )
    const_selections["Pareto-Pruning"] = pareto_pruning(
        profile_entries, model_names, problem_ref.cost, regime_weight_by_id)["selected"]

    const_per_regime = {}
    for name, sel in const_selections.items():
        const_per_regime[name] = eval_per_regime(name, sel)
        log(f"  [{name}] selecionados={sel}")

    all_tables = {}
    all_stats = {}
    cd_diagram_paths = {}

    for K in budgets:
        log(f"--- Orçamento K={K} modelos ---")
        problem = build_problem(
            profile_entries, model_names, regime_ids, regime_weight=regime_weight_arr,
            critical_mask=critical_mask_by_regime, psnr_thresh=cfg["psnr_thresh"],
            fss_thresh=cfg["fss_thresh"], csi_thresh=cfg["csi_thresh"],
        )
        radarcover_res = _solve(problem, max_models=K)
        traditional_res = traditional_set_cover(problem, max_models=K)
        topk_res = top_k(profile_entries, model_names, regime_weight_by_id, k=K)
        random_res = random_pruning(model_names, k=K, seed=cfg["seed"])
        diversity_res = diversity_based_pruning(profile_entries, model_names, regime_ids, k=K, seed=cfg["seed"])

        variable_selections = {
            "RadarCover-Multicover": radarcover_res["selected"],
            "Top-K": topk_res["selected"],
            "Traditional-Set-Cover": traditional_res["selected"],
            "Random-Pruning": random_res["selected"],
            "Diversity-Based": diversity_res["selected"],
        }

        per_regime_results = dict(const_per_regime)  # reaproveita os constantes
        for name, sel in variable_selections.items():
            per_regime_results[name] = eval_per_regime(name, sel)
            n_sel = len(sel)
            log(f"  [{name}] K={K} -> {n_sel} modelo(s) selecionado(s): {sel}")

        method_order = ["RadarCover-Multicover", "Top-K", "Full-Ensemble", "Best-Single",
                         "Pareto-Pruning", "Traditional-Set-Cover", "Random-Pruning", "Diversity-Based"]

        # ---- Monta a tabela (regime x método) estilo Tabela 1/2/3 do modelo ---- #
        rows = []
        for idx, u in enumerate(regime_ids):
            row = {"regime": u, "regime_weight": round(regime_weight_by_id[u], 3),
                   "critical": bool(critical_mask_by_regime[idx])}
            for m in method_order:
                mean, std = per_regime_results[m].get(u, {}).get("psnr", (np.nan, np.nan))
                row[m] = f"{mean:.2f}±{std:.2f}"
            rows.append(row)
        table_df = pd.DataFrame(rows)
        table_path = os.path.join(out_dir, f"budget_K{K}_per_regime_psnr.csv")
        table_df.to_csv(table_path, index=False)
        all_tables[K] = table_df
        log(f"  Tabela por-regime (PSNR, estilo Tabela 1/2/3 do modelo) salva em {table_path}")

        # ---- Teste de Friedman + Nemenyi + diagrama CD para este orçamento ---- #
        matrix = pd.DataFrame({
            m: [per_regime_results[m].get(u, {}).get("psnr", (np.nan, np.nan))[0] for u in regime_ids]
            for m in method_order
        }, index=regime_ids)
        res = friedman_nemenyi(matrix, higher_is_better=True)
        all_stats[K] = res
        log(f"  {format_result_text(res)}")

        cd_path = os.path.join(out_dir, f"cd_diagram_K{K}.png")
        plot_cd_diagram(res, f"Teste de Nemenyi — orçamento K={K} modelos (N={res.n} regimes)", cd_path)
        cd_diagram_paths[K] = cd_path

        # ---- Resumo agregado (compatibilidade com a tabela única anterior) ---- #
        agg_rows = []
        for m in method_order:
            sel = variable_selections.get(m, const_selections.get(m))
            psnr_vals = [per_regime_results[m][u]["psnr"][0] for u in regime_ids if u in per_regime_results[m]]
            agg_rows.append({
                "method": m, "budget_K": K, "n_models": len(sel),
                "psnr_mean_over_regimes": float(np.mean(psnr_vals)) if psnr_vals else np.nan,
                "avg_rank": float(res.avg_ranks.get(m, np.nan)),
            })
        agg_df = pd.DataFrame(agg_rows).sort_values("avg_rank")
        agg_df.to_csv(os.path.join(out_dir, f"budget_K{K}_summary.csv"), index=False)

    elapsed = time.time() - t0
    log(f"Estudo multi-orçamento concluído em {elapsed:.1f}s.")

    return {
        "tables": all_tables, "stats": all_stats, "cd_diagrams": cd_diagram_paths,
        "const_selections": const_selections, "elapsed_sec": elapsed,
    }
