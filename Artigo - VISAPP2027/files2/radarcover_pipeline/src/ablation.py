"""
ablation.py — Capítulo 6: Ablation Study.

Reaproveita as Etapas 1-5 (já treinadas/perfiladas uma única vez, via
`prepare_pipeline`) e reexecuta apenas as Etapas 6-8 sob 6 configurações,
isolando a contribuição de cada componente de design do RadarCover listado
no diagrama da metodologia:

  A. RadarCover completo (referência)          — multicover + pesos + custo + fusão ponderada
  B. Sem multicobertura                         — r_u = 1 para todos os regimes (mesmo críticos)
  C. Sem pesos de regime (severidade)           — w_u = 1 uniforme
  D. Sem métricas meteorológicas na cobertura    — alpha definido só por PSNR (ignora FSS/CSI)
  E. Sem custo computacional (cost-unaware)      — minimiza apenas nº de modelos, ignora custo_j
  F. Fusão simples (não ponderada)               — troca fusão ponderada por média aritmética

A comparação isola, um de cada vez, o efeito de cada peça da formulação do
Weighted Set Multicover (Etapa 7) e da fusão do ensemble (Etapa 8).
"""

from __future__ import annotations
import os
import time

import numpy as np
import pandas as pd

from pipeline import prepare_pipeline, log
from coverage import build_problem, solve_ilp, solve_greedy
from evaluate import evaluate_selection


def _solve(problem, max_models):
    try:
        res = solve_ilp(problem, max_models=max_models)
    except Exception:
        res = solve_greedy(problem, max_models=max_models)
    return res


def run_ablation_study(cfg: dict) -> pd.DataFrame:
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
    uniform_weight_arr = np.ones_like(regime_weight_arr)
    max_models = cfg.get("max_models")

    rows = []

    def evaluate_variant(variant_name, problem, fusion_mode="weighted", max_models_override=None):
        res = _solve(problem, max_models_override if max_models_override is not None else max_models)
        selected = res["selected"]
        metrics = evaluate_selection(
            variant_name, selected, pool, patches_by_regime, scale, blur_sigma, noise_std,
            regime_weight_by_id, model_mean_psnr, seed=cfg["seed"] + 3, fusion_mode=fusion_mode,
        )
        metrics["selected_models"] = ", ".join(selected)
        metrics["n_regimes_unmet"] = len(res.get("unmet_regimes") or {})
        log(f"    [{variant_name}] n_models={metrics['n_models']} PSNR={metrics['psnr']:.2f} "
            f"SSIM={metrics['ssim']:.3f} FSS={metrics['fss']:.3f} CSI={metrics['csi']:.3f} "
            f"latency={metrics['latency_ms']:.2f}ms unmet_regimes={metrics['n_regimes_unmet']}")
        rows.append(metrics)
        return problem, res

    log("=== Ablation Study (Capítulo 6) — variando um componente por vez ===")

    # --- A. RadarCover completo (referência) --------------------------- #
    problem_full = build_problem(
        profile_entries, model_names, regime_ids, regime_weight=regime_weight_arr,
        critical_mask=critical_mask_by_regime, psnr_thresh=cfg["psnr_thresh"],
        fss_thresh=cfg["fss_thresh"], csi_thresh=cfg["csi_thresh"],
    )
    log("  (A) RadarCover completo (referência)")
    evaluate_variant("A_RadarCover_Full", problem_full, fusion_mode="weighted")

    # --- B. Sem multicobertura (r_u = 1 sempre) -------------------------- #
    problem_no_multicover = build_problem(
        profile_entries, model_names, regime_ids, regime_weight=regime_weight_arr,
        critical_mask=np.zeros_like(critical_mask_by_regime),  # nenhum regime exige >=2
        psnr_thresh=cfg["psnr_thresh"], fss_thresh=cfg["fss_thresh"], csi_thresh=cfg["csi_thresh"],
    )
    log("  (B) Sem multicobertura (r_u=1 para regimes críticos)")
    evaluate_variant("B_No_Multicoverage", problem_no_multicover, fusion_mode="weighted")

    # --- C. Sem pesos de regime (severidade uniforme) --------------------- #
    problem_no_weights = build_problem(
        profile_entries, model_names, regime_ids, regime_weight=uniform_weight_arr,
        critical_mask=critical_mask_by_regime, psnr_thresh=cfg["psnr_thresh"],
        fss_thresh=cfg["fss_thresh"], csi_thresh=cfg["csi_thresh"],
    )
    log("  (C) Sem pesos de importância de regime (w_u=1 uniforme)")
    evaluate_variant("C_No_Regime_Weights", problem_no_weights, fusion_mode="weighted")

    # --- D. Sem métricas meteorológicas na matriz de cobertura ----------- #
    problem_no_meteo = build_problem(
        profile_entries, model_names, regime_ids, regime_weight=regime_weight_arr,
        critical_mask=critical_mask_by_regime, psnr_thresh=cfg["psnr_thresh"],
        fss_thresh=-1.0, csi_thresh=-1.0,  # threshold trivialmente satisfeito -> só PSNR filtra
    )
    log("  (D) Cobertura baseada só em PSNR (sem FSS/CSI meteorológicos)")
    evaluate_variant("D_No_Meteorological_Metrics", problem_no_meteo, fusion_mode="weighted")

    # --- E. Cost-unaware (ignora custo computacional na otimização) ------ #
    problem_no_cost = build_problem(
        profile_entries, model_names, regime_ids, regime_weight=regime_weight_arr,
        critical_mask=critical_mask_by_regime, psnr_thresh=cfg["psnr_thresh"],
        fss_thresh=cfg["fss_thresh"], csi_thresh=cfg["csi_thresh"],
    )
    problem_no_cost.cost[:] = 1.0  # todos os modelos custam "1" -> otimização ignora latência/params
    log("  (E) Otimização cost-unaware (custo uniforme; minimiza só nº de modelos)")
    evaluate_variant("E_Cost_Unaware", problem_no_cost, fusion_mode="weighted")

    # --- F. Fusão simples (não ponderada) --------------------------------- #
    log("  (F) Fusão simples (média aritmética) sobre a mesma seleção do RadarCover completo")
    evaluate_variant("F_Simple_Fusion", problem_full, fusion_mode="simple")

    df = pd.DataFrame(rows)
    cols = ["method", "n_models", "n_regimes_unmet", "total_params", "latency_ms",
            "psnr", "ssim", "fss", "csi", "recall", "selected_models"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(os.path.join(out_dir, "ablation_results.csv"), index=False)

    elapsed = time.time() - t0
    log(f"Ablation Study concluído em {elapsed:.1f}s. Resultados em: "
        f"{os.path.join(out_dir, 'ablation_results.csv')}")
    return df
