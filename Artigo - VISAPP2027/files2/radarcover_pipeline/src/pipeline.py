"""
pipeline.py — Orquestra as 9 etapas do RadarCover sobre o IPMet Radar Dataset.

Uso:
    python run_demo.py --config configs/default.yaml

Este módulo implementa `run_pipeline(cfg)` que executa, em ordem:
  1. Carregamento do dataset + split cronológico
  2. Extração de patches HR + geração de pares (LR,HR) degradados
  3. Construção do pool de modelos candidatos
  4. Construção dos regimes (descritores + clustering)
  5. Treinamento leve e profiling por regime
  6. Construção da matriz de cobertura
  7. Otimização Weighted Set Multicover (+ baselines)
  8. Montagem do ensemble podado
  9. Avaliação comparativa final
"""

from __future__ import annotations
import json
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from data import list_dataset, load_reflectivity, chronological_split, degrade
from regimes import (RegimeDescriptor, extract_features, build_regimes,
                      regime_weights_by_severity, critical_regimes, FEATURE_NAMES)
from models import build_candidate_pool
from profiling import profile_pool_over_regimes
from coverage import build_problem, solve_ilp, solve_greedy
from baselines import (best_single_model, full_ensemble, random_pruning, top_k,
                        diversity_based_pruning, pareto_pruning, traditional_set_cover)
from evaluate import evaluate_selection, build_results_table


def log(msg):
    print(f"[RadarCover] {msg}", flush=True)


def prepare_pipeline(cfg: dict) -> dict:
    """Executa as Etapas 1-5 (dataset -> preprocessing -> pool -> regimes ->
    profiling), que são custosas (treino dos modelos) e comuns tanto ao
    `run_pipeline` (Etapas 6-9) quanto ao `ablation.py` (Capítulo 6), que
    precisa reexecutar apenas as Etapas 6-8 sob configurações variadas."""
    t_start = time.time()
    rng = np.random.default_rng(cfg["seed"])
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # ---------------- Etapa 1: IPMet Radar Dataset ---------------------- #
    log("Etapa 1/9 — Carregando dataset e realizando split cronológico...")
    frames = list_dataset(cfg["images_dir"])
    frames_sample = list(np.array(frames, dtype=object)[
        rng.choice(len(frames), size=min(cfg["n_frames"], len(frames)), replace=False)
    ])
    frames_sample.sort(key=lambda f: f.acq_dt)
    train_f, val_f, test_f = chronological_split(frames_sample, train=0.6, val=0.2)
    log(f"  frames totais no dataset: {len(frames)} | amostrados: {len(frames_sample)} "
        f"(train={len(train_f)}, val={len(val_f)}, test={len(test_f)})")

    # ---------------- Etapa 2: Image Preprocessing ----------------------- #
    log("Etapa 2/9 — Extraindo patches HR e definindo protocolo de degradação...")
    patch_size = cfg["patch_size"]
    scale = cfg["scale"]
    blur_sigma = cfg["blur_sigma"]
    noise_std = cfg["noise_std"]

    def extract_hr_patches(frame_list, patches_per_frame):
        out = []
        for f in frame_list:
            img = load_reflectivity(f.path)
            ys = list(range(0, img.shape[0] - patch_size + 1, patch_size))
            xs = list(range(0, img.shape[1] - patch_size + 1, patch_size))
            coords = [(y, x) for y in ys for x in xs]
            # prioriza patches com eco (sinal útil para treino/avaliação)
            scored = [(y, x, img[y:y + patch_size, x:x + patch_size].mean()) for y, x in coords]
            scored.sort(key=lambda t: t[2], reverse=True)
            chosen = scored[:patches_per_frame]
            for y, x, _ in chosen:
                patch = img[y:y + patch_size, x:x + patch_size]
                out.append((patch, (x + patch_size // 2, y + patch_size // 2)))
        return out

    train_patches = extract_hr_patches(train_f, cfg["patches_per_frame_train"])
    eval_patches = extract_hr_patches(test_f, cfg["patches_per_frame_eval"])
    log(f"  patches de treino: {len(train_patches)} | patches de avaliação: {len(eval_patches)}")

    # ---------------- Etapa 3: Candidate Model Pool ----------------------- #
    log("Etapa 3/9 — Construindo o pool de modelos candidatos...")
    pool = build_candidate_pool(scale=scale, seeds=tuple(cfg["seeds"]))
    log(f"  {len(pool)} candidatos: {list(pool.keys())}")

    log(f"  Treinando cada candidato por {cfg['train_iters']} iterações "
        f"sobre {len(train_patches)} patches...")
    train_rng = np.random.default_rng(cfg["seed"] + 1)
    for key, model in pool.items():
        losses = []
        for it in range(cfg["train_iters"]):
            patch, _ = train_patches[train_rng.integers(0, len(train_patches))]
            lr, hr = _lr_hr(patch, scale, blur_sigma, noise_std, train_rng)
            loss = model.train_step(lr, hr, lr_rate=cfg["lr_rate"])
            losses.append(loss)
        log(f"    {key:22s} loss_inicial={losses[0]:.4f} -> loss_final={np.mean(losses[-10:]):.4f}")

    # ---------------- Etapa 4: Regime Construction ----------------------- #
    log("Etapa 4/9 — Construindo regimes meteorológicos (descritores + clustering)...")
    descriptors = [
        RegimeDescriptor(patch=p, center_xy=xy, features=extract_features(p, xy))
        for p, xy in eval_patches
    ]
    n_regimes = min(cfg["n_regimes"], max(2, len(descriptors) // 3))
    descriptors, km, scaler = build_regimes(descriptors, n_regimes=n_regimes, seed=cfg["seed"])
    regime_ids = sorted(set(d.regime_id for d in descriptors))
    regime_w_arr = regime_weights_by_severity(descriptors, n_regimes=n_regimes)
    crit_mask = critical_regimes(descriptors, n_regimes=n_regimes)
    regime_weight_by_id = {u: float(regime_w_arr[u]) for u in regime_ids}
    log(f"  {n_regimes} regimes construídos | críticos: "
        f"{[u for u in regime_ids if crit_mask[u]]}")

    patches_by_regime = defaultdict(list)
    for d in descriptors:
        patches_by_regime[d.regime_id].append(d.patch)

    # ---------------- Etapa 5: Per-Regime Model Profiling ----------------- #
    log("Etapa 5/9 — Perfilando cada modelo em cada regime (PSNR/SSIM/FSS/CSI/Recall/latência)...")
    profile_entries = profile_pool_over_regimes(pool, patches_by_regime, scale, blur_sigma,
                                                 noise_std, seed=cfg["seed"] + 2)
    prof_df = pd.DataFrame([e.__dict__ for e in profile_entries])
    prof_df.to_csv(os.path.join(out_dir, "profile_table.csv"), index=False)
    log(f"  tabela de profiling salva ({len(prof_df)} linhas) em profile_table.csv")

    model_mean_psnr = prof_df.groupby("model")["psnr"].mean().to_dict()
    model_names = list(pool.keys())
    critical_mask_by_regime = np.array([crit_mask[u] for u in regime_ids])
    elapsed_prep = time.time() - t_start
    log(f"  Etapas 1-5 concluídas em {elapsed_prep:.1f}s.")

    return {
        "cfg": cfg, "out_dir": out_dir, "pool": pool, "model_names": model_names,
        "patches_by_regime": patches_by_regime, "profile_entries": profile_entries,
        "prof_df": prof_df, "model_mean_psnr": model_mean_psnr, "regime_ids": regime_ids,
        "regime_weight_by_id": regime_weight_by_id, "critical_mask_by_regime": critical_mask_by_regime,
        "scale": scale, "blur_sigma": blur_sigma, "noise_std": noise_std, "elapsed_prep": elapsed_prep,
    }


def run_pipeline(cfg: dict) -> dict:
    prep = prepare_pipeline(cfg)
    t_start = time.time()
    out_dir, pool, model_names = prep["out_dir"], prep["pool"], prep["model_names"]
    patches_by_regime, profile_entries = prep["patches_by_regime"], prep["profile_entries"]
    regime_ids, regime_weight_by_id = prep["regime_ids"], prep["regime_weight_by_id"]
    critical_mask_by_regime = prep["critical_mask_by_regime"]
    scale, blur_sigma, noise_std = prep["scale"], prep["blur_sigma"], prep["noise_std"]
    model_mean_psnr = prep["model_mean_psnr"]

    # ---------------- Etapa 6: Coverage Matrix Definition ------------------ #
    log("Etapa 6/9 — Construindo matriz de cobertura...")
    problem = build_problem(
        profile_entries, model_names, regime_ids,
        regime_weight=np.array([regime_weight_by_id[u] for u in regime_ids]),
        critical_mask=critical_mask_by_regime,
        psnr_thresh=cfg["psnr_thresh"], fss_thresh=cfg["fss_thresh"], csi_thresh=cfg["csi_thresh"],
    )
    coverage_rate = problem.alpha.mean()
    log(f"  matriz alpha: {problem.alpha.shape} | taxa média de cobertura: {coverage_rate:.2%}")

    # ---------------- Etapa 7: Weighted Set Multicover Optimization ------- #
    log("Etapa 7/9 — Resolvendo Weighted Set Multicover (ILP + heurística gulosa)...")
    try:
        result_ilp = solve_ilp(problem, max_models=cfg.get("max_models"))
    except Exception as ex:
        log(f"  ILP falhou ({ex}); usando apenas heurística gulosa.")
        result_ilp = None
    result_greedy = solve_greedy(problem, max_models=cfg.get("max_models"))
    radarcover_result = result_ilp if result_ilp is not None else result_greedy
    radarcover_result["status"] = "RadarCover-Multicover(" + radarcover_result["status"] + ")"
    log(f"  RadarCover selecionou {len(radarcover_result['selected'])} modelos: "
        f"{radarcover_result['selected']}")

    # ---------------- Baselines (para a Etapa 9) --------------------------- #
    log("  Calculando seleções dos baselines...")
    k_budget = max(1, len(radarcover_result["selected"]))
    baseline_results = {
        "Best-Single": best_single_model(profile_entries, model_names, regime_weight_by_id),
        "Full-Ensemble": full_ensemble(model_names),
        "Random-Pruning": random_pruning(model_names, k=k_budget, seed=cfg["seed"]),
        "Top-K": top_k(profile_entries, model_names, regime_weight_by_id, k=k_budget),
        "Diversity-Based": diversity_based_pruning(profile_entries, model_names, regime_ids,
                                                     k=k_budget, seed=cfg["seed"]),
        "Pareto-Pruning": pareto_pruning(profile_entries, model_names, problem.cost, regime_weight_by_id),
        "Traditional-Set-Cover": traditional_set_cover(problem, max_models=cfg.get("max_models")),
        "RadarCover-Multicover": radarcover_result,
    }

    # ---------------- Etapa 8 + 9: Ensemble podado + Avaliação ------------- #
    log("Etapa 8-9/9 — Montando ensembles podados e avaliando (Etapa 9)...")
    all_results = []
    for name, res in baseline_results.items():
        metrics = evaluate_selection(
            name, res["selected"], pool, patches_by_regime, scale, blur_sigma, noise_std,
            regime_weight_by_id, model_mean_psnr, seed=cfg["seed"] + 3,
        )
        all_results.append(metrics)
        log(f"    {name:24s} n_models={metrics['n_models']:2d} "
            f"PSNR={metrics['psnr']:.2f} SSIM={metrics['ssim']:.3f} "
            f"FSS={metrics['fss']:.3f} CSI={metrics['csi']:.3f} "
            f"latency={metrics['latency_ms']:.2f}ms")

    results_df = build_results_table(all_results)
    results_df.to_csv(os.path.join(out_dir, "final_comparison.csv"), index=False)

    elapsed = time.time() - t_start
    log(f"Pipeline concluído em {elapsed:.1f}s. Resultados em: {out_dir}")

    return {
        "profile_table": prof_df,
        "results_table": results_df,
        "coverage_matrix": problem.alpha,
        "regime_ids": regime_ids,
        "model_names": model_names,
        "radarcover_selection": radarcover_result,
        "baseline_selections": {k: v["selected"] for k, v in baseline_results.items()},
        "elapsed_sec": elapsed,
    }


def _lr_hr(patch, scale, blur_sigma, noise_std, rng):
    lr = degrade(patch, scale, blur_sigma, noise_std, rng)
    return lr[None, None, ...], patch[None, None, ...]
