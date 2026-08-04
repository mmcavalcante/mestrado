"""
evaluate.py — Etapa 9: Evaluation and Analysis.

Para cada método de seleção de ensemble (RadarCover/Multicover + 7
baselines), calcula:
  - Qualidade visual  : PSNR, SSIM médios (ponderados por peso de regime)
  - Meteorológico     : FSS, CSI, Recall médios
  - Eficiência        : latência total do ensemble podado, nº de parâmetros
                         totais, nº de modelos selecionados ("model count")
"""

from __future__ import annotations
from typing import Dict, List
import numpy as np
import pandas as pd

from ensemble import PrunedEnsemble
from profiling import psnr, ssim, fractions_skill_score, critical_success_index, recall_critical


def _predict_or_fallback(ens, lr, scale, fusion_mode):
    """Se não há modelos selecionados, a predição degrada para o upsample
    bicúbico trivial da entrada LR — o comportamento real de um sistema sem
    nenhum modelo de restauração disponível — em vez de retornar zero ou
    NaN, o que impediria comparações e testes estatísticos válidos."""
    if ens is None:
        from nn_core import bicubic_upsample
        return bicubic_upsample(lr, scale)[0, 0]
    return np.clip(ens.predict(lr, mode=fusion_mode)[0, 0], 0.0, 1.0)


def evaluate_selection(name: str, selected: List[str], pool: Dict[str, object],
                        eval_patches_by_regime: Dict[int, list], scale: int, blur_sigma: float,
                        noise_std: float, regime_weight_by_id: Dict[int, float],
                        model_mean_psnr: Dict[str, float], seed: int = 0,
                        fusion_mode: str = "weighted") -> dict:
    """`fusion_mode`: 'weighted' (Etapa 8, padrão do RadarCover — pondera por
    PSNR médio de cada modelo selecionado) ou 'simple' (média aritmética
    entre os modelos selecionados, usada no Ablation Study — Capítulo 6 —
    para isolar a contribuição específica da fusão ponderada)."""
    from data import degrade

    ens = None
    if selected:
        weights = {k: max(model_mean_psnr.get(k, 1.0), 1e-3) for k in selected}
        ens = PrunedEnsemble(pool, selected, weights=weights)

    rng = np.random.default_rng(seed)
    psnrs, ssims, fsss, csis, recalls, ws = [], [], [], [], [], []
    sample_lr = None
    for regime_id, patches in eval_patches_by_regime.items():
        w_u = regime_weight_by_id.get(regime_id, 1.0)
        for hr in patches:
            lr = degrade(hr, scale, blur_sigma, noise_std, rng)[None, None, ...]
            if sample_lr is None:
                sample_lr = lr
            pred = _predict_or_fallback(ens, lr, scale, fusion_mode)
            psnrs.append(psnr(pred, hr)); ssims.append(ssim(pred, hr))
            fsss.append(fractions_skill_score(pred, hr))
            csis.append(critical_success_index(pred, hr))
            recalls.append(recall_critical(pred, hr))
            ws.append(w_u)

    ws = np.array(ws)
    lat = ens.mean_latency_ms(sample_lr) if (ens is not None and sample_lr is not None) else 0.0

    def wavg(vals):
        return float(np.average(vals, weights=ws)) if vals else 0.0

    return {
        "method": name,
        "n_models": len(selected),
        "selected_models": ", ".join(selected) if selected else "(nenhum — fallback bicúbico)",
        "total_params": ens.n_params_total() if ens is not None else 0,
        "latency_ms": lat,
        "psnr": wavg(psnrs),
        "ssim": wavg(ssims),
        "fss": wavg(fsss),
        "csi": wavg(csis),
        "recall": wavg(recalls),
    }


def evaluate_selection_per_regime(name: str, selected: List[str], pool: Dict[str, object],
                                   eval_patches_by_regime: Dict[int, list], scale: int, blur_sigma: float,
                                   noise_std: float, model_mean_psnr: Dict[str, float],
                                   seeds: List[int] = (0, 1, 2), fusion_mode: str = "weighted") -> dict:
    """
    Evaluates a method's pruned ensemble SEPARATELY on each regime's own
    patches (never pooled across regimes), repeated over multiple random
    seeds for the degradation noise. This mirrors how OPFsembleR evaluates
    each pruning strategy independently on each of its N datasets: here,
    each of the U meteorological regimes plays the role of one "dataset",
    which is what makes a Friedman/Nemenyi test across regimes valid later
    (Section 5 of the restructured article).

    An empty `selected` list falls back to bicubic upsampling (see
    `_predict_or_fallback`) rather than returning `{}`, so every method has
    a defined score in every regime — required for the Friedman test, which
    cannot handle missing cells.

    Returns: {regime_id: {metric: (mean_over_seeds, std_over_seeds)}}
    """
    from data import degrade

    ens = None
    if selected:
        weights = {k: max(model_mean_psnr.get(k, 1.0), 1e-3) for k in selected}
        ens = PrunedEnsemble(pool, selected, weights=weights)

    per_regime_per_seed = {u: {"psnr": [], "ssim": [], "fss": [], "csi": [], "recall": []}
                            for u in eval_patches_by_regime}

    for seed in seeds:
        rng = np.random.default_rng(seed)
        for regime_id, patches in eval_patches_by_regime.items():
            psnrs, ssims, fsss, csis, recalls = [], [], [], [], []
            for hr in patches:
                lr = degrade(hr, scale, blur_sigma, noise_std, rng)[None, None, ...]
                pred = _predict_or_fallback(ens, lr, scale, fusion_mode)
                psnrs.append(psnr(pred, hr)); ssims.append(ssim(pred, hr))
                fsss.append(fractions_skill_score(pred, hr))
                csis.append(critical_success_index(pred, hr))
                recalls.append(recall_critical(pred, hr))
            if psnrs:
                per_regime_per_seed[regime_id]["psnr"].append(float(np.mean(psnrs)))
                per_regime_per_seed[regime_id]["ssim"].append(float(np.mean(ssims)))
                per_regime_per_seed[regime_id]["fss"].append(float(np.mean(fsss)))
                per_regime_per_seed[regime_id]["csi"].append(float(np.mean(csis)))
                per_regime_per_seed[regime_id]["recall"].append(float(np.mean(recalls)))

    out = {}
    for regime_id, metrics in per_regime_per_seed.items():
        out[regime_id] = {}
        for metric_name, vals in metrics.items():
            if vals:
                out[regime_id][metric_name] = (float(np.mean(vals)), float(np.std(vals)))
            else:
                out[regime_id][metric_name] = (0.0, 0.0)
    return out


def build_results_table(results: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    cols = ["method", "n_models", "total_params", "latency_ms", "psnr", "ssim", "fss", "csi", "recall"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].sort_values("psnr", ascending=False).reset_index(drop=True)
