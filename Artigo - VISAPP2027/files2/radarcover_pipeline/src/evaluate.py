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

    if not selected:
        return {"method": name, "n_models": 0, "total_params": 0, "latency_ms": 0.0,
                "psnr": 0.0, "ssim": 0.0, "fss": 0.0, "csi": 0.0, "recall": 0.0}

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
            pred = ens.predict(lr, mode=fusion_mode)[0, 0]
            pred = np.clip(pred, 0.0, 1.0)
            psnrs.append(psnr(pred, hr)); ssims.append(ssim(pred, hr))
            fsss.append(fractions_skill_score(pred, hr))
            csis.append(critical_success_index(pred, hr))
            recalls.append(recall_critical(pred, hr))
            ws.append(w_u)

    ws = np.array(ws)
    lat = ens.mean_latency_ms(sample_lr) if sample_lr is not None else 0.0

    def wavg(vals):
        return float(np.average(vals, weights=ws)) if vals else 0.0

    return {
        "method": name,
        "n_models": len(selected),
        "selected_models": ", ".join(selected),
        "total_params": ens.n_params_total(),
        "latency_ms": lat,
        "psnr": wavg(psnrs),
        "ssim": wavg(ssims),
        "fss": wavg(fsss),
        "csi": wavg(csis),
        "recall": wavg(recalls),
    }


def build_results_table(results: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    cols = ["method", "n_models", "total_params", "latency_ms", "psnr", "ssim", "fss", "csi", "recall"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].sort_values("psnr", ascending=False).reset_index(drop=True)
