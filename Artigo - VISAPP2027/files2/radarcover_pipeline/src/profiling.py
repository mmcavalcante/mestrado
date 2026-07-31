"""
profiling.py — Etapa 5: Per-Regime Model Profiling.

Avalia cada modelo candidato em cada regime meteorológico com:
  - Qualidade visual : PSNR, SSIM
  - Preservação meteorológica : FSS (Fractions Skill Score), CSI (Critical
    Success Index) e Recall sobre ecos críticos (pixels acima de um limiar
    de refletividade operacionalmente relevante)
  - Eficiência : latência de inferência, nº de parâmetros (proxy de FLOPs)

O resultado é uma tabela (model, regime) -> métricas, usada para (a) definir
a matriz de cobertura binária da Etapa 6 via thresholds de qualidade e (b)
alimentar a avaliação final (Etapa 9).
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np
from skimage.metrics import structural_similarity as ssim_fn

from models import measure_latency


def psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    mse = float(np.mean((pred - target) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * np.log10((data_range ** 2) / mse)


def ssim(pred: np.ndarray, target: np.ndarray) -> float:
    return float(ssim_fn(target, pred, data_range=1.0))


def fractions_skill_score(pred: np.ndarray, target: np.ndarray, thresh: float = 0.1,
                           window: int = 8) -> float:
    """FSS clássico (Roberts & Lean, 2008), usado em verificação de previsão
    meteorológica espacial: compara frações de excedência de limiar em
    janelas locais entre previsão (restaurada) e observação (HR real)."""
    from scipy.ndimage import uniform_filter
    pf = (pred >= thresh).astype(np.float32)
    tf = (target >= thresh).astype(np.float32)
    pf_frac = uniform_filter(pf, size=window)
    tf_frac = uniform_filter(tf, size=window)
    num = np.mean((pf_frac - tf_frac) ** 2)
    denom = np.mean(pf_frac ** 2) + np.mean(tf_frac ** 2)
    if denom <= 1e-12:
        return 1.0
    return float(1.0 - num / denom)


def critical_success_index(pred: np.ndarray, target: np.ndarray, thresh: float = 0.1) -> float:
    """CSI = hits / (hits + misses + false_alarms) sobre pixels de eco crítico."""
    pf = pred >= thresh
    tf = target >= thresh
    hits = np.logical_and(pf, tf).sum()
    misses = np.logical_and(~pf, tf).sum()
    false_alarms = np.logical_and(pf, ~tf).sum()
    denom = hits + misses + false_alarms
    if denom == 0:
        return 1.0
    return float(hits / denom)


def recall_critical(pred: np.ndarray, target: np.ndarray, thresh: float = 0.1) -> float:
    pf = pred >= thresh
    tf = target >= thresh
    hits = np.logical_and(pf, tf).sum()
    total_pos = tf.sum()
    if total_pos == 0:
        return 1.0
    return float(hits / total_pos)


@dataclass
class ProfileEntry:
    model: str
    regime_id: int
    n_samples: int
    psnr: float
    ssim: float
    fss: float
    csi: float
    recall: float
    latency_ms: float
    n_params: int


def profile_model_on_regime(model, patches: List[np.ndarray], scale: int, blur_sigma: float,
                             noise_std: float, rng: np.random.Generator, regime_id: int) -> ProfileEntry:
    """Avalia `model` (já treinado) sobre todos os patches HR de um regime."""
    from data import degrade

    psnrs, ssims, fsss, csis, recalls = [], [], [], [], []
    for hr in patches:
        lr = degrade(hr, scale, blur_sigma, noise_std, rng)
        pred = model.predict(lr[None, None, ...])[0, 0]
        pred = np.clip(pred, 0.0, 1.0)
        psnrs.append(psnr(pred, hr))
        ssims.append(ssim(pred, hr))
        fsss.append(fractions_skill_score(pred, hr))
        csis.append(critical_success_index(pred, hr))
        recalls.append(recall_critical(pred, hr))

    lat = measure_latency(model, patches[0][None, None, ...].astype(np.float32) if patches else
                           np.zeros((1, 1, 8, 8), dtype=np.float32))

    return ProfileEntry(
        model=model.name if hasattr(model, "name") else str(model),
        regime_id=regime_id,
        n_samples=len(patches),
        psnr=float(np.mean(psnrs)) if psnrs else 0.0,
        ssim=float(np.mean(ssims)) if ssims else 0.0,
        fss=float(np.mean(fsss)) if fsss else 0.0,
        csi=float(np.mean(csis)) if csis else 0.0,
        recall=float(np.mean(recalls)) if recalls else 0.0,
        latency_ms=lat,
        n_params=model.n_params(),
    )


def profile_pool_over_regimes(pool: Dict[str, object], descriptors_by_regime: Dict[int, list],
                               scale: int, blur_sigma: float, noise_std: float,
                               seed: int = 0) -> List[ProfileEntry]:
    rng = np.random.default_rng(seed)
    entries = []
    for model_key, model in pool.items():
        for regime_id, patches in descriptors_by_regime.items():
            entry = profile_model_on_regime(model, patches, scale, blur_sigma, noise_std, rng, regime_id)
            entry.model = model_key  # usa chave única (nome#seed)
            entries.append(entry)
    return entries
