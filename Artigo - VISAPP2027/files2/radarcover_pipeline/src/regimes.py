"""
regimes.py — Etapa 4: Regime Construction.

Extrai descritores de cada patch/imagem de radar e agrupa (clustering) em
U regimes meteorológicos (conforme diagrama: 30-50 regimes). Os descritores
seguem os itens listados no diagrama da metodologia:

  - Estatísticas de intensidade (média, desvio-padrão, percentis)
  - Echo ratio (fração de pixels com eco > 0)
  - Entropia (textura/heterogeneidade espacial)
  - Densidade de eco (densidade local via kernel)
  - Componentes conectados (nº de células de chuva distintas)
  - Distância ao centro do radar (posição do patch relativa ao centro 640x640)

Cada regime representa um "modo" de operação (ex.: céu limpo, chuva estratiforme
difusa, núcleos convectivos intensos e localizados, etc.) sobre o qual os
modelos candidatos serão perfilados (Etapa 5) e a matriz de cobertura
construída (Etapa 6).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass
class RegimeDescriptor:
    patch: np.ndarray            # HR patch (H,W) float32 [0,1]
    center_xy: Tuple[int, int]   # posição do patch na imagem original
    features: np.ndarray = field(default=None)
    regime_id: int = -1


def _entropy(img: np.ndarray, bins: int = 16) -> float:
    hist, _ = np.histogram(img, bins=bins, range=(0, 1), density=False)
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _connected_components(img: np.ndarray, thresh: float = 0.05) -> int:
    binary = img > thresh
    if not binary.any():
        return 0
    labeled, n = ndimage.label(binary)
    return int(n)


def _local_density(img: np.ndarray, thresh: float = 0.05) -> float:
    """Densidade local de eco via média de uma máscara suavizada (proxy de
    'echo density' sem exigir kernel KDE completo)."""
    binary = (img > thresh).astype(np.float32)
    smoothed = ndimage.uniform_filter(binary, size=max(3, img.shape[0] // 8))
    return float(smoothed.mean())


def extract_features(patch: np.ndarray, center_xy: Tuple[int, int], image_center=(320, 320),
                      radar_max_dist=452.5) -> np.ndarray:
    """Vetor de descritores de um patch, conforme Etapa 4 do diagrama."""
    mean_i = float(patch.mean())
    std_i = float(patch.std())
    p90 = float(np.percentile(patch, 90))
    echo_ratio = float((patch > 0.02).mean())
    ent = _entropy(patch)
    density = _local_density(patch)
    n_components = _connected_components(patch)
    dist_to_center = float(np.hypot(center_xy[0] - image_center[0], center_xy[1] - image_center[1]))
    dist_norm = dist_to_center / radar_max_dist
    return np.array([mean_i, std_i, p90, echo_ratio, ent, density, n_components, dist_norm],
                     dtype=np.float32)


FEATURE_NAMES = ["mean_intensity", "std_intensity", "p90_intensity", "echo_ratio",
                  "entropy", "echo_density", "n_connected_components", "dist_to_radar_norm"]


def build_regimes(descriptors: List[RegimeDescriptor], n_regimes: int = 40, seed: int = 0):
    """Clustering K-Means dos descritores em `n_regimes` regimes (30-50 default),
    retornando os descritores anotados + o modelo de clustering (para reuso
    em novos patches, ex. conjunto de teste)."""
    X = np.stack([d.features for d in descriptors])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_regimes, random_state=seed, n_init=10)
    labels = km.fit_predict(Xs)
    for d, lab in zip(descriptors, labels):
        d.regime_id = int(lab)
    return descriptors, km, scaler


def regime_weights_by_severity(descriptors: List[RegimeDescriptor], n_regimes: int) -> np.ndarray:
    """
    Peso de importância meteorológica w_u por regime (Etapa 6: 'Regime
    importance weights'). Regimes com maior intensidade média e maior nº de
    componentes conectados (núcleos convectivos) recebem peso maior — eles
    correspondem a eventos de maior risco operacional (ex. alertas de
    tempestade), sendo mais custoso perder cobertura nesses regimes.
    """
    weights = np.ones(n_regimes, dtype=np.float32)
    for r in range(n_regimes):
        members = [d for d in descriptors if d.regime_id == r]
        if not members:
            continue
        mean_intensity = np.mean([d.features[0] for d in members])
        mean_components = np.mean([d.features[6] for d in members])
        severity = 1.0 + 3.0 * mean_intensity + 0.05 * mean_components
        weights[r] = severity
    # normaliza para média 1.0 (mantém escala interpretável)
    weights = weights / weights.mean()
    return weights


def critical_regimes(descriptors: List[RegimeDescriptor], n_regimes: int,
                      echo_ratio_thresh: float = 0.15) -> np.ndarray:
    """
    Marca regimes 'críticos' (Etapa 6: 'Critical echoes may require
    multicoverage') — aqueles cuja fração média de eco excede um limiar,
    indicando eventos de precipitação significativa. Esses regimes devem ser
    cobertos por >= 2 modelos no Weighted Set Multicover (redundância).
    """
    crit = np.zeros(n_regimes, dtype=bool)
    for r in range(n_regimes):
        members = [d for d in descriptors if d.regime_id == r]
        if not members:
            continue
        mean_echo_ratio = np.mean([d.features[3] for d in members])
        crit[r] = mean_echo_ratio >= echo_ratio_thresh
    return crit
