"""
ensemble.py — Etapa 8: Pruned Ensemble Inference.

Combina apenas os modelos selecionados pelo Weighted Set Multicover (Etapa 7)
para restaurar imagens de radar de forma eficiente, via:
  - simple_average : média aritmética das predições
  - weighted_fusion : média ponderada pelo desempenho médio (PSNR) de cada
                       modelo selecionado — dá mais peso a modelos mais
                       precisos no ensemble final.
"""

from __future__ import annotations
from typing import Dict, List
import numpy as np


class PrunedEnsemble:
    def __init__(self, models: Dict[str, object], selected: List[str], weights: Dict[str, float] = None):
        self.models = {k: models[k] for k in selected}
        if weights is None:
            weights = {k: 1.0 for k in selected}
        total = sum(weights.get(k, 1.0) for k in selected)
        self.weights = {k: weights.get(k, 1.0) / total for k in selected}

    def predict(self, lr_batch: np.ndarray, mode: str = "weighted") -> np.ndarray:
        preds = []
        for k, m in self.models.items():
            p = m.predict(lr_batch)
            w = self.weights[k] if mode == "weighted" else 1.0 / len(self.models)
            preds.append(w * p)
        out = np.sum(preds, axis=0)
        return np.clip(out, 0.0, 1.0)

    def n_params_total(self) -> int:
        return sum(m.n_params() for m in self.models.values())

    def mean_latency_ms(self, lr_batch: np.ndarray) -> float:
        from models import measure_latency
        lats = [measure_latency(m, lr_batch, n_reps=3) for m in self.models.values()]
        return float(np.sum(lats))  # custo total = soma (execução sequencial dos membros)
