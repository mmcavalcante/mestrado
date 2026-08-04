"""
cd_diagram.py — Renderiza o diagrama de diferença crítica (CD diagram) no
estilo Demšar (2006, Figura 1) / Figura 6 do artigo-modelo (OPFsembleR).

Eixo horizontal = rank médio (1 = melhor, à direita); cada método é um
tick; métodos cujo rank médio difere por menos que CD são conectados por
uma barra horizontal (não distinguíveis estatisticamente).
"""

from __future__ import annotations
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stats_test import FriedmanNemenyiResult, cliques_within_cd


def plot_cd_diagram(res: FriedmanNemenyiResult, title: str, out_path: str):
    avg_ranks = res.avg_ranks.sort_values()
    names = list(avg_ranks.index)
    ranks = list(avg_ranks.values)
    k = res.k
    cliques = cliques_within_cd(res.avg_ranks, res.cd_005)

    fig, ax = plt.subplots(figsize=(8, 1.6 + 0.32 * k))
    lo, hi = 1, k
    ax.set_xlim(hi + 0.5, lo - 0.5)  # invertido: rank 1 (melhor) à direita
    ax.set_ylim(0, 1)
    ax.axis("off")

    axis_y = 0.78
    ax.plot([lo, hi], [axis_y, axis_y], color="black", lw=1.2)
    for r in range(lo, hi + 1):
        ax.plot([r, r], [axis_y - 0.02, axis_y + 0.02], color="black", lw=1.2)
        ax.text(r, axis_y + 0.05, str(r), ha="center", va="bottom", fontsize=9)

    # CD bar de referência
    cd_y = axis_y + 0.16
    cd_left = hi - 0.5
    ax.plot([cd_left, cd_left - res.cd_005], [cd_y, cd_y], color="black", lw=1.2)
    ax.plot([cd_left, cd_left], [cd_y - 0.015, cd_y + 0.015], color="black", lw=1.2)
    ax.plot([cd_left - res.cd_005, cd_left - res.cd_005], [cd_y - 0.015, cd_y + 0.015], color="black", lw=1.2)
    ax.text((2 * cd_left - res.cd_005) / 2, cd_y + 0.04, f"CD = {res.cd_005:.2f}",
            ha="center", va="bottom", fontsize=9)

    # posições de rótulo: metade dos métodos (melhores ranks) rotulados à
    # direita do eixo, a outra metade (piores ranks) à esquerda — convenção
    # padrão de Demšar (2006), com uma linha pontilhada ligando cada tick
    # ao seu rótulo.
    n = len(names)
    half = (n + 1) // 2
    for idx, (name, r) in enumerate(zip(names, ranks)):
        if idx < half:
            y_text = axis_y - 0.10 - idx * 0.14
            ax.text(lo - 0.15, y_text, name, ha="left", va="center", fontsize=9, color="black")
            ax.plot([r, lo - 0.2], [y_text, y_text], color="gray", lw=0.6, linestyle=":")
        else:
            j = idx - half
            y_text = axis_y - 0.10 - j * 0.14
            ax.text(hi + 0.15, y_text, name, ha="right", va="center", fontsize=9, color="black")
            ax.plot([r, hi + 0.2], [y_text, y_text], color="gray", lw=0.6, linestyle=":")

    # barras conectando cliques estatisticamente indistinguíveis
    bar_y0 = axis_y - 0.04
    step = 0.045
    for i, clique in enumerate(cliques):
        if len(clique) < 2:
            continue
        clique_ranks = [avg_ranks[m] for m in clique]
        y = bar_y0 - i * step
        ax.plot([min(clique_ranks), max(clique_ranks)], [y, y], color="black", lw=2.2, solid_capstyle="butt")

    ax.set_title(title, fontsize=11, pad=18)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
