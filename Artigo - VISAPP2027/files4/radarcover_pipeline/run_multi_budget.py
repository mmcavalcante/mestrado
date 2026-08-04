#!/usr/bin/env python3
"""Ponto de entrada do estudo multi-orçamento (K=2,4,6), modelo OPFsembleR."""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from multi_budget import run_multi_budget_study  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--budgets", default="2,4,6")
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    budgets = tuple(int(x) for x in args.budgets.split(","))
    seeds = tuple(int(x) for x in args.seeds.split(","))

    result = run_multi_budget_study(cfg, budgets=budgets, n_seeds=seeds)

    print("\n=== Resumo final ===")
    for K in budgets:
        print(f"\n--- Orçamento K={K} ---")
        from stats_test import format_result_text
        print(format_result_text(result["stats"][K]))


if __name__ == "__main__":
    main()
