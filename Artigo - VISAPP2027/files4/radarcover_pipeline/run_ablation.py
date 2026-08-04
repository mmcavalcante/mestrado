#!/usr/bin/env python3
"""Ponto de entrada do Ablation Study (Capítulo 6)."""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from ablation import run_ablation_study  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    df = run_ablation_study(cfg)
    print("\n=== Tabela do Ablation Study (Capítulo 6) ===")
    print(df.drop(columns=["selected_models"]).to_string(index=False))


if __name__ == "__main__":
    main()
