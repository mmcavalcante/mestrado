#!/usr/bin/env python3
"""Ponto de entrada do pipeline RadarCover (Etapas 1-9)."""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from pipeline import run_pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    result = run_pipeline(cfg)

    print("\n=== Tabela final de comparação (Etapa 9) ===")
    print(result["results_table"].to_string(index=False))


if __name__ == "__main__":
    main()
