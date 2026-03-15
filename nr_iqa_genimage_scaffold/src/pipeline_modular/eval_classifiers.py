"""Summarize classifier results table into a ranked markdown report."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate/rank classifier results")
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/results_classification.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/results_classification_summary.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.metrics)
    if df.empty:
        raise RuntimeError("Metrics file is empty")

    rank_df = df.sort_values(by=["auroc", "f1", "accuracy"], ascending=False).reset_index(drop=True)

    lines = []
    lines.append("# Classification Summary\n")
    lines.append("Ranked by AUROC -> F1 -> Accuracy\n")
    lines.append("| Rank | Model | Accuracy | F1 | AUROC | AUPRC |")
    lines.append("|---:|---|---:|---:|---:|---:|")

    for i, row in rank_df.iterrows():
        lines.append(
            f"| {i+1} | {row['model']} | {row['accuracy']:.4f} | {row['f1']:.4f} | {row['auroc']:.4f} | {row['auprc']:.4f} |"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] Best model: {rank_df.iloc[0]['model']}")
    print(f"[OK] Wrote summary: {args.output.resolve()}")


if __name__ == "__main__":
    main()
