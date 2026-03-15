"""Dummy entrypoint evaluasi model utama + evaluasi semantik opsional."""

from __future__ import annotations

import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.pipeline_modular.eval import evaluate_with_labels


def main() -> None:
    root = ROOT
    pred_path = os.path.join(root, "artifacts", "predictions_dummy.csv")
    labels_path = os.path.join(root, "data", "labels_quality.csv")
    out_path = os.path.join(root, "artifacts", "metrics_dummy.csv")

    pred = pd.read_csv(pred_path)
    labels = pd.read_csv(labels_path)
    metrics = evaluate_with_labels(pred, labels)
    metrics.to_csv(out_path, index=False)
    print(f"[DUMMY] Wrote eval metrics: {out_path}")
    print("[DUMMY] Optional Gemini subset analysis remains manual.")


if __name__ == "__main__":
    main()
