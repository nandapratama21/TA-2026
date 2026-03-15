"""Dummy entrypoint fusion FFT+CLIP dan regresi skor NR-IQA."""

from __future__ import annotations

import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.pipeline_modular.fusion import fuse_features, predict_dummy_score


def main() -> None:
    root = ROOT
    fft_path = os.path.join(root, "artifacts", "features_fft_dummy.csv")
    clip_path = os.path.join(root, "artifacts", "features_clip_dummy.csv")
    out_path = os.path.join(root, "artifacts", "predictions_dummy.csv")

    fft_df = pd.read_csv(fft_path)
    clip_df = pd.read_csv(clip_path)
    fused = fuse_features(fft_df, clip_df)
    pred = predict_dummy_score(fused)
    pred.to_csv(out_path, index=False)
    print(f"[DUMMY] Wrote predictions: {out_path}")


if __name__ == "__main__":
    main()
