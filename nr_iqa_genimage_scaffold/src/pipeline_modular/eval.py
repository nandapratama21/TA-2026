"""Dummy evaluation helpers."""

from __future__ import annotations

import pandas as pd


def evaluate_with_labels(pred_df: pd.DataFrame, label_df: pd.DataFrame) -> pd.DataFrame:
    merged = pred_df.merge(label_df, on="image_id", how="inner")
    if merged.empty:
        return pd.DataFrame([{"metric": "note", "value": "no_overlapping_labels"}])

    rank_pred = merged["pred_score"].rank(method="average")
    rank_gt = merged["quality_score"].rank(method="average")
    corr = rank_pred.corr(rank_gt, method="pearson")
    mae = (merged["pred_score"] - merged["quality_score"]).abs().mean()
    return pd.DataFrame(
        [
            {"metric": "dummy_srocc", "value": float(corr) if corr == corr else 0.0},
            {"metric": "dummy_mae", "value": float(mae)},
        ]
    )
