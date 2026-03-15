"""Feature fusion and dummy score prediction."""

from __future__ import annotations

import pandas as pd


def fuse_features(fft_df: pd.DataFrame, clip_df: pd.DataFrame) -> pd.DataFrame:
    return fft_df.merge(clip_df, on="image_id", how="inner")


def predict_dummy_score(fused_df: pd.DataFrame) -> pd.DataFrame:
    out = fused_df[["image_id"]].copy()
    out["pred_score"] = (
        0.35 * _safe_col(fused_df, "fft_low_energy")
        + 0.25 * _safe_col(fused_df, "fft_mid_energy")
        + 0.15 * _safe_col(fused_df, "fft_high_energy")
        + 0.25 * _safe_col(fused_df, "clip_feat_001")
    )
    out["pred_score"] = (out["pred_score"] - out["pred_score"].min()) / (
        (out["pred_score"].max() - out["pred_score"].min()) + 1e-8
    )
    out["pred_label_optional"] = out["pred_score"].apply(
        lambda x: "real_like" if x >= 0.5 else "ai_like"
    )
    out["uncertainty"] = (0.5 - (out["pred_score"] - 0.5).abs()) * 2.0
    return out


def _safe_col(df: pd.DataFrame, col: str):
    if col in df.columns:
        return df[col]
    return 0.0
