from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


RANDOM_STATE = 42
N_BOOTSTRAP = 10_000
ALPHA = 0.05

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

CONFIG_FILES = {
    "FFT only": "predictions_classification_biggan_midjourney_multiclass_fft_only.csv",
    "CLIP only": "predictions_classification_biggan_midjourney_multiclass_clip_only.csv",
    "IQA only": "predictions_classification_biggan_midjourney_multiclass_iqa_only.csv",
    "FFT + CLIP": "predictions_classification_biggan_midjourney_multiclass_fft_clip.csv",
    "IQA + FFT": "predictions_classification_biggan_midjourney_multiclass_iqa_fft.csv",
    "IQA + CLIP": "predictions_classification_biggan_midjourney_multiclass_iqa_clip.csv",
    "IQA + FFT + CLIP": "predictions_classification_biggan_midjourney_multiclass_iqa_fft_clip.csv",
}

COMPARISONS = [
    ("FFT", "IQA only", "IQA + FFT", "Menambahkan FFT pada IQA only"),
    ("CLIP", "IQA only", "IQA + CLIP", "Menambahkan CLIP pada IQA only"),
    ("FFT", "CLIP only", "FFT + CLIP", "Menambahkan FFT pada CLIP only"),
    ("FFT", "IQA + CLIP", "IQA + FFT + CLIP", "Menambahkan FFT pada IQA + CLIP"),
    ("IQA", "CLIP only", "IQA + CLIP", "Menambahkan IQA pada CLIP only"),
    ("CLIP", "FFT only", "FFT + CLIP", "Menambahkan CLIP pada FFT only"),
    ("IQA", "FFT only", "IQA + FFT", "Menambahkan IQA pada FFT only"),
    ("CLIP", "IQA + FFT", "IQA + FFT + CLIP", "Menambahkan CLIP pada IQA + FFT"),
    ("IQA", "FFT + CLIP", "IQA + FFT + CLIP", "Menambahkan IQA pada FFT + CLIP"),
]


def load_xgboost_predictions() -> dict[str, pd.DataFrame]:
    predictions: dict[str, pd.DataFrame] = {}
    reference_ids: list[str] | None = None
    reference_y: np.ndarray | None = None

    for config_name, filename in CONFIG_FILES.items():
        path = ARTIFACT_DIR / filename
        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)
        df = df[df["model"] == "XGBoost"].copy()
        df = df.sort_values("image_id").reset_index(drop=True)
        if df.empty:
            raise ValueError(f"Tidak ada prediksi XGBoost pada {filename}")

        image_ids = df["image_id"].tolist()
        y_true = df["y_ai"].to_numpy(dtype=int)

        if reference_ids is None:
            reference_ids = image_ids
            reference_y = y_true
        else:
            if image_ids != reference_ids:
                raise ValueError(f"Urutan image_id berbeda pada {filename}")
            if not np.array_equal(y_true, reference_y):
                raise ValueError(f"Label y_ai berbeda pada {filename}")

        predictions[config_name] = df

    return predictions


def metric_values(df: pd.DataFrame) -> dict[str, float]:
    y_true = df["y_ai"].to_numpy(dtype=int)
    y_pred = df["pred_ai"].to_numpy(dtype=int)
    score = df["score_ai"].to_numpy(dtype=float)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, score)),
    }


def bootstrap_accuracy_difference(
    baseline_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    rng: np.random.Generator,
) -> dict[str, float | bool]:
    n = len(baseline_df)
    baseline_correct = (
        baseline_df["y_ai"].to_numpy(dtype=int) == baseline_df["pred_ai"].to_numpy(dtype=int)
    ).astype(float)
    comparison_correct = (
        comparison_df["y_ai"].to_numpy(dtype=int) == comparison_df["pred_ai"].to_numpy(dtype=int)
    ).astype(float)

    observed = comparison_correct.mean() - baseline_correct.mean()
    indices = rng.integers(0, n, size=(N_BOOTSTRAP, n), dtype=np.int32)
    diffs = comparison_correct[indices].mean(axis=1) - baseline_correct[indices].mean(axis=1)

    ci_low, ci_high = np.quantile(diffs, [ALPHA / 2, 1 - ALPHA / 2])
    p_lower = (np.sum(diffs <= 0) + 1) / (N_BOOTSTRAP + 1)
    p_upper = (np.sum(diffs >= 0) + 1) / (N_BOOTSTRAP + 1)
    p_value = min(1.0, 2 * min(p_lower, p_upper))

    return {
        "accuracy_diff": float(observed),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_bootstrap": float(p_value),
        "significant": bool(ci_low > 0 or ci_high < 0),
    }


def main() -> None:
    predictions = load_xgboost_predictions()
    rng = np.random.default_rng(RANDOM_STATE)

    summary_rows = []
    for config_name, df in predictions.items():
        metrics = metric_values(df)
        y_true = df["y_ai"].to_numpy(dtype=int)
        y_pred = df["pred_ai"].to_numpy(dtype=int)
        summary_rows.append(
            {
                "feature_set": config_name,
                "model": "XGBoost",
                **metrics,
                "n_eval": len(df),
                "n_wrong": int((y_true != y_pred).sum()),
            }
        )

    comparison_rows = []
    for feature_added, baseline, comparison, interpretation in COMPARISONS:
        result = bootstrap_accuracy_difference(predictions[baseline], predictions[comparison], rng)
        comparison_rows.append(
            {
                "feature_added": feature_added,
                "baseline": baseline,
                "comparison": comparison,
                "interpretation": interpretation,
                "baseline_accuracy": metric_values(predictions[baseline])["accuracy"],
                "comparison_accuracy": metric_values(predictions[comparison])["accuracy"],
                **result,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    comparison_df = pd.DataFrame(comparison_rows)

    summary_out = ARTIFACT_DIR / "bootstrap_multiclass_xgboost_feature_summary.csv"
    comparison_out = ARTIFACT_DIR / "bootstrap_multiclass_xgboost_feature_ablation.csv"
    report_out = ARTIFACT_DIR / "bootstrap_multiclass_xgboost_report.txt"

    summary_df.to_csv(summary_out, index=False)
    comparison_df.to_csv(comparison_out, index=False)

    strongest = comparison_df.sort_values("accuracy_diff", ascending=False).iloc[0]
    with report_out.open("w", encoding="utf-8") as handle:
        handle.write("Paired bootstrap feature comparison on XGBoost predictions\n")
        handle.write(f"n_bootstrap={N_BOOTSTRAP}, alpha={ALPHA}, random_state={RANDOM_STATE}\n\n")
        handle.write("Feature summary\n")
        for _, row in summary_df.iterrows():
            handle.write(
                f"- {row['feature_set']}: accuracy={row['accuracy']:.6f}, "
                f"f1={row['f1']:.6f}, auroc={row['auroc']:.6f}, wrong={int(row['n_wrong'])}\n"
            )
        handle.write("\nAblation comparisons based on XGBoost accuracy\n")
        for _, row in comparison_df.sort_values("accuracy_diff", ascending=False).iterrows():
            handle.write(
                f"- {row['interpretation']}: diff={row['accuracy_diff']:.6f}, "
                f"95% CI=[{row['ci_low']:.6f}, {row['ci_high']:.6f}], "
                f"p={row['p_bootstrap']:.6g}, significant={row['significant']}\n"
            )
        handle.write("\nStrongest feature contribution\n")
        handle.write(
            f"{strongest['interpretation']} gives the largest observed accuracy gain "
            f"({strongest['accuracy_diff']:.6f}).\n"
        )

    print("Saved:", summary_out)
    print("Saved:", comparison_out)
    print("Saved:", report_out)


if __name__ == "__main__":
    main()
