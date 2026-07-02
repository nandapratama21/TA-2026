from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
OUT_DIR = PROJECT_ROOT / "assets" / "pics"

EXPERIMENTS = [
    {
        "experiment_id": "fft_only",
        "feature_label": "FFT only",
        "metrics": "results_classification_biggan_midjourney_cat4_fft_only.csv",
        "predictions": "predictions_classification_biggan_midjourney_cat4_fft_only.csv",
    },
    {
        "experiment_id": "clip_only",
        "feature_label": "CLIP only",
        "metrics": "results_classification_biggan_midjourney_cat4_clip_only.csv",
        "predictions": "predictions_classification_biggan_midjourney_cat4_clip_only.csv",
    },
    {
        "experiment_id": "iqa_only",
        "feature_label": "IQA only",
        "metrics": "results_classification_biggan_midjourney_cat4_iqa_only.csv",
        "predictions": "predictions_classification_biggan_midjourney_cat4_iqa_only.csv",
    },
    {
        "experiment_id": "fft_clip",
        "feature_label": "FFT + CLIP",
        "metrics": "results_classification_biggan_midjourney_cat4_fft_clip.csv",
        "predictions": "predictions_classification_biggan_midjourney_cat4_fft_clip.csv",
    },
    {
        "experiment_id": "iqa_fft",
        "feature_label": "IQA + FFT",
        "metrics": "results_classification_biggan_midjourney_cat4_iqa_fft.csv",
        "predictions": "predictions_classification_biggan_midjourney_cat4_iqa_fft.csv",
    },
    {
        "experiment_id": "iqa_clip",
        "feature_label": "IQA + CLIP",
        "metrics": "results_classification_biggan_midjourney_cat4_iqa_clip.csv",
        "predictions": "predictions_classification_biggan_midjourney_cat4_iqa_clip.csv",
    },
    {
        "experiment_id": "iqa_fft_clip",
        "feature_label": "IQA + FFT + CLIP",
        "metrics": "results_classification_biggan_midjourney_cat4_iqa_fft_clip.csv",
        "predictions": "predictions_classification_biggan_midjourney_cat4_iqa_fft_clip.csv",
    },
]


def error_type(row):
    y = int(row["y_ai"])
    p = int(row["pred_ai"])
    if y == 1 and p == 1:
        return "TP_ai_correct"
    if y == 0 and p == 0:
        return "TN_nature_correct"
    if y == 0 and p == 1:
        return "FP_nature_as_ai"
    return "FN_ai_as_nature"


def load_data():
    metrics_frames = []
    pred_frames = []

    for experiment in EXPERIMENTS:
        metrics = pd.read_csv(ARTIFACT_DIR / experiment["metrics"])
        preds = pd.read_csv(ARTIFACT_DIR / experiment["predictions"])

        for key, value in experiment.items():
            if key not in {"metrics", "predictions"}:
                metrics[key] = value
                preds[key] = value

        metrics_frames.append(metrics)
        pred_frames.append(preds)

    metrics_all = pd.concat(metrics_frames, ignore_index=True)
    preds_all = pd.concat(pred_frames, ignore_index=True)

    # Match notebook 15: accuracy, then f1, then auroc.
    best_metrics = (
        metrics_all.sort_values(
            ["feature_label", "accuracy", "f1", "auroc"],
            ascending=[True, False, False, False],
        )
        .groupby("feature_label", as_index=False)
        .head(1)
        .sort_values("accuracy", ascending=False)
        .reset_index(drop=True)
    )

    best_pairs = set(zip(best_metrics["experiment_id"], best_metrics["model"]))
    best_preds = preds_all[
        preds_all.apply(lambda r: (r["experiment_id"], r["model"]) in best_pairs, axis=1)
    ].copy()
    best_preds["label_text"] = best_preds["y_ai"].map({0: "nature", 1: "AI"})
    best_preds["pred_text"] = best_preds["pred_ai"].map({0: "nature", 1: "AI"})
    best_preds["error_type"] = best_preds.apply(error_type, axis=1)

    return best_metrics, best_preds


def select_examples(df, kind, max_images=8):
    sub = df[df["error_type"] == kind].copy()
    if sub.empty:
        return sub

    if kind == "FP_nature_as_ai":
        return sub.sort_values("score_ai", ascending=False).head(max_images)
    if kind == "FN_ai_as_nature":
        return sub.sort_values("score_ai", ascending=True).head(max_images)
    return sub.head(max_images)


def slugify(text):
    return (
        text.lower()
        .replace(" + ", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )


def draw_group(axes, rows, group_title):
    axes[0].set_ylabel(group_title, fontsize=15, fontweight="bold")

    if rows.empty:
        for ax in axes:
            ax.axis("off")
        axes[0].text(0.5, 0.5, "Tidak ada sampel", ha="center", va="center", fontsize=13)
        return

    for ax, row in zip(axes, rows.itertuples(index=False)):
        try:
            img = Image.open(row.path).convert("RGB")
            ax.imshow(img)
        except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
            ax.text(0.5, 0.5, f"Gambar tidak dapat dibuka\n{exc}", ha="center", va="center", fontsize=9)

        content = getattr(row, "content_label", getattr(row, "class_name", "-"))
        generator = getattr(row, "generator", "-")
        filename = Path(row.path).name
        ax.set_title(
            f"GT: {row.label_text} | Pred: {row.pred_text}\n"
            f"AI prob: {float(row.score_ai):.4f}\n"
            f"{generator} | {content}\n"
            f"{filename}",
            fontsize=9,
            pad=6,
        )
        ax.axis("off")

    for ax in axes[len(rows):]:
        ax.axis("off")


def make_false_detection_figure(sub, feature_label, model, max_per_group=4):
    fp = select_examples(sub, "FP_nature_as_ai", max_per_group)
    fn = select_examples(sub, "FN_ai_as_nature", max_per_group)

    ncols = max(1, len(fp), len(fn))
    fig, axes = plt.subplots(2, ncols, figsize=(4.1 * ncols, 7.8))
    axes = np.asarray(axes).reshape(2, ncols)

    draw_group(axes[0], fp, "False Positive\nnature -> AI")
    draw_group(axes[1], fn, "False Negative\nAI -> nature")

    fig.suptitle(f"{feature_label} ({model})", fontsize=18, fontweight="bold", y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    out_name = f"15_false_detection_{slugify(feature_label)}_{model.lower()}.png"
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return out_path, len(fp), len(fn)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best_metrics, best_preds = load_data()

    rows = []
    for row in best_metrics.sort_values("feature_label").itertuples(index=False):
        sub = best_preds[
            (best_preds["experiment_id"] == row.experiment_id)
            & (best_preds["model"] == row.model)
        ].copy()
        out_path, fp_count, fn_count = make_false_detection_figure(
            sub, row.feature_label, row.model
        )
        rows.append(
            {
                "feature_label": row.feature_label,
                "model": row.model,
                "figure": out_path.relative_to(PROJECT_ROOT).as_posix(),
                "shown_fp": fp_count,
                "shown_fn": fn_count,
            }
        )
        print(out_path.relative_to(PROJECT_ROOT).as_posix())

    pd.DataFrame(rows).to_csv(
        ARTIFACT_DIR / "notebook15_false_detection_figures.csv", index=False
    )


if __name__ == "__main__":
    main()
