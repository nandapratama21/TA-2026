from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


PROJECT = Path("/home/nanda/folder belajar/TA-2026/nr_iqa_genimage_scaffold")
LATEX = Path("/home/nanda/folder belajar/TA-2026/skripsi_nanda_zip_work")
FIG_DIR = PROJECT / "artifacts/figures"
PIC_DIR = LATEX / "assets/pics"

RANDOM_STATE = 42
PALETTE = {"Nature": "#2ca25f", "AI-generated": "#de2d26"}
MARKERS = {"BigGAN": "o", "MidJourney": "X"}


CONFIGS = [
    {
        "feature_file": "feature_vector_biggan_midjourney_multiclass_clip_only.csv",
        "title": "t-SNE Visualisasi Fitur CLIP",
        "stem": "16_clip_tsne",
        "feature_kind": "clip",
    },
    {
        "feature_file": "feature_vector_biggan_midjourney_multiclass_fft_clip.csv",
        "title": "t-SNE Visualisasi Fitur FFT + CLIP",
        "stem": "16_fft_clip_tsne",
        "feature_kind": "fft_clip",
    },
    {
        "feature_file": "feature_vector_biggan_midjourney_multiclass_iqa_clip.csv",
        "title": "t-SNE Visualisasi Fitur IQA + CLIP",
        "stem": "16_iqa_clip_tsne",
        "feature_kind": "iqa_clip",
    },
    {
        "feature_file": "feature_vector_biggan_midjourney_multiclass_iqa_fft_clip.csv",
        "title": "t-SNE Visualisasi Fitur IQA + FFT + CLIP",
        "stem": "16_all_features_tsne",
        "feature_kind": "iqa_fft_clip",
    },
]


def feature_columns(df: pd.DataFrame, feature_kind: str) -> list[str]:
    cols = []
    if "fft" in feature_kind:
        cols.extend([col for col in df.columns if col.startswith("fft_")])
    if "clip" in feature_kind:
        cols.extend([col for col in df.columns if col.startswith("clip_feat_")])
    if "iqa" in feature_kind:
        cols.extend([col for col in ["piqe", "brisque", "niqe"] if col in df.columns])
    return cols


def compute_tsne(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    x = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x_scaled = StandardScaler().fit_transform(x)
    coords = TSNE(
        n_components=2,
        perplexity=30,
        init="random",
        learning_rate="auto",
        random_state=RANDOM_STATE,
    ).fit_transform(x_scaled)

    out = df[["image_id", "generator", "content_label", "y_ai"]].copy()
    out["label_text"] = out["y_ai"].map({0: "Nature", 1: "AI-generated"})
    out["tsne_1"] = coords[:, 0]
    out["tsne_2"] = coords[:, 1]
    return out


def scatter_by_order(ax, data: pd.DataFrame, label_order: list[str]) -> None:
    for label in label_order:
        for generator in ["BigGAN", "MidJourney"]:
            part = data[(data["label_text"] == label) & (data["generator"] == generator)]
            ax.scatter(
                part["tsne_1"],
                part["tsne_2"],
                s=26,
                marker=MARKERS[generator],
                c=PALETTE[label],
                alpha=0.58,
                linewidths=0,
                label=f"{label} - {generator}",
            )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.28)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(
        unique.values(),
        unique.keys(),
        title="label - generator",
        fontsize=9,
        title_fontsize=10,
        loc="best",
        frameon=True,
    )


def save_plot(data: pd.DataFrame, title: str, stem: str, top_label: str) -> None:
    label_order = ["Nature", "AI-generated"] if top_label == "AI-generated" else ["AI-generated", "Nature"]
    suffix = "ai_on_top" if top_label == "AI-generated" else "nature_on_top"
    fig, ax = plt.subplots(figsize=(10, 7.2), constrained_layout=True)
    scatter_by_order(ax, data, label_order)
    ax.set_title(title)
    out_name = f"{stem}_{suffix}.png"
    fig.savefig(FIG_DIR / out_name, dpi=220)
    fig.savefig(PIC_DIR / out_name, dpi=220)
    plt.close(fig)


def save_by_label(data: pd.DataFrame, title: str, stem: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True, constrained_layout=True)
    for ax, label in zip(axes, ["Nature", "AI-generated"]):
        subset = data[data["label_text"] == label]
        for generator in ["BigGAN", "MidJourney"]:
            part = subset[subset["generator"] == generator]
            ax.scatter(
                part["tsne_1"],
                part["tsne_2"],
                s=24,
                marker=MARKERS[generator],
                c=PALETTE[label],
                alpha=0.62,
                linewidths=0,
                label=generator,
                rasterized=True,
            )
        ax.set_title(f"{title} - {label}")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(True, alpha=0.28)
        ax.legend(title="generator", loc="best", frameon=True)
    out_name = f"{stem}_by_label.png"
    fig.savefig(FIG_DIR / out_name, dpi=220)
    fig.savefig(PIC_DIR / out_name, dpi=220)
    plt.close(fig)


def robust_stats(data: pd.DataFrame, stem: str) -> pd.DataFrame:
    rows = []
    for keys, part in data.groupby(["label_text", "generator"]):
        label, generator = keys
        rows.append(
            {
                "figure": stem,
                "label": label,
                "generator": generator,
                "n": len(part),
                "tsne_1_min": part["tsne_1"].min(),
                "tsne_1_q05": part["tsne_1"].quantile(0.05),
                "tsne_1_q95": part["tsne_1"].quantile(0.95),
                "tsne_1_max": part["tsne_1"].max(),
                "tsne_2_min": part["tsne_2"].min(),
                "tsne_2_q05": part["tsne_2"].quantile(0.05),
                "tsne_2_q95": part["tsne_2"].quantile(0.95),
                "tsne_2_max": part["tsne_2"].max(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PIC_DIR.mkdir(parents=True, exist_ok=True)

    all_stats = []
    for cfg in CONFIGS:
        df = pd.read_csv(PROJECT / "artifacts" / cfg["feature_file"])
        cols = feature_columns(df, cfg["feature_kind"])
        print(f"{cfg['stem']}: {len(cols)} fitur, {len(df)} citra")
        tsne_df = compute_tsne(df, cols)
        tsne_df.to_csv(FIG_DIR / f"{cfg['stem']}_coordinates.csv", index=False)
        save_plot(tsne_df, cfg["title"], cfg["stem"], "AI-generated")
        save_plot(tsne_df, cfg["title"], cfg["stem"], "Nature")
        save_by_label(tsne_df, cfg["title"], cfg["stem"])
        all_stats.append(robust_stats(tsne_df, cfg["stem"]))

    stats = pd.concat(all_stats, ignore_index=True)
    stats.to_csv(FIG_DIR / "16_tsne_visualization_ranges.csv", index=False)
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
