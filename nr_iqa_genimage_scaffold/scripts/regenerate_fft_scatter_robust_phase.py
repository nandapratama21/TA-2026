from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT = Path("/home/nanda/folder belajar/TA-2026/nr_iqa_genimage_scaffold")
LATEX = Path("/home/nanda/folder belajar/TA-2026/skripsi_nanda_zip_work")
FIG_DIR = PROJECT / "artifacts/figures"
PIC_DIR = LATEX / "assets/pics"
FEATURE_PATH = PROJECT / "artifacts/feature_vector_biggan_midjourney_multiclass_fft_only.csv"

PALETTE = {"Nature": "#2ca25f", "AI-generated": "#de2d26"}
MARKERS = {"BigGAN": "o", "MidJourney": "X"}
PHASE_SIN_LIMIT = 8e-9


def prepare_df() -> pd.DataFrame:
    df = pd.read_csv(FEATURE_PATH)
    df["label_text"] = df["y_ai"].map({0: "Nature", 1: "AI-generated"})
    return df


def scatter_by_order(ax, df: pd.DataFrame, x: str, y: str, label_order, title: str, xlabel: str, ylabel: str):
    for label in label_order:
        for generator in ["BigGAN", "MidJourney"]:
            part = df[(df["label_text"] == label) & (df["generator"] == generator)]
            ax.scatter(
                part[x],
                part[y],
                s=28,
                marker=MARKERS[generator],
                c=PALETTE[label],
                alpha=0.72,
                linewidths=0,
                label=f"{label} - {generator}",
            )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.28)


def save_combined(df: pd.DataFrame, out_name: str, top_label: str):
    if top_label == "AI-generated":
        label_order = ["Nature", "AI-generated"]
    else:
        label_order = ["AI-generated", "Nature"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    scatter_by_order(
        axes[0],
        df,
        "fft_mag_mean",
        "fft_phase_mean",
        label_order,
        "FFT Mean Magnitude vs Mean Phase",
        "Mean Magnitude",
        "Mean Phase",
    )
    scatter_by_order(
        axes[1],
        df,
        "fft_phase_cos_mean",
        "fft_phase_sin_mean",
        label_order,
        "FFT Phase Circular Features",
        "Mean cos(phase)",
        "Mean sin(phase)",
    )
    axes[1].set_ylim(-PHASE_SIN_LIMIT, PHASE_SIN_LIMIT)
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(-9, -9))

    # Keep one compact legend for both subplots.
    handles, labels = axes[1].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[0].legend_.remove() if axes[0].legend_ else None
    axes[1].legend(
        unique.values(),
        unique.keys(),
        title="label - generator",
        fontsize=8,
        title_fontsize=9,
        loc="best",
        frameon=True,
    )
    fig.savefig(FIG_DIR / out_name, dpi=220)
    fig.savefig(PIC_DIR / out_name, dpi=220)
    plt.close(fig)


def save_standalone_phase(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    scatter_by_order(
        ax,
        df,
        "fft_phase_cos_mean",
        "fft_phase_sin_mean",
        ["Nature", "AI-generated"],
        "FFT Phase Circular Features",
        "Mean cos(phase)",
        "Mean sin(phase)",
    )
    ax.set_ylim(-PHASE_SIN_LIMIT, PHASE_SIN_LIMIT)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-9, -9))
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), title="label - generator", loc="best", frameon=True)
    for name in ["16_fft_scatter_phase_circular.png"]:
        fig.savefig(FIG_DIR / name, dpi=220)
        fig.savefig(PIC_DIR / name, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PIC_DIR.mkdir(parents=True, exist_ok=True)
    data = prepare_df()
    save_combined(data, "16_fft_scatter_ai_on_top.png", "AI-generated")
    save_combined(data, "16_fft_scatter_nature_on_top.png", "Nature")
    save_combined(data, "16_fft_scatter_ai_vs_nature.png", "AI-generated")
    save_combined(data, "16_fft_scatter_by_label.png", "AI-generated")
    save_standalone_phase(data)
    print("Regenerated FFT scatter figures with robust phase-sin axis.")
