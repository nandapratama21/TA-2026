from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT = Path("/home/nanda/folder belajar/TA-2026/nr_iqa_genimage_scaffold")
LATEX = Path("/home/nanda/folder belajar/TA-2026/skripsi_nanda_zip_work")
FIG_DIR = PROJECT / "artifacts/figures"
PIC_DIR = LATEX / "assets/pics"
FEATURE_PATH = PROJECT / "artifacts/feature_vector_biggan_midjourney_multiclass_iqa_only.csv"

PALETTE = {"Nature": "#2ca25f", "AI-generated": "#de2d26"}
MARKERS = {"BigGAN": "o", "MidJourney": "X"}
PAIRS = [
    ("piqe", "brisque", "PIQE", "BRISQUE", "piqe_brisque"),
    ("piqe", "niqe", "PIQE", "NIQE", "piqe_niqe"),
    ("brisque", "niqe", "BRISQUE", "NIQE", "brisque_niqe"),
]


def prepare_df() -> pd.DataFrame:
    df = pd.read_csv(FEATURE_PATH)
    df["label_text"] = df["y_ai"].map({0: "Nature", 1: "AI-generated"})
    return df


def scatter_pair(ax, df: pd.DataFrame, x: str, y: str, top_label: str) -> None:
    label_order = ["Nature", "AI-generated"]
    if top_label == "Nature":
        label_order = ["AI-generated", "Nature"]

    for label in label_order:
        is_top = label == top_label
        for generator in ["BigGAN", "MidJourney"]:
            part = df[(df["label_text"] == label) & (df["generator"] == generator)]
            ax.scatter(
                part[x],
                part[y],
                s=34 if is_top else 24,
                marker=MARKERS[generator],
                c=PALETTE[label],
                alpha=0.78 if is_top else 0.30,
                edgecolors="black" if is_top else "none",
                linewidths=0.25 if is_top else 0,
                label=f"{label} - {generator}",
                rasterized=True,
            )
    ax.set_xlabel(x.upper())
    ax.set_ylabel(y.upper())
    ax.grid(True, alpha=0.28)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), title="label - generator", fontsize=8, title_fontsize=9, loc="best")


def save_pair(df: pd.DataFrame, x: str, y: str, x_title: str, y_title: str, stem: str, top_label: str) -> None:
    suffix = "ai_on_top" if top_label == "AI-generated" else "nature_on_top"
    title_label = "AI-generated" if top_label == "AI-generated" else "Nature"
    fig, ax = plt.subplots(figsize=(8.4, 6.3), constrained_layout=True)
    scatter_pair(ax, df, x, y, top_label)
    ax.set_title(f"{x_title} vs {y_title} ({title_label} on top)")
    out_name = f"16_iqa_scatter_{stem}_{suffix}.png"
    fig.savefig(FIG_DIR / out_name, dpi=220)
    fig.savefig(PIC_DIR / out_name, dpi=220)
    plt.close(fig)

    # Keep the old three filenames usable; use AI on top as the default view.
    if top_label == "AI-generated":
        legacy_name = f"16_iqa_scatter_{stem}.png"
        fig, ax = plt.subplots(figsize=(8.4, 6.3), constrained_layout=True)
        scatter_pair(ax, df, x, y, top_label)
        ax.set_title(f"{x_title} vs {y_title}")
        fig.savefig(FIG_DIR / legacy_name, dpi=220)
        fig.savefig(PIC_DIR / legacy_name, dpi=220)
        plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PIC_DIR.mkdir(parents=True, exist_ok=True)
    df = prepare_df()
    for x, y, x_title, y_title, stem in PAIRS:
        save_pair(df, x, y, x_title, y_title, stem, "AI-generated")
        save_pair(df, x, y, x_title, y_title, stem, "Nature")
    print("Regenerated separated IQA pair scatter figures.")


if __name__ == "__main__":
    main()
