import json
from pathlib import Path


HELPER_CODE = '''def layered_label_scatter_with_order(ax, data, xcol, ycol, title, top_label="AI", style_col="generator"):
    """Plot dua label dengan pilihan label yang digambar paling atas."""
    marker_map = {"BigGAN": "o", "MidJourney": "X"}
    label_styles = {
        "Nature": {"color": "#2ca25f"},
        "AI": {"color": "#de2d26"},
    }
    order = ["Nature", "AI"]
    if top_label == "Nature":
        order = ["AI", "Nature"]

    for zorder, label in enumerate(order, start=1):
        label_df = data[data["label"] == label]
        is_top = label == top_label
        for generator, marker in marker_map.items():
            subset = label_df[label_df[style_col] == generator]
            if subset.empty:
                continue
            ax.scatter(
                subset[xcol],
                subset[ycol],
                s=36 if is_top else 28,
                c=label_styles[label]["color"],
                marker=marker,
                alpha=0.82 if is_top else 0.28,
                edgecolors="black" if is_top else "none",
                linewidths=0.25 if is_top else 0,
                label=f"{label} - {generator}",
                zorder=zorder,
                rasterized=True,
            )

    ax.set_title(title)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.grid(True, alpha=0.25)
'''


FFT_NATURE_TOP_CELL = '''fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
layered_label_scatter_with_order(
    axes[0],
    fft_df,
    "fft_mag_mean",
    "fft_phase_mean",
    "FFT Mean Magnitude vs Mean Phase",
    top_label="Nature",
)
layered_label_scatter_with_order(
    axes[1],
    fft_df,
    "fft_phase_cos_mean",
    "fft_phase_sin_mean",
    "FFT Phase Circular Features",
    top_label="Nature",
)
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=True)
plt.suptitle("FFT Scatter Plot with Nature Points Drawn on Top", y=1.03)
plt.subplots_adjust(bottom=0.22)
savefig("16_fft_scatter_nature_on_top.png")
plt.show()
'''


IQA_SCATTER_CELL = '''iqa_cols = ["piqe", "brisque", "niqe"]
iqa_vis = iqa_df[["image_id", "generator", "class_name", "label", "y_ai", *iqa_cols]].copy()

display(iqa_vis[iqa_cols].describe())

pairs = [("piqe", "brisque"), ("piqe", "niqe"), ("brisque", "niqe")]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (xcol, ycol) in zip(axes, pairs):
    layered_label_scatter_with_order(
        ax,
        iqa_vis,
        xcol,
        ycol,
        f"{xcol.upper()} vs {ycol.upper()}",
        top_label="AI",
    )
handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=True)
plt.suptitle("NR-IQA Scatter Plot with AI Points Drawn on Top", y=1.04)
plt.subplots_adjust(bottom=0.23)
savefig("16_iqa_scatter_ai_on_top.png")
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (xcol, ycol) in zip(axes, pairs):
    layered_label_scatter_with_order(
        ax,
        iqa_vis,
        xcol,
        ycol,
        f"{xcol.upper()} vs {ycol.upper()}",
        top_label="Nature",
    )
handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=True)
plt.suptitle("NR-IQA Scatter Plot with Nature Points Drawn on Top", y=1.04)
plt.subplots_adjust(bottom=0.23)
savefig("16_iqa_scatter_nature_on_top.png")
plt.show()


fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex="col", sharey="col")
for row_idx, label in enumerate(["Nature", "AI"]):
    subset = iqa_vis[iqa_vis["label"] == label]
    for col_idx, (xcol, ycol) in enumerate(pairs):
        ax = axes[row_idx, col_idx]
        sns.scatterplot(
            data=subset,
            x=xcol,
            y=ycol,
            hue="generator",
            style="generator",
            alpha=0.65,
            s=32,
            ax=ax,
        )
        ax.set_title(f"{label}: {xcol.upper()} vs {ycol.upper()}")
        ax.grid(True, alpha=0.25)

plt.suptitle("NR-IQA Scatter Plot Separated by Label", y=1.02)
savefig("16_iqa_scatter_by_label.png")
plt.show()
'''


def set_source(cell: dict, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)


def patch(path: Path) -> bool:
    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = False

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "PALETTE =" in src and "layered_label_scatter_with_order" not in src:
            set_source(cell, src.rstrip() + "\n\n" + HELPER_CODE)
            changed = True
            break

    # Add Nature-on-top FFT after the existing AI-on-top FFT cell.
    for idx, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", []))
        if cell.get("cell_type") == "code" and 'savefig("16_fft_scatter_ai_on_top.png")' in src:
            already = any(
                'savefig("16_fft_scatter_nature_on_top.png")' in "".join(c.get("source", []))
                for c in nb["cells"]
            )
            if not already:
                nb["cells"].insert(
                    idx + 1,
                    {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": FFT_NATURE_TOP_CELL.splitlines(keepends=True),
                    },
                )
                changed = True
            break

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if (
            'iqa_cols = ["piqe", "brisque", "niqe"]' in src
            and 'savefig("16_iqa_scatter_ai_vs_nature.png")' in src
        ):
            set_source(cell, IQA_SCATTER_CELL)
            changed = True

    if changed:
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return changed


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    targets = [
        root / "notebooks" / "16_feature_space_visualization_from_saved_csv_multiclass.ipynb",
        root / "notebooks" / "16_feature_space_visualization_from_saved_csv.ipynb",
    ]
    for path in targets:
        if path.exists() and patch(path):
            print("patched", path.relative_to(root))


if __name__ == "__main__":
    main()
