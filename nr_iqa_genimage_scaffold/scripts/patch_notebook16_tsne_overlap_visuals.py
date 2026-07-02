import json
from pathlib import Path


HELPER_CODE = '''def layered_label_scatter(ax, data, xcol, ycol, title, style_col="generator"):
    """Gambar Nature lebih dulu, lalu AI di atasnya agar titik AI tidak tertutup."""
    marker_map = {"BigGAN": "o", "MidJourney": "X"}
    plot_specs = [
        ("Nature", "#2ca25f", 28, 0.28, 1),
        ("AI", "#de2d26", 36, 0.82, 2),
    ]

    for label, color, size, alpha, zorder in plot_specs:
        label_df = data[data["label"] == label]
        for generator, marker in marker_map.items():
            subset = label_df[label_df[style_col] == generator]
            if subset.empty:
                continue
            ax.scatter(
                subset[xcol],
                subset[ycol],
                s=size,
                c=color,
                marker=marker,
                alpha=alpha,
                edgecolors="black" if label == "AI" else "none",
                linewidths=0.25 if label == "AI" else 0,
                label=f"{label} - {generator}",
                zorder=zorder,
                rasterized=True,
            )

    ax.set_title(title)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.grid(True, alpha=0.25)


def plot_tsne_ai_on_top(data, xcol, ycol, title, filename):
    fig, ax = plt.subplots(figsize=(9, 7))
    layered_label_scatter(ax, data, xcol, ycol, title)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="best", frameon=True)
    savefig(filename)
    plt.show()


def plot_tsne_by_label(data, xcol, ycol, title, filename):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    for ax, label in zip(axes, ["Nature", "AI"]):
        subset = data[data["label"] == label]
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
        ax.set_title(f"{title} - {label}")
        ax.grid(True, alpha=0.25)
    plt.suptitle(f"{title} separated by label", y=1.02)
    savefig(filename)
    plt.show()
'''


def set_source(cell: dict, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)


def append_after_tsne(cell_src: str, data_name: str, xcol: str, ycol: str, title: str, stem: str) -> str:
    if f"{stem}_ai_on_top.png" in cell_src:
        return cell_src
    extra = f'''

plot_tsne_ai_on_top(
    {data_name},
    "{xcol}",
    "{ycol}",
    "{title} (AI points drawn on top)",
    "{stem}_ai_on_top.png",
)

plot_tsne_by_label(
    {data_name},
    "{xcol}",
    "{ycol}",
    "{title}",
    "{stem}_by_label.png",
)
'''
    return cell_src + extra


def patch(path: Path) -> bool:
    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = False

    # Add helper after savefig/PALETTE cell.
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if cell.get("cell_type") == "code" and "PALETTE =" in src and "layered_label_scatter" not in src:
            set_source(cell, src.rstrip() + "\n\n" + HELPER_CODE)
            changed = True
            break

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        new_src = src
        if 'savefig("16_clip_tsne_ai_vs_nature.png")' in src:
            new_src = append_after_tsne(
                src,
                "clip_vis",
                "clip_tsne_1",
                "clip_tsne_2",
                "CLIP t-SNE",
                "16_clip_tsne",
            )
        elif 'savefig("16_fft_clip_tsne_ai_vs_nature.png")' in src:
            new_src = append_after_tsne(
                src,
                "fft_clip_vis",
                "fft_clip_tsne_1",
                "fft_clip_tsne_2",
                "FFT + CLIP t-SNE",
                "16_fft_clip_tsne",
            )
        elif 'savefig("16_all_features_tsne_ai_vs_nature.png")' in src:
            new_src = append_after_tsne(
                src,
                "all_tsne_vis",
                "all_tsne_1",
                "all_tsne_2",
                "Combined Features t-SNE",
                "16_all_features_tsne",
            )
        if new_src != src:
            set_source(cell, new_src)
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
