import json
from pathlib import Path


FFT_SCATTER_CELL = '''fft_cols = ["fft_mag_mean", "fft_phase_mean", "fft_phase_cos_mean", "fft_phase_sin_mean"]
fft_df = all_df[["image_id", "generator", "class_name", "label", "y_ai", *fft_cols]].copy()

display(fft_df[fft_cols].describe())


def layered_fft_scatter(ax, xcol, ycol, title):
    """Gambar Nature lebih dulu, lalu AI di atasnya agar titik AI tidak tertutup."""
    marker_map = {"BigGAN": "o", "MidJourney": "X"}
    plot_specs = [
        ("Nature", "#2ca25f", 28, 0.28, 1),
        ("AI", "#de2d26", 36, 0.82, 2),
    ]

    for label, color, size, alpha, zorder in plot_specs:
        label_df = fft_df[fft_df["label"] == label]
        for generator, marker in marker_map.items():
            subset = label_df[label_df["generator"] == generator]
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


fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
layered_fft_scatter(
    axes[0],
    "fft_mag_mean",
    "fft_phase_mean",
    "FFT Mean Magnitude vs Mean Phase",
)
layered_fft_scatter(
    axes[1],
    "fft_phase_cos_mean",
    "fft_phase_sin_mean",
    "FFT Phase Circular Features",
)
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=True)
plt.suptitle("FFT Scatter Plot with AI Points Drawn on Top", y=1.03)
plt.subplots_adjust(bottom=0.22)
savefig("16_fft_scatter_ai_on_top.png")
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex="col", sharey="col")
pairs = [
    ("fft_mag_mean", "fft_phase_mean", "Mean Magnitude vs Mean Phase"),
    ("fft_phase_cos_mean", "fft_phase_sin_mean", "Mean cos(phase) vs Mean sin(phase)"),
]
labels_order = ["Nature", "AI"]

for row_idx, label in enumerate(labels_order):
    subset = fft_df[fft_df["label"] == label]
    for col_idx, (xcol, ycol, title) in enumerate(pairs):
        ax = axes[row_idx, col_idx]
        sns.scatterplot(
            data=subset,
            x=xcol,
            y=ycol,
            hue="generator",
            style="generator",
            alpha=0.65,
            s=30,
            ax=ax,
        )
        ax.set_title(f"{label}: {title}")
        ax.grid(True, alpha=0.25)

plt.suptitle("FFT Scatter Plot Separated by Label", y=1.02)
savefig("16_fft_scatter_by_label.png")
plt.show()
'''


FFT_DISTRIBUTION_CELL = '''# Distribusi tiap fitur FFT agar pemisahan marginal terlihat tanpa saling menutupi.
fft_long = fft_df.melt(
    id_vars=["image_id", "generator", "class_name", "label", "y_ai"],
    value_vars=fft_cols,
    var_name="feature",
    value_name="value",
)

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes = axes.ravel()

for ax, col in zip(axes, fft_cols):
    sns.ecdfplot(
        data=fft_df,
        x=col,
        hue="label",
        palette=PALETTE,
        linewidth=2,
        ax=ax,
    )
    ax.set_title(f"ECDF {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Cumulative proportion")
    ax.grid(True, alpha=0.25)

plt.suptitle("FFT Feature ECDF by Label", y=1.02)
savefig("16_fft_feature_ecdf_by_label.png")
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes = axes.ravel()

for ax, col in zip(axes, fft_cols):
    sns.histplot(
        data=fft_df,
        x=col,
        hue="label",
        palette=PALETTE,
        bins=40,
        element="step",
        fill=False,
        stat="density",
        common_norm=False,
        linewidth=1.8,
        ax=ax,
    )
    ax.set_title(f"Density Histogram {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)

plt.suptitle("FFT Feature Density Histogram by Label", y=1.02)
savefig("16_fft_feature_density_by_label.png")
plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(
    data=fft_long,
    x="feature",
    y="value",
    hue="label",
    palette=PALETTE,
    showfliers=False,
)
plt.title("FFT Feature Value Distribution by Label")
plt.xlabel("FFT feature")
plt.ylabel("Feature value")
plt.xticks(rotation=15)
savefig("16_fft_feature_boxplot_by_label.png")
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
        if (
            'fft_df = all_df[["image_id", "generator", "class_name", "label", "y_ai", *fft_cols]].copy()'
            in src
            and "FFT Mean Magnitude vs Mean Phase" in src
        ):
            set_source(cell, FFT_SCATTER_CELL)
            changed = True
        elif "Histogram distribusi tiap fitur FFT" in src or "FFT Feature Distributions" in src:
            set_source(cell, FFT_DISTRIBUTION_CELL)
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
