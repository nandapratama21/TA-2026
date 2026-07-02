import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/16_feature_space_visualization_from_saved_csv_multiclass.ipynb"


NATURE_FUNCTION = '''

def plot_tsne_nature_on_top(data, xcol, ycol, title, filename):
    fig, ax = plt.subplots(figsize=(9, 7))
    layered_label_scatter_with_order(
        ax,
        data,
        xcol,
        ycol,
        title,
        top_label="Nature",
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="best", frameon=True)
    savefig(filename)
    plt.show()
'''


INSERTIONS = [
    (
        'plot_tsne_ai_on_top(\n'
        '    clip_vis,\n'
        '    "clip_tsne_1",\n'
        '    "clip_tsne_2",\n'
        '    "CLIP t-SNE (AI points drawn on top)",\n'
        '    "16_clip_tsne_ai_on_top.png",\n'
        ')\n',
        '\nplot_tsne_nature_on_top(\n'
        '    clip_vis,\n'
        '    "clip_tsne_1",\n'
        '    "clip_tsne_2",\n'
        '    "CLIP t-SNE (Nature points drawn on top)",\n'
        '    "16_clip_tsne_nature_on_top.png",\n'
        ')\n',
    ),
    (
        'plot_tsne_ai_on_top(\n'
        '    fft_clip_vis,\n'
        '    "fft_clip_tsne_1",\n'
        '    "fft_clip_tsne_2",\n'
        '    "FFT + CLIP t-SNE (AI points drawn on top)",\n'
        '    "16_fft_clip_tsne_ai_on_top.png",\n'
        ')\n',
        '\nplot_tsne_nature_on_top(\n'
        '    fft_clip_vis,\n'
        '    "fft_clip_tsne_1",\n'
        '    "fft_clip_tsne_2",\n'
        '    "FFT + CLIP t-SNE (Nature points drawn on top)",\n'
        '    "16_fft_clip_tsne_nature_on_top.png",\n'
        ')\n',
    ),
    (
        'plot_tsne_ai_on_top(\n'
        '    all_tsne_vis,\n'
        '    "all_tsne_1",\n'
        '    "all_tsne_2",\n'
        '    "Combined Features t-SNE (AI points drawn on top)",\n'
        '    "16_all_features_tsne_ai_on_top.png",\n'
        ')\n',
        '\nplot_tsne_nature_on_top(\n'
        '    all_tsne_vis,\n'
        '    "all_tsne_1",\n'
        '    "all_tsne_2",\n'
        '    "Combined Features t-SNE (Nature points drawn on top)",\n'
        '    "16_all_features_tsne_nature_on_top.png",\n'
        ')\n',
    ),
]


def set_source(cell: dict, source: str) -> None:
    cell["source"] = source.splitlines(keepends=True)


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    changed = False

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "def layered_label_scatter_with_order" in source and "def plot_tsne_nature_on_top" not in source:
            set_source(cell, source.rstrip() + NATURE_FUNCTION + "\n")
            changed = True
            break

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        new_source = source
        for anchor, addition in INSERTIONS:
            if anchor in new_source and addition.strip() not in new_source:
                new_source = new_source.replace(anchor, anchor + addition)
        if new_source != source:
            set_source(cell, new_source)
            changed = True

    if changed:
        NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"patched {NOTEBOOK.relative_to(ROOT)}")
    else:
        print("no changes needed")


if __name__ == "__main__":
    main()
