import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/16_feature_space_visualization_from_saved_csv_multiclass.ipynb"


PAIR_CODE = '''

pair_filename_map = {
    ("piqe", "brisque"): "piqe_brisque",
    ("piqe", "niqe"): "piqe_niqe",
    ("brisque", "niqe"): "brisque_niqe",
}

for xcol, ycol in pairs:
    stem = pair_filename_map[(xcol, ycol)]

    fig, ax = plt.subplots(figsize=(8.4, 6.3))
    layered_label_scatter_with_order(
        ax,
        iqa_vis,
        xcol,
        ycol,
        f"{xcol.upper()} vs {ycol.upper()} (AI points drawn on top)",
        top_label="AI",
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="best", frameon=True)
    savefig(f"16_iqa_scatter_{stem}_ai_on_top.png")
    plt.show()

    fig, ax = plt.subplots(figsize=(8.4, 6.3))
    layered_label_scatter_with_order(
        ax,
        iqa_vis,
        xcol,
        ycol,
        f"{xcol.upper()} vs {ycol.upper()} (Nature points drawn on top)",
        top_label="Nature",
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="best", frameon=True)
    savefig(f"16_iqa_scatter_{stem}_nature_on_top.png")
    plt.show()
'''


def set_source(cell: dict, source: str) -> None:
    cell["source"] = source.splitlines(keepends=True)


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    changed = False
    marker = 'savefig("16_iqa_scatter_nature_on_top.png")'

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if marker in source and "pair_filename_map" not in source:
            insertion_point = source.find('\n\nfig, axes = plt.subplots(2, 3')
            if insertion_point == -1:
                source = source.rstrip() + PAIR_CODE + "\n"
            else:
                source = source[:insertion_point] + PAIR_CODE + source[insertion_point:]
            set_source(cell, source)
            changed = True
            break

    if changed:
        NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"patched {NOTEBOOK.relative_to(ROOT)}")
    else:
        print("no changes needed")


if __name__ == "__main__":
    main()
