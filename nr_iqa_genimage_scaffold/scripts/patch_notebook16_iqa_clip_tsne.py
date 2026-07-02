import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/16_feature_space_visualization_from_saved_csv_multiclass.ipynb"


IQA_CLIP_CELL = '''iqa_cols_iqa_clip = ["piqe", "brisque", "niqe"]
clip_cols_iqa_clip = [c for c in iqa_clip_df.columns if c.startswith("clip_feat_")]
iqa_clip_feature_cols = iqa_cols_iqa_clip + clip_cols_iqa_clip

X_iqa_clip = iqa_clip_df[iqa_clip_feature_cols].to_numpy(dtype=np.float32)
iqa_clip_xy, iqa_clip_perplexity = compute_tsne_2d(X_iqa_clip)

iqa_clip_vis = iqa_clip_df[["image_id", "generator", "class_name", "label", "y_ai"]].copy()
iqa_clip_vis["iqa_clip_tsne_1"] = iqa_clip_xy[:, 0]
iqa_clip_vis["iqa_clip_tsne_2"] = iqa_clip_xy[:, 1]

print("IQA + CLIP feature columns:", len(iqa_clip_feature_cols))
print("- IQA columns:", len(iqa_cols_iqa_clip))
print("- CLIP columns:", len(clip_cols_iqa_clip))
print("t-SNE perplexity:", iqa_clip_perplexity)
display(iqa_clip_vis.head())

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=iqa_clip_vis,
    x="iqa_clip_tsne_1",
    y="iqa_clip_tsne_2",
    hue="label",
    style="generator",
    palette=PALETTE,
    alpha=0.75,
    s=45,
)
plt.title("t-SNE 2D of IQA + CLIP Features")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
savefig("16_iqa_clip_tsne_ai_vs_nature.png")
plt.show()

plot_tsne_ai_on_top(
    iqa_clip_vis,
    "iqa_clip_tsne_1",
    "iqa_clip_tsne_2",
    "IQA + CLIP t-SNE (AI points drawn on top)",
    "16_iqa_clip_tsne_ai_on_top.png",
)

plot_tsne_nature_on_top(
    iqa_clip_vis,
    "iqa_clip_tsne_1",
    "iqa_clip_tsne_2",
    "IQA + CLIP t-SNE (Nature points drawn on top)",
    "16_iqa_clip_tsne_nature_on_top.png",
)

plot_tsne_by_label(
    iqa_clip_vis,
    "iqa_clip_tsne_1",
    "iqa_clip_tsne_2",
    "IQA + CLIP t-SNE",
    "16_iqa_clip_tsne_by_label.png",
)
'''


def set_source(cell: dict, source: str) -> None:
    cell["source"] = source.splitlines(keepends=True)


def new_code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    changed = False

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "CSV_IQA =" in source and "CSV_IQA_CLIP" not in source:
            source = source.replace(
                'CSV_IQA = ARTIFACT_DIR / "feature_vector_biggan_midjourney_multiclass_iqa_only.csv"\n',
                'CSV_IQA = ARTIFACT_DIR / "feature_vector_biggan_midjourney_multiclass_iqa_only.csv"\n'
                'CSV_IQA_CLIP = ARTIFACT_DIR / "feature_vector_biggan_midjourney_multiclass_iqa_clip.csv"\n',
            )
            set_source(cell, source)
            changed = True
            break

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "iqa_df = pd.read_csv(CSV_IQA)" in source and "iqa_clip_df" not in source:
            source = source.replace(
                "iqa_df = pd.read_csv(CSV_IQA)\n",
                "iqa_df = pd.read_csv(CSV_IQA)\n"
                "iqa_clip_df = pd.read_csv(CSV_IQA_CLIP)\n",
            )
            source = source.replace(
                "for df in [all_df, clip_df, fft_clip_df, iqa_df]:\n",
                "for df in [all_df, clip_df, fft_clip_df, iqa_df, iqa_clip_df]:\n",
            )
            source = source.replace(
                'print("IQA vector:", iqa_df.shape)\n',
                'print("IQA vector:", iqa_df.shape)\n'
                'print("IQA + CLIP vector:", iqa_clip_df.shape)\n',
            )
            set_source(cell, source)
            changed = True
            break

    has_iqa_clip_cell = any(
        cell.get("cell_type") == "code" and "16_iqa_clip_tsne_ai_on_top.png" in "".join(cell.get("source", []))
        for cell in nb["cells"]
    )
    if not has_iqa_clip_cell:
        insert_at = None
        for idx, cell in enumerate(nb["cells"]):
            source = "".join(cell.get("source", []))
            if cell.get("cell_type") == "code" and 'savefig("16_iqa_scatter_by_label.png")' in source:
                insert_at = idx + 1
                break
        if insert_at is None:
            raise RuntimeError("Tidak menemukan cell IQA scatter sebagai titik sisip.")
        nb["cells"].insert(insert_at, new_code_cell(IQA_CLIP_CELL))
        changed = True

    if changed:
        NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"patched {NOTEBOOK.relative_to(ROOT)}")
    else:
        print("no changes needed")


if __name__ == "__main__":
    main()
