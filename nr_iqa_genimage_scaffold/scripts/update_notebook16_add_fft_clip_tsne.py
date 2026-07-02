import json
from pathlib import Path


NOTEBOOK = Path("notebooks/16_feature_space_visualization_from_saved_csv.ipynb")


def lines(text: str):
    return text.splitlines(keepends=True)


def main():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    src = "".join(nb["cells"][0]["source"])
    src = src.replace(
        "- t-SNE 2 dimensi untuk fitur CLIP only.\n"
        "- t-SNE 2 dimensi untuk seluruh fitur gabungan (`CLIP + FFT + IQA`).",
        "- t-SNE 2 dimensi untuk fitur CLIP only.\n"
        "- t-SNE 2 dimensi untuk fitur FFT + CLIP.\n"
        "- t-SNE 2 dimensi untuk seluruh fitur gabungan (`CLIP + FFT + IQA`).",
    )
    nb["cells"][0]["source"] = lines(src)

    src = "".join(nb["cells"][1]["source"])
    if "CSV_FFT_CLIP" not in src:
        src = src.replace(
            'CSV_CLIP = ARTIFACT_DIR / "feature_vector_biggan_midjourney_cat4_clip_only.csv"\n'
            'CSV_IQA = ARTIFACT_DIR / "feature_vector_biggan_midjourney_cat4_iqa_only.csv"',
            'CSV_CLIP = ARTIFACT_DIR / "feature_vector_biggan_midjourney_cat4_clip_only.csv"\n'
            'CSV_FFT_CLIP = ARTIFACT_DIR / "feature_vector_biggan_midjourney_cat4_fft_clip.csv"\n'
            'CSV_IQA = ARTIFACT_DIR / "feature_vector_biggan_midjourney_cat4_iqa_only.csv"',
        )
    nb["cells"][1]["source"] = lines(src)

    src = "".join(nb["cells"][2]["source"])
    if "fft_clip_df = pd.read_csv(CSV_FFT_CLIP)" not in src:
        src = src.replace(
            "clip_df = pd.read_csv(CSV_CLIP)\n"
            "iqa_df = pd.read_csv(CSV_IQA)",
            "clip_df = pd.read_csv(CSV_CLIP)\n"
            "fft_clip_df = pd.read_csv(CSV_FFT_CLIP)\n"
            "iqa_df = pd.read_csv(CSV_IQA)",
        )
        src = src.replace(
            "for df in [all_df, clip_df, iqa_df]:",
            "for df in [all_df, clip_df, fft_clip_df, iqa_df]:",
        )
        src = src.replace(
            'print("IQA vector:", iqa_df.shape)',
            'print("FFT + CLIP vector:", fft_clip_df.shape)\n'
            'print("IQA vector:", iqa_df.shape)',
        )
    nb["cells"][2]["source"] = lines(src)

    exists = any(
        "t-SNE Fitur FFT + CLIP" in "".join(cell.get("source", []))
        for cell in nb["cells"]
    )
    if not exists:
        markdown_cell = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## t-SNE Fitur FFT + CLIP\n",
                "\n",
                "Bagian ini memakai fitur gabungan FFT dan CLIP dari CSV `feature_vector_biggan_midjourney_cat4_fft_clip.csv`. Visualisasi ini digunakan untuk melihat ruang fitur dari konfigurasi yang memperoleh performa terbaik pada eksperimen klasifikasi.\n",
                "\n",
                "Seperti visualisasi t-SNE lainnya, grafik ini hanya digunakan untuk interpretasi pola kedekatan relatif antar sampel dan tidak digunakan sebagai fitur pelatihan model.\n",
            ],
        }

        plot_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                'fft_cols = ["fft_mag_mean", "fft_phase_mean", "fft_phase_cos_mean", "fft_phase_sin_mean"]\n',
                'clip_cols_fft_clip = [c for c in fft_clip_df.columns if c.startswith("clip_feat_")]\n',
                "fft_clip_feature_cols = fft_cols + clip_cols_fft_clip\n",
                "\n",
                "X_fft_clip = fft_clip_df[fft_clip_feature_cols].to_numpy(dtype=np.float32)\n",
                "fft_clip_xy, fft_clip_perplexity = compute_tsne_2d(X_fft_clip)\n",
                "\n",
                'fft_clip_vis = fft_clip_df[["image_id", "generator", "class_name", "label", "y_ai"]].copy()\n',
                'fft_clip_vis["fft_clip_tsne_1"] = fft_clip_xy[:, 0]\n',
                'fft_clip_vis["fft_clip_tsne_2"] = fft_clip_xy[:, 1]\n',
                "\n",
                'print("FFT + CLIP feature columns:", len(fft_clip_feature_cols))\n',
                'print("- FFT columns:", len(fft_cols))\n',
                'print("- CLIP columns:", len(clip_cols_fft_clip))\n',
                'print("t-SNE perplexity:", fft_clip_perplexity)\n',
                "display(fft_clip_vis.head())\n",
                "\n",
                "plt.figure(figsize=(8, 6))\n",
                "sns.scatterplot(\n",
                "    data=fft_clip_vis,\n",
                '    x="fft_clip_tsne_1",\n',
                '    y="fft_clip_tsne_2",\n',
                '    hue="label",\n',
                '    style="generator",\n',
                "    palette=PALETTE,\n",
                "    alpha=0.75,\n",
                "    s=45,\n",
                ")\n",
                'plt.title("t-SNE 2D of FFT + CLIP Features")\n',
                'plt.xlabel("t-SNE 1")\n',
                'plt.ylabel("t-SNE 2")\n',
                'savefig("16_fft_clip_tsne_ai_vs_nature.png")\n',
                "plt.show()\n",
            ],
        }

        by_generator_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)\n",
                'for ax, generator in zip(axes, sorted(fft_clip_vis["generator"].unique())):\n',
                '    subset = fft_clip_vis[fft_clip_vis["generator"] == generator]\n',
                "    sns.scatterplot(\n",
                "        data=subset,\n",
                '        x="fft_clip_tsne_1",\n',
                '        y="fft_clip_tsne_2",\n',
                '        hue="label",\n',
                "        palette=PALETTE,\n",
                "        alpha=0.75,\n",
                "        s=45,\n",
                "        ax=ax,\n",
                "    )\n",
                '    ax.set_title(f"FFT + CLIP t-SNE - {generator}")\n',
                '    ax.set_xlabel("t-SNE 1")\n',
                '    ax.set_ylabel("t-SNE 2")\n',
                'plt.suptitle("FFT + CLIP Feature Space by Generator", y=1.02)\n',
                'savefig("16_fft_clip_tsne_by_generator.png")\n',
                "plt.show()\n",
            ],
        }

        nb["cells"][7:7] = [markdown_cell, plot_cell, by_generator_cell]

    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Updated {NOTEBOOK}")


if __name__ == "__main__":
    main()
