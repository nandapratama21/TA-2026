import json
import re
from copy import deepcopy
from pathlib import Path


NOTEBOOKS = [
    "08_biggan_midjourney_cat4_fft_xgboost_mlp.ipynb",
    "09_biggan_midjourney_cat4_fft_clip_xgboost_mlp.ipynb",
    "10_biggan_midjourney_cat4_clip_xgboost_mlp.ipynb",
    "11_biggan_midjourney_cat4_iqa_xgboost_mlp.ipynb",
    "12_biggan_midjourney_cat4_iqa_fft_xgboost_mlp.ipynb",
    "13_biggan_midjourney_cat4_iqa_clip_xgboost_mlp.ipynb",
    "14_biggan_midjourney_cat4_iqa_fft_clip_xgboost_mlp.ipynb",
    "15_results_visualization_and_qualitative_analysis.ipynb",
    "16_feature_space_visualization_from_saved_csv.ipynb",
    "17_false_detection_feature_overlap_analysis.ipynb",
    "18_single_image_inference_fft_clip.ipynb",
]


CONFIG_RE = re.compile(
    r'DATA_ROOTS = \{.*?\}\nIMAGENET_MAP = PROJECT_ROOT / "data" / "imagenet_class_index\.json"',
    re.S,
)
COMMON_RE = re.compile(
    r'\nCOMMON_CLASSES = \{.*?print\("Cat classes:", COMMON_CLASSES\)\n?',
    re.S,
)


MULTICLASS_CONFIG = '''MULTICLASS_ROOT = PROJECT_ROOT / "data" / "raw" / "genimage_multiclass_balanced"
DATA_ROOT = MULTICLASS_ROOT / "genimage"
DATA_ROOTS = {
    "BigGAN": DATA_ROOT / "BigGAN" / "train",
    "MidJourney": DATA_ROOT / "MidJourney" / "train",
}
MANIFEST_SOURCE = MULTICLASS_ROOT / "genimage_multiclass_manifest.csv"
CLASS_TABLE_SOURCE = MULTICLASS_ROOT / "genimage_multiclass_classes.csv"
IMAGENET_MAP = PROJECT_ROOT / "data" / "imagenet_class_index.json"'''


AUDIT_CELL = '''manifest_source = pd.read_csv(MANIFEST_SOURCE)
class_table = pd.read_csv(CLASS_TABLE_SOURCE)

manifest_source["path_exists"] = manifest_source["path"].map(lambda p: Path(p).is_file())
print("Manifest source:", MANIFEST_SOURCE.resolve())
print("Class table:", CLASS_TABLE_SOURCE.resolve())
print("Manifest shape:", manifest_source.shape)
print("Jumlah kelas:", manifest_source["imagenet_id"].nunique())
print("Semua path ada:", bool(manifest_source["path_exists"].all()))

display(class_table.head(10))
display(
    manifest_source
    .groupby(["generator", "class_name"])
    .size()
    .rename("count")
    .reset_index()
)
display(
    manifest_source
    .groupby(["generator", "class_name", "imagenet_id", "wordnet_id", "content_label"])
    .size()
    .rename("count")
    .reset_index()
    .describe(include="all")
)
'''


MANIFEST_CELL = '''manifest = pd.read_csv(MANIFEST_SOURCE).copy()
manifest["subset_name"] = "biggan_midjourney_multiclass_train"
manifest["split"] = "train"
manifest["path"] = manifest["path"].map(lambda p: str(Path(p).resolve()))

missing_paths = manifest.loc[~manifest["path"].map(lambda p: Path(p).is_file()), "path"].head()
if len(missing_paths):
    raise FileNotFoundError(
        "Ada path manifest yang tidak ditemukan, contoh: "
        + str(missing_paths.tolist())
    )

manifest.to_csv(MANIFEST_OUT, index=False)
print("Saved manifest:", MANIFEST_OUT.resolve())
print("Manifest shape:", manifest.shape)
display(manifest.head())
display(manifest.groupby(["generator", "y_ai"]).size().rename("count").reset_index())
display(
    manifest
    .groupby(["generator", "class_name", "content_label"])
    .size()
    .rename("count")
    .reset_index()
    .head(20)
)
'''


def multiclass_name(old_name: str) -> str:
    if old_name.startswith(("08_", "09_", "10_", "11_", "12_", "13_", "14_")):
        return old_name.replace("_cat4_", "_multiclass_")
    return old_name.replace(".ipynb", "_multiclass.ipynb")


def replace_common_text(src: str) -> str:
    src = src.replace("biggan_midjourney_cat4_train", "biggan_midjourney_multiclass_train")
    src = src.replace("biggan_midjourney_cat4_", "biggan_midjourney_multiclass_")
    src = src.replace("_iqa_safe_images_cat4", "_iqa_safe_images_multiclass")
    src = src.replace(
        "data/raw/genimage/BigGAN/train/ai/281_biggan_00000.png",
        "data/raw/genimage_multiclass_balanced/genimage/BigGAN/train/ai/000_biggan_00037.png",
    )
    src = src.replace("cat4", "multiclass")
    src = src.replace("Cat4", "Multiclass")
    src = src.replace("CAT4", "MULTICLASS")
    src = src.replace(
        "feature_cols = [c for c in data.columns if c not in meta_cols]",
        'feature_cols = [\n'
        '    c for c in data.columns\n'
        '    if c.startswith(("fft_", "clip_feat_")) or c in ["piqe", "brisque", "niqe"]\n'
        ']',
    )
    return src


def clear_outputs(cell: dict) -> None:
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None


def as_text(cell: dict) -> str:
    source = cell.get("source", [])
    return "".join(source) if isinstance(source, list) else source


def set_text(cell: dict, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)


def patch_fft_only_vector_output(nb: dict) -> None:
    for cell in nb["cells"]:
        text = as_text(cell)
        if (
            cell.get("cell_type") == "code"
            and "FFT_OUT = PROJECT_ROOT" in text
            and "results_classification_biggan_midjourney_multiclass_fft_only.csv" in text
            and "VECTOR_OUT" not in text
        ):
            text = text.replace(
                'FFT_OUT = PROJECT_ROOT / "artifacts" / "features_fft_mean_biggan_midjourney_multiclass_train.csv"\n',
                'FFT_OUT = PROJECT_ROOT / "artifacts" / "features_fft_mean_biggan_midjourney_multiclass_train.csv"\n'
                'VECTOR_OUT = PROJECT_ROOT / "artifacts" / "feature_vector_biggan_midjourney_multiclass_fft_only.csv"\n',
            )
            set_text(cell, text)
        elif (
            cell.get("cell_type") == "code"
            and 'feature_cols = ["fft_mag_mean", "fft_phase_mean", "fft_phase_cos_mean", "fft_phase_sin_mean"]' in text
            and "data.to_csv(VECTOR_OUT" not in text
        ):
            text = text.replace(
                'feature_cols = ["fft_mag_mean", "fft_phase_mean", "fft_phase_cos_mean", "fft_phase_sin_mean"]\n'
                'print(data.shape)\n',
                'feature_cols = ["fft_mag_mean", "fft_phase_mean", "fft_phase_cos_mean", "fft_phase_sin_mean"]\n'
                'data.to_csv(VECTOR_OUT, index=False)\n'
                'print("Saved feature vector:", VECTOR_OUT.resolve())\n'
                'print(data.shape)\n',
            )
            set_text(cell, text)


def patch_clip_constants(nb: dict) -> None:
    for cell in nb["cells"]:
        text = as_text(cell)
        if (
            cell.get("cell_type") == "code"
            and "CLIP_OUT = PROJECT_ROOT" in text
            and "CLIP_MODEL_NAME =" not in text
        ):
            marker = 'PRED_OUT = PROJECT_ROOT / "artifacts" / '
            insert_after = None
            lines = text.splitlines(keepends=True)
            for idx, line in enumerate(lines):
                if marker in line:
                    insert_after = idx + 1
                    break
            if insert_after is None:
                continue
            lines[insert_after:insert_after] = [
                'CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"\n',
                "CLIP_OUT_DIM = 64\n",
            ]
            set_text(cell, "".join(lines))


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    nb_dir = project_root / "notebooks"
    created = []

    for old_name in NOTEBOOKS:
        src_path = nb_dir / old_name
        nb = json.loads(src_path.read_text(encoding="utf-8"))
        nb = deepcopy(nb)

        for cell in nb.get("cells", []):
            text = replace_common_text(as_text(cell))
            clear_outputs(cell)
            set_text(cell, text)

        if old_name.startswith(("08_", "09_", "10_", "11_", "12_", "13_", "14_")):
            for cell in nb["cells"]:
                text = as_text(cell)
                if cell.get("cell_type") == "code" and "DATA_ROOTS = {" in text and "IMAGENET_MAP" in text:
                    text = CONFIG_RE.sub(MULTICLASS_CONFIG, text)
                    text = COMMON_RE.sub(
                        '\nprint("Data root:", DATA_ROOT.resolve())\n'
                        'print("Manifest source:", MANIFEST_SOURCE.resolve())\n',
                        text,
                    )
                    set_text(cell, text)
                elif cell.get("cell_type") == "code" and "audit_rows = []" in text:
                    set_text(cell, AUDIT_CELL)
                elif (
                    cell.get("cell_type") == "code"
                    and text.lstrip().startswith("rows = []")
                    and "manifest = pd.DataFrame(rows)" in text
                ):
                    set_text(cell, MANIFEST_CELL)

        if old_name.startswith("08_"):
            patch_fft_only_vector_output(nb)
        if "_clip_" in old_name or old_name.startswith(("09_", "10_")):
            patch_clip_constants(nb)

        out_path = nb_dir / multiclass_name(old_name)
        out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        created.append(out_path.relative_to(project_root))

    print("Created notebooks:")
    for path in created:
        print(path)

    from patch_multiclass_notebooks_reuse_saved_features import patch_notebook

    print("Applying saved-feature reuse patch:")
    for path in created:
        full_path = project_root / path
        if full_path.name.startswith(tuple(f"{i:02d}_" for i in range(8, 15))):
            if patch_notebook(full_path):
                print("patched", path)


if __name__ == "__main__":
    main()
