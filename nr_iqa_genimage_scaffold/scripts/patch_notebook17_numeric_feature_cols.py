import json
from pathlib import Path


OLD = '''META_COLS = {
    "image_id", "path", "relative_path", "generator", "subset_name", "split",
    "class_name", "content_id", "content_label", "is_real", "y_ai",
}
feature_cols = [col for col in features.columns if col not in META_COLS]
'''


NEW = '''FEATURE_PREFIXES = ("fft_", "clip_feat_")
FEATURE_NAMES = {"piqe", "brisque", "niqe"}
feature_cols = [
    col for col in features.columns
    if col.startswith(FEATURE_PREFIXES) or col in FEATURE_NAMES
]

if not feature_cols:
    raise ValueError("Tidak ada kolom fitur numerik yang ditemukan pada feature vector.")

non_numeric_cols = features[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric_cols:
    raise ValueError(f"Kolom fitur non-numerik terdeteksi: {non_numeric_cols}")

print("Jumlah fitur untuk t-SNE:", len(feature_cols))
'''


def patch(path: Path) -> bool:
    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if OLD in src:
            cell["source"] = src.replace(OLD, NEW).splitlines(keepends=True)
            changed = True
    if changed:
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return changed


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in [
        "17_false_detection_feature_overlap_analysis_multiclass.ipynb",
        "17_false_detection_feature_overlap_analysis.ipynb",
    ]:
        path = root / "notebooks" / name
        if path.exists() and patch(path):
            print("patched", path.relative_to(root))


if __name__ == "__main__":
    main()
