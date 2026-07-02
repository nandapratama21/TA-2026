import ast
import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in [
        "17_false_detection_feature_overlap_analysis_multiclass.ipynb",
        "17_false_detection_feature_overlap_analysis.ipynb",
    ]:
        path = root / "notebooks" / name
        text = path.read_text(encoding="utf-8")
        nb = json.loads(text)
        print(f"\n### {name}")
        print('uses explicit numeric features:', 'FEATURE_PREFIXES = ("fft_", "clip_feat_")' in text)
        print(
            "old meta logic remains:",
            "feature_cols = [col for col in features.columns if col not in META_COLS]" in text,
        )
        for cell in nb["cells"]:
            if cell.get("cell_type") == "code":
                ast.parse("".join(cell.get("source", [])))
    print("\nOK")


if __name__ == "__main__":
    main()
