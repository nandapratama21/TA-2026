import ast
import json
from pathlib import Path


NOTEBOOKS = [
    "09_biggan_midjourney_multiclass_fft_clip_xgboost_mlp.ipynb",
    "10_biggan_midjourney_multiclass_clip_xgboost_mlp.ipynb",
    "12_biggan_midjourney_multiclass_iqa_fft_xgboost_mlp.ipynb",
    "13_biggan_midjourney_multiclass_iqa_clip_xgboost_mlp.ipynb",
    "14_biggan_midjourney_multiclass_iqa_fft_clip_xgboost_mlp.ipynb",
    "18_single_image_inference_fft_clip_multiclass.ipynb",
]


def main() -> None:
    nb_dir = Path(__file__).resolve().parents[1] / "notebooks"
    for name in NOTEBOOKS:
        path = nb_dir / name
        nb = json.loads(path.read_text(encoding="utf-8"))
        print(f"\n### {name}")
        for i, cell in enumerate(nb["cells"]):
            src = "".join(cell.get("source", []))
            if cell.get("cell_type") == "code":
                ast.parse(src)
            if "feature_cols = [" in src:
                lines = src.splitlines()
                for j, line in enumerate(lines):
                    if "feature_cols = [" in line:
                        print("cell", i)
                        print("\n".join(lines[j : j + 5]))
                        break
        text = path.read_text(encoding="utf-8")
        assert "feature_cols = [c for c in data.columns if c not in meta_cols]" not in text
        assert "archive_path" not in text
    print("\nOK")


if __name__ == "__main__":
    main()
