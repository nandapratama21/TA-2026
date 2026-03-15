"""Dummy entrypoint ekstraksi embedding CLIP visual."""

from __future__ import annotations

import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.pipeline_modular.clip_features import extract_clip_features
from src.pipeline_modular.io_utils import load_manifest


def main() -> None:
    root = ROOT
    manifest_path = os.path.join(root, "data", "genimage_manifest.csv")
    out_path = os.path.join(root, "artifacts", "features_clip_dummy.csv")

    manifest = load_manifest(manifest_path)
    rows = []
    for image_id in manifest["image_id"].tolist():
        rows.append({"image_id": image_id, **extract_clip_features(image_id), "clip_model": "ViT-B/32_dummy"})

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[DUMMY] Wrote CLIP features: {out_path}")


if __name__ == "__main__":
    main()
