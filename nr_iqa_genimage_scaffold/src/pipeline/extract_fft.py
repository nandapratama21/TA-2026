"""Dummy entrypoint ekstraksi fitur FFT dengan perhitungan nyata sederhana."""

from __future__ import annotations

import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.pipeline_modular.fft_features import extract_fft_features_from_id
from src.pipeline_modular.io_utils import load_manifest


def main() -> None:
    root = ROOT
    manifest_path = os.path.join(root, "data", "genimage_manifest.csv")
    out_path = os.path.join(root, "artifacts", "features_fft_dummy.csv")

    manifest = load_manifest(manifest_path)
    rows = []
    for image_id in manifest["image_id"].tolist():
        rows.append({"image_id": image_id, **extract_fft_features_from_id(image_id), "fft_cfg": "fft_v2_dummy"})

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[DUMMY] Wrote FFT features: {out_path}")


if __name__ == "__main__":
    main()
