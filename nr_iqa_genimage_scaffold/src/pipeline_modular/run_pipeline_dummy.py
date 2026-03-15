"""Run full dummy modular pipeline and save outputs."""

from __future__ import annotations

import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.pipeline_modular.clip_features import extract_clip_features
from src.pipeline_modular.eval import evaluate_with_labels
from src.pipeline_modular.fft_features import extract_fft_features_from_id
from src.pipeline_modular.fusion import fuse_features, predict_dummy_score
from src.pipeline_modular.io_utils import load_manifest, load_quality_labels


def main() -> None:
    root = ROOT
    manifest_path = os.path.join(root, "data", "genimage_manifest.csv")
    labels_path = os.path.join(root, "data", "labels_quality.csv")
    artifacts_dir = os.path.join(root, "artifacts")

    manifest = load_manifest(manifest_path)
    labels = load_quality_labels(labels_path)

    fft_rows = []
    clip_rows = []
    for image_id in manifest["image_id"].tolist():
        fft_row = {"image_id": image_id, **extract_fft_features_from_id(image_id)}
        clip_row = {"image_id": image_id, **extract_clip_features(image_id)}
        fft_rows.append(fft_row)
        clip_rows.append(clip_row)

    fft_df = pd.DataFrame(fft_rows)
    clip_df = pd.DataFrame(clip_rows)
    fused = fuse_features(fft_df, clip_df)
    pred = predict_dummy_score(fused)
    metrics = evaluate_with_labels(pred, labels)

    fft_df.to_csv(os.path.join(artifacts_dir, "features_fft_modular_dummy.csv"), index=False)
    clip_df.to_csv(os.path.join(artifacts_dir, "features_clip_modular_dummy.csv"), index=False)
    pred.to_csv(os.path.join(artifacts_dir, "predictions_modular_dummy.csv"), index=False)
    metrics.to_csv(os.path.join(artifacts_dir, "metrics_modular_dummy.csv"), index=False)

    print("[DUMMY] Modular pipeline complete. Outputs written to artifacts/.")


if __name__ == "__main__":
    main()
