"""Merge feature CSVs into one classification-ready feature vector file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build merged feature vector")
    parser.add_argument("--manifest", type=Path, default=Path("data/genimage_manifest.csv"))
    parser.add_argument("--fft", type=Path, default=Path("artifacts/features_fft_mean.csv"))
    parser.add_argument("--clip", type=Path, default=Path("artifacts/features_clip.csv"))
    parser.add_argument("--nriqa", type=Path, default=Path("artifacts/features_nriqa.csv"))
    parser.add_argument("--periodic", type=Path, default=Path("artifacts/features_periodic.csv"))
    parser.add_argument("--use-periodic", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/feature_vector_classification.csv"))
    return parser.parse_args()


def load_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=["image_id"])


def main() -> None:
    args = parse_args()

    manifest = pd.read_csv(args.manifest)
    out = manifest.copy()

    for df in [load_if_exists(args.fft), load_if_exists(args.clip), load_if_exists(args.nriqa)]:
        if len(df.columns) > 1:
            out = out.merge(df, on="image_id", how="inner")

    if args.use_periodic:
        periodic = load_if_exists(args.periodic)
        if len(periodic.columns) > 1:
            out = out.merge(periodic, on="image_id", how="inner")

    out["y_ai"] = out["y_ai"].astype(int)

    # Keep metadata columns for split-aware training.
    meta_cols = ["image_id", "path", "relative_path", "generator", "split", "class_name", "is_real", "y_ai"]
    feat_cols = [c for c in out.columns if c not in meta_cols and c != "nriqa_source" and c != "clip_source"]

    # Fill missing numeric feature values with column median.
    for c in feat_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        med = float(out[c].median()) if out[c].notna().any() else 0.0
        out[c] = out[c].fillna(med)

    out = out[meta_cols + feat_cols]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"[OK] Merged rows: {len(out)}")
    print(f"[OK] Feature dims: {len(feat_cols)}")
    print(f"[OK] Wrote: {args.output.resolve()}")


if __name__ == "__main__":
    main()
