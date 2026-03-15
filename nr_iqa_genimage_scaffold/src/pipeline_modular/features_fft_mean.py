"""Extract FFT mean magnitude/phase features per image.

Core feature (as requested):
- mean magnitude
- mean phase

Additional stable phase summary:
- mean(cos phase)
- mean(sin phase)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract FFT mean features")
    parser.add_argument("--manifest", type=Path, default=Path("data/genimage_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/features_fft_mean.csv"))
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all")
    return parser.parse_args()


def load_gray(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32)
    return arr


def extract_fft_mean_features(gray: np.ndarray) -> dict[str, float]:
    f = np.fft.fft2(gray)
    mag = np.abs(f)
    phs = np.angle(f)

    return {
        "fft_mag_mean": float(np.mean(mag)),
        "fft_phase_mean": float(np.mean(phs)),
        "fft_phase_cos_mean": float(np.mean(np.cos(phs))),
        "fft_phase_sin_mean": float(np.mean(np.sin(phs))),
    }


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.manifest)
    if args.max_samples > 0:
        df = df.head(args.max_samples)

    rows = []
    failed = 0
    for row in df.itertuples(index=False):
        try:
            gray = load_gray(row.path)
            feats = extract_fft_mean_features(gray)
            rows.append({"image_id": row.image_id, **feats})
        except Exception:
            failed += 1

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"[OK] FFT features: {len(out)}")
    print(f"[WARN] Failed images: {failed}")
    print(f"[OK] Wrote: {args.output.resolve()}")


if __name__ == "__main__":
    main()
