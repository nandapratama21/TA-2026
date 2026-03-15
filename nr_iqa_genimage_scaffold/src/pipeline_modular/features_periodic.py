"""Extract periodic artifact indicators from frequency domain.

Reference direction:
- Checkerboard artifact literature for CNN-generated images.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract periodic artifact features")
    parser.add_argument("--manifest", type=Path, default=Path("data/genimage_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/features_periodic.csv"))
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def load_gray(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32)


def periodic_features(gray: np.ndarray) -> dict[str, float]:
    f = np.fft.fft2(gray)
    s = np.fft.fftshift(f)
    mag = np.log1p(np.abs(s))

    h, w = mag.shape
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    r_norm = r / (r.max() + 1e-12)

    # Radial mean baseline.
    bins = np.minimum((r_norm * 100).astype(np.int32), 99)
    radial_means = np.zeros(100, dtype=np.float64)
    for b in range(100):
        m = mag[bins == b]
        radial_means[b] = float(m.mean()) if m.size else 0.0

    baseline = radial_means[bins]
    residual = mag - baseline

    mask = (r_norm >= 0.2) & (r_norm <= 0.85)
    vals = residual[mask]
    if vals.size == 0:
        return {
            "periodic_peak_ratio": 0.0,
            "periodic_topk_energy_ratio": 0.0,
            "periodic_score": 0.0,
        }

    vals_sorted = np.sort(vals)
    peak = float(np.percentile(vals_sorted, 99.5))
    med = float(np.median(vals_sorted))
    peak_ratio = peak / (abs(med) + 1e-6)

    topk = vals_sorted[-max(10, vals_sorted.size // 200) :]
    topk_energy = float(np.sum(np.square(topk)))
    total_energy = float(np.sum(np.square(vals_sorted))) + 1e-12
    topk_energy_ratio = topk_energy / total_energy

    periodic_score = float(np.tanh(0.2 * peak_ratio) * 0.5 + np.tanh(5.0 * topk_energy_ratio) * 0.5)

    return {
        "periodic_peak_ratio": float(peak_ratio),
        "periodic_topk_energy_ratio": float(topk_energy_ratio),
        "periodic_score": periodic_score,
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
            feats = periodic_features(gray)
            rows.append({"image_id": row.image_id, **feats})
        except Exception:
            failed += 1

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"[OK] Periodic features: {len(out)}")
    print(f"[WARN] Failed images: {failed}")
    print(f"[OK] Wrote: {args.output.resolve()}")


if __name__ == "__main__":
    main()
