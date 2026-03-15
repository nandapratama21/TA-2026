"""Extract NR-IQA scalar features.

If pyiqa is available, computes NIQE/BRISQUE/PIQE.
Otherwise, falls back to proxy scalars derived from image statistics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract NR-IQA features")
    parser.add_argument("--manifest", type=Path, default=Path("data/genimage_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/features_nriqa.csv"))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--engine", choices=["auto", "pyiqa", "proxy"], default="auto")
    return parser.parse_args()


def load_gray(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32)


def laplacian_var(gray: np.ndarray) -> float:
    # Lightweight Laplacian approximation without cv2.
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    pad = np.pad(gray, ((1, 1), (1, 1)), mode="reflect")
    out = (
        k[0, 1] * pad[:-2, 1:-1]
        + k[1, 0] * pad[1:-1, :-2]
        + k[1, 1] * pad[1:-1, 1:-1]
        + k[1, 2] * pad[1:-1, 2:]
        + k[2, 1] * pad[2:, 1:-1]
    )
    return float(np.var(out))


def proxy_scores(gray: np.ndarray) -> tuple[float, float, float]:
    # Proxies only; these are not official PIQE/BRISQUE/NIQE outputs.
    contrast = float(np.std(gray))
    sharp = laplacian_var(gray)
    hist, _ = np.histogram(gray, bins=256, range=(0, 255), density=True)
    entropy = float(-np.sum(hist * np.log2(hist + 1e-12)))

    brisque = float(np.clip(100.0 - 0.12 * np.sqrt(sharp) - 0.25 * entropy + 0.2 * contrast, 0, 100))
    niqe = float(np.clip(35.0 - 0.15 * entropy + 0.03 * contrast, 0, 100))
    piqe = float(np.clip(70.0 - 0.06 * contrast + 0.0008 * sharp, 0, 100))
    return piqe, brisque, niqe


def get_pyiqa_engine(engine: str):
    if engine == "proxy":
        return None, None
    try:
        import torch
        import pyiqa

        device = "cuda" if torch.cuda.is_available() else "cpu"
        m_niqe = pyiqa.create_metric("niqe", device=device)
        m_brisque = pyiqa.create_metric("brisque", device=device)
        m_piqe = pyiqa.create_metric("piqe", device=device)
        return (m_piqe, m_brisque, m_niqe), torch
    except Exception:
        if engine == "pyiqa":
            raise
        return None, None


def pyiqa_scores(path: str, metrics, torch_mod) -> tuple[float, float, float]:
    m_piqe, m_brisque, m_niqe = metrics
    # pyiqa accepts path input for most metrics.
    with torch_mod.no_grad():
        piqe = float(m_piqe(path).item())
        brisque = float(m_brisque(path).item())
        niqe = float(m_niqe(path).item())
    return piqe, brisque, niqe


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.manifest)
    if args.max_samples > 0:
        df = df.head(args.max_samples)

    metrics, torch_mod = get_pyiqa_engine(args.engine)
    source = "pyiqa" if metrics is not None else "proxy"

    rows = []
    failed = 0
    for row in df.itertuples(index=False):
        try:
            if source == "pyiqa":
                piqe, brisque, niqe = pyiqa_scores(row.path, metrics, torch_mod)
            else:
                gray = load_gray(row.path)
                piqe, brisque, niqe = proxy_scores(gray)
            rows.append(
                {
                    "image_id": row.image_id,
                    "piqe": piqe,
                    "brisque": brisque,
                    "niqe": niqe,
                    "nriqa_source": source,
                }
            )
        except Exception:
            failed += 1

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"[OK] NR-IQA features: {len(out)}")
    print(f"[OK] Source mode: {source}")
    print(f"[WARN] Failed images: {failed}")
    print(f"[OK] Wrote: {args.output.resolve()}")


if __name__ == "__main__":
    main()
