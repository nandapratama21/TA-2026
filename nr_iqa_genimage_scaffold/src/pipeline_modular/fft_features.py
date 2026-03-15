"""Minimal but real FFT feature extraction for dummy pipeline."""

from __future__ import annotations

import hashlib
from typing import Dict

import numpy as np


def _dummy_image_from_id(image_id: str, size: int = 64) -> np.ndarray:
    """Create deterministic pseudo-image from image_id to avoid I/O deps."""
    seed_bytes = hashlib.sha256(image_id.encode("utf-8")).digest()[:8]
    seed = int.from_bytes(seed_bytes, "little", signed=False)
    rng = np.random.default_rng(seed)
    img = rng.normal(loc=127.0, scale=40.0, size=(size, size)).astype(np.float32)
    return np.clip(img, 0, 255)


def extract_fft_features_from_image(img: np.ndarray) -> Dict[str, float]:
    """Extract a tiny FFT descriptor: low/mid/high radial energy + global stats."""
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    h, w = img.shape
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    r_norm = r / (r.max() + 1e-8)

    low = magnitude[r_norm <= 0.2].mean()
    mid = magnitude[(r_norm > 0.2) & (r_norm <= 0.5)].mean()
    high = magnitude[r_norm > 0.5].mean()

    return {
        "fft_mag_mean": float(magnitude.mean()),
        "fft_mag_std": float(magnitude.std()),
        "fft_low_energy": float(low),
        "fft_mid_energy": float(mid),
        "fft_high_energy": float(high),
    }


def extract_fft_features_from_id(image_id: str) -> Dict[str, float]:
    """Convenience wrapper used by dummy pipeline."""
    img = _dummy_image_from_id(image_id)
    return extract_fft_features_from_image(img)
