"""Dummy CLIP feature generator with deterministic values."""

from __future__ import annotations

import hashlib
from typing import Dict

import numpy as np


def extract_clip_features(image_id: str, dim: int = 3) -> Dict[str, float]:
    seed_bytes = hashlib.md5(image_id.encode("utf-8")).digest()[:8]
    seed = int.from_bytes(seed_bytes, "little", signed=False)
    rng = np.random.default_rng(seed)
    vec = rng.random(dim)
    return {f"clip_feat_{i+1:03d}": float(v) for i, v in enumerate(vec)}
