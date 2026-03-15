"""Utilities for loading tabular inputs."""

from __future__ import annotations

import pandas as pd


def load_manifest(path: str) -> pd.DataFrame:
    """Load image manifest CSV."""
    return pd.read_csv(path)


def load_quality_labels(path: str) -> pd.DataFrame:
    """Load quality label CSV."""
    return pd.read_csv(path)
