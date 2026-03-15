"""Build a GenImage manifest from extracted folder structure.

Expected structure example:
  raw_root/<generator>/<split>/<ai|nature>/*.jpg
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build manifest from GenImage folder")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/genimage"),
        help="Root folder of extracted GenImage data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/genimage_manifest.csv"),
        help="Output CSV manifest",
    )
    return parser.parse_args()


def make_image_id(rel_path: str) -> str:
    digest = hashlib.md5(rel_path.encode("utf-8")).hexdigest()[:16]
    return f"img_{digest}"


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw root not found: {raw_root}")

    rows = []
    for p in sorted(raw_root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue

        rel = p.relative_to(raw_root)
        parts = rel.parts
        if len(parts) < 4:
            # Skip files that do not match generator/split/class/image format.
            continue

        generator = parts[0]
        split = parts[1].lower()
        class_name = parts[2].lower()

        if class_name not in {"ai", "nature", "real"}:
            continue

        is_real = 1 if class_name in {"nature", "real"} else 0
        y_ai = 0 if is_real == 1 else 1
        rel_str = rel.as_posix()

        rows.append(
            {
                "image_id": make_image_id(rel_str),
                "path": str(p.resolve()),
                "relative_path": rel_str,
                "generator": generator,
                "split": split,
                "class_name": class_name,
                "is_real": is_real,
                "y_ai": y_ai,
            }
        )

    if not rows:
        raise RuntimeError(
            "No images found for manifest. Check folder structure and extensions."
        )

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"[OK] Manifest rows: {len(df)}")
    print(f"[OK] Wrote: {args.output.resolve()}")


if __name__ == "__main__":
    main()
