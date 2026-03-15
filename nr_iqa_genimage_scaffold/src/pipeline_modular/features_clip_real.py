"""Extract CLIP image embedding features.

Modes:
- auto: use transformers CLIP if available, else deterministic fallback
- dummy: deterministic fallback only
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract CLIP features")
    parser.add_argument("--manifest", type=Path, default=Path("data/genimage_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/features_clip.csv"))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--out-dim", type=int, default=64)
    parser.add_argument("--mode", choices=["auto", "dummy"], default="auto")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    return parser.parse_args()


def deterministic_vector(key: str, dim: int) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=dim).astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-12
    return vec


def get_clip_backend(mode: str, model_name: str):
    if mode == "dummy":
        return None, None, "dummy"

    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name).to(device)
        model.eval()
        return (processor, model, device), torch, "clip"
    except Exception:
        return None, None, "dummy"


def clip_vector_from_image(path: str, backend, torch_mod, out_dim: int) -> np.ndarray:
    processor, model, device = backend
    image = Image.open(path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch_mod.no_grad():
        feats = model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    vec = feats.squeeze(0).detach().cpu().numpy().astype(np.float32)

    if out_dim < vec.shape[0]:
        vec = vec[:out_dim]
    elif out_dim > vec.shape[0]:
        pad = np.zeros((out_dim - vec.shape[0],), dtype=np.float32)
        vec = np.concatenate([vec, pad], axis=0)
    return vec


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.manifest)
    if args.max_samples > 0:
        df = df.head(args.max_samples)

    backend, torch_mod, source = get_clip_backend(args.mode, args.model_name)

    rows = []
    failed = 0
    for row in df.itertuples(index=False):
        try:
            if source == "clip":
                vec = clip_vector_from_image(row.path, backend, torch_mod, args.out_dim)
            else:
                vec = deterministic_vector(row.path, args.out_dim)

            out = {"image_id": row.image_id, "clip_source": source}
            for i, v in enumerate(vec):
                out[f"clip_feat_{i:03d}"] = float(v)
            rows.append(out)
        except Exception:
            failed += 1

    out_df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)

    print(f"[OK] CLIP features: {len(out_df)}")
    print(f"[OK] Source mode: {source}")
    print(f"[WARN] Failed images: {failed}")
    print(f"[OK] Wrote: {args.output.resolve()}")


if __name__ == "__main__":
    main()
