import json
from pathlib import Path


FFT_CELL = '''if FFT_OUT.exists():
    fft_df = pd.read_csv(FFT_OUT)
    print("Loaded FFT features:", FFT_OUT.resolve())
else:
    def load_gray(path):
        return np.asarray(Image.open(path).convert("L"), dtype=np.float32)

    def extract_fft_mean(gray):
        f = np.fft.fft2(gray)
        mag = np.abs(f)
        phase = np.angle(f)
        return {
            "fft_mag_mean": float(np.mean(mag)),
            "fft_phase_mean": float(np.mean(phase)),
            "fft_phase_cos_mean": float(np.mean(np.cos(phase))),
            "fft_phase_sin_mean": float(np.mean(np.sin(phase))),
        }

    fft_rows = []
    for idx, row in enumerate(manifest.itertuples(index=False), start=1):
        gray = load_gray(row.path)
        feats = extract_fft_mean(gray)
        fft_rows.append({"image_id": row.image_id, **feats})
        if idx % 200 == 0:
            print(f"Processed FFT {idx}/{len(manifest)} images")

    fft_df = pd.DataFrame(fft_rows)
    fft_df.to_csv(FFT_OUT, index=False)
    print("Saved FFT features:", FFT_OUT.resolve())

print("FFT shape:", fft_df.shape)
display(fft_df.head())
'''


IQA_SETUP_CELL = '''if IQA_OUT.exists():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loaded existing IQA features later; skip IQA metric initialization.")
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric_piqe = pyiqa.create_metric("piqe", device=device)
    metric_brisque = pyiqa.create_metric("brisque", device=device)
    metric_niqe = pyiqa.create_metric("niqe", device=device)
    print("device:", device)
'''


IQA_CELL = '''if IQA_OUT.exists():
    iqa_df = pd.read_csv(IQA_OUT)
    print("Loaded existing IQA features:", IQA_OUT.resolve())
else:
    from PIL import Image

    IQA_CACHE_DIR = IQA_OUT.parent / "_iqa_safe_images_multiclass"
    IQA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MIN_IQA_SIDE = 256

    def make_iqa_safe_image(path: str) -> str:
        """NIQE needs enough spatial support; upsample small images to avoid zero-size blocks."""
        src = Path(path)
        img = Image.open(src).convert("RGB")
        w, h = img.size

        if min(w, h) >= MIN_IQA_SIDE:
            return str(src)

        scale = MIN_IQA_SIDE / min(w, h)
        new_size = (int(round(w * scale)), int(round(h * scale)))
        safe_path = IQA_CACHE_DIR / f"{src.stem}_{new_size[0]}x{new_size[1]}.png"

        if not safe_path.exists():
            img = img.resize(new_size, Image.Resampling.BICUBIC)
            img.save(safe_path)

        return str(safe_path)

    def extract_iqa_scores(path: str):
        safe_path = make_iqa_safe_image(path)
        with torch.no_grad():
            piqe = float(metric_piqe(safe_path).item())
            brisque = float(metric_brisque(safe_path).item())
            niqe = float(metric_niqe(safe_path).item())
        return {
            "piqe": piqe,
            "brisque": brisque,
            "niqe": niqe,
        }

    iqa_rows = []
    iqa_errors = []

    for idx, row in enumerate(manifest.itertuples(index=False), start=1):
        try:
            scores = extract_iqa_scores(row.path)
            iqa_rows.append({"image_id": row.image_id, **scores})
        except Exception as exc:
            iqa_errors.append({
                "image_id": row.image_id,
                "path": row.path,
                "error": repr(exc),
            })

        if idx % 100 == 0:
            print(f"Processed IQA {idx}/{len(manifest)} images")

    if iqa_errors:
        err_df = pd.DataFrame(iqa_errors)
        err_out = IQA_OUT.parent / "features_iqa_errors_biggan_midjourney_multiclass_train.csv"
        err_df.to_csv(err_out, index=False)
        print("IQA errors:", len(iqa_errors))
        print("Saved IQA error log:", err_out.resolve())
        display(err_df.head())

    if not iqa_rows:
        raise RuntimeError("No IQA features were extracted. Check the IQA error log.")

    iqa_df = pd.DataFrame(iqa_rows)
    iqa_df.to_csv(IQA_OUT, index=False)
    print("Saved IQA features:", IQA_OUT.resolve())

display(iqa_df.head())
'''


CLIP_SETUP_LOCAL = '''if CLIP_OUT.exists():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loaded existing CLIP features later; skip CLIP model initialization.")
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME, local_files_only=True)
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, local_files_only=True).to(device)
    model.eval()
    print("device:", device)
    print("projection_dim:", model.config.projection_dim)
'''


CLIP_CELL_LOCAL = '''if CLIP_OUT.exists():
    clip_df = pd.read_csv(CLIP_OUT)
    print("Loaded CLIP features:", CLIP_OUT.resolve())
else:
    def extract_clip_vector(path: str, out_dim: int = 64):
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.get_image_features(**inputs)
            if hasattr(out, "pooler_output"):
                feats = out.pooler_output
            elif torch.is_tensor(out):
                feats = out
            else:
                raise TypeError(f"Unexpected CLIP output type: {type(out)}")
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        vec = feats.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if out_dim < vec.shape[0]:
            vec = vec[:out_dim]
        elif out_dim > vec.shape[0]:
            vec = np.concatenate([vec, np.zeros(out_dim - vec.shape[0], dtype=np.float32)])
        return vec

    clip_rows = []
    for idx, row in enumerate(manifest.itertuples(index=False), start=1):
        vec = extract_clip_vector(row.path, out_dim=CLIP_OUT_DIM)
        item = {"image_id": row.image_id}
        for i, v in enumerate(vec):
            item[f"clip_feat_{i:03d}"] = float(v)
        clip_rows.append(item)
        if idx % 100 == 0:
            print(f"Processed CLIP {idx}/{len(manifest)} images")

    clip_df = pd.DataFrame(clip_rows)
    clip_df.to_csv(CLIP_OUT, index=False)
    print("Saved CLIP features:", CLIP_OUT.resolve())

display(clip_df.head())
'''


CLIP_CELL_HF = '''if CLIP_OUT.exists():
    clip_df = pd.read_csv(CLIP_OUT)
    print("Loaded CLIP features:", CLIP_OUT.resolve())
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    clip_model.eval()
    print("CLIP device:", device)

    def extract_clip_vector(path: str, out_dim: int = 64):
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = clip_model.vision_model(**inputs)
            feats = out.pooler_output
            feats = clip_model.visual_projection(feats)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        vec = feats.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if out_dim < vec.shape[0]:
            vec = vec[:out_dim]
        elif out_dim > vec.shape[0]:
            vec = np.concatenate([vec, np.zeros(out_dim - vec.shape[0], dtype=np.float32)])
        return vec

    clip_rows = []
    for idx, row in enumerate(manifest.itertuples(index=False), start=1):
        vec = extract_clip_vector(row.path, out_dim=CLIP_OUT_DIM)
        item = {"image_id": row.image_id}
        for i, v in enumerate(vec):
            item[f"clip_feat_{i:03d}"] = float(v)
        clip_rows.append(item)
        if idx % 100 == 0:
            print(f"Processed CLIP {idx}/{len(manifest)} images")

    clip_df = pd.DataFrame(clip_rows)
    clip_df.to_csv(CLIP_OUT, index=False)
    print("Saved CLIP features:", CLIP_OUT.resolve())

display(clip_df.head())
'''


def set_source(cell: dict, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)


def patch_notebook(path: Path) -> bool:
    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if cell.get("cell_type") != "code":
            continue
        if "extract_fft_mean" in src and "to_csv(FFT_OUT" in src:
            set_source(cell, FFT_CELL)
            changed = True
        elif 'pyiqa.create_metric("piqe"' in src and "metric_brisque" in src:
            set_source(cell, IQA_SETUP_CELL)
            changed = True
        elif "extract_iqa_scores" in src and "to_csv(IQA_OUT" in src:
            set_source(cell, IQA_CELL)
            changed = True
        elif "CLIPProcessor.from_pretrained(CLIP_MODEL_NAME, local_files_only=True)" in src and "projection_dim" in src:
            set_source(cell, CLIP_SETUP_LOCAL)
            changed = True
        elif "extract_clip_vector" in src and "model.get_image_features" in src and "to_csv(CLIP_OUT" in src:
            set_source(cell, CLIP_CELL_LOCAL)
            changed = True
        elif "extract_clip_vector" in src and "clip_model.vision_model" in src and "to_csv(CLIP_OUT" in src:
            set_source(cell, CLIP_CELL_HF)
            changed = True
    if changed:
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return changed


def main() -> None:
    nb_dir = Path(__file__).resolve().parents[1] / "notebooks"
    for path in sorted(nb_dir.glob("*multiclass*.ipynb")):
        if not path.name.startswith(tuple(f"{i:02d}_" for i in range(8, 15))):
            continue
        if patch_notebook(path):
            print("patched", path.name)


if __name__ == "__main__":
    main()
