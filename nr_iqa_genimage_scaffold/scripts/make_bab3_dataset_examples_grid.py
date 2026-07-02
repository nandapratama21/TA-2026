from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "raw" / "genimage"
OUT_DIR = ROOT / "assets" / "pics"

CLASSES = [
    ("281", "n02123045", "tabby"),
    ("283", "n02123394", "Persian cat"),
    ("284", "n02123597", "Siamese cat"),
    ("285", "n02124075", "Egyptian cat"),
]

CATEGORIES = [
    ("BigGAN", "ai", "BigGAN AI"),
    ("BigGAN", "nature", "BigGAN Nature"),
    ("MidJourney", "ai", "MidJourney AI"),
    ("MidJourney", "nature", "MidJourney Nature"),
]

EXCLUDED_NAMES = {
    "281_biggan_00000.png",
    "n02123045_10001.JPEG",
    "281_midjourney_0.png",
    "n02123045_10203.JPEG",
}


def get_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def pick_image(generator: str, label: str, numeric_id: str, wordnet_id: str) -> Path:
    folder = DATA_ROOT / generator / "train" / label
    pattern = f"{numeric_id}_*.png" if label == "ai" else f"{wordnet_id}_*.JPEG"
    files = sorted(path for path in folder.glob(pattern) if path.name not in EXCLUDED_NAMES)
    if not files:
        raise FileNotFoundError(f"No image found for {generator} {label} {pattern}")

    return files[min(18, len(files) - 1)]


def resize_crop(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    scale = max(size / width, size / height)
    image = image.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)
    left = (image.width - size) // 2
    top = (image.height - size) // 2
    return image.crop((left, top, left + size, top + size))


def draw_centered(draw: ImageDraw.ImageDraw, text: str, box, font, fill=(20, 20, 20)):
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = left + (right - left - text_width) / 2
    y = top + (bottom - top - text_height) / 2
    draw.text((x, y), text, font=font, fill=fill)


def main():
    tile = 210
    row_label_w = 150
    header_h = 62
    footer_h = 40
    gap = 16
    margin = 30

    grid_w = row_label_w + len(CATEGORIES) * tile + (len(CATEGORIES) - 1) * gap
    grid_h = header_h + len(CLASSES) * (tile + footer_h) + (len(CLASSES) - 1) * gap
    canvas = Image.new("RGB", (grid_w + 2 * margin, grid_h + 2 * margin), "white")
    draw = ImageDraw.Draw(canvas)

    font_title = get_font(24, bold=True)
    font_header = get_font(17, bold=True)
    font_class = get_font(17, bold=True)
    font_file = get_font(11)

    title = "Contoh Dataset Penelitian"
    draw_centered(draw, title, (margin, 6, margin + grid_w, margin), font_title)

    x0 = margin + row_label_w
    y0 = margin + header_h

    for col, (_, _, title_text) in enumerate(CATEGORIES):
        x = x0 + col * (tile + gap)
        draw_centered(draw, title_text, (x, margin + 28, x + tile, margin + header_h), font_header)

    used_files = []
    for row, (numeric_id, wordnet_id, class_name) in enumerate(CLASSES):
        y = y0 + row * (tile + footer_h + gap)
        draw_centered(draw, class_name, (margin, y, margin + row_label_w - 12, y + tile), font_class)

        for col, (generator, label, _) in enumerate(CATEGORIES):
            path = pick_image(generator, label, numeric_id, wordnet_id)
            used_files.append(path.relative_to(ROOT).as_posix())
            image = resize_crop(Image.open(path), tile)
            x = x0 + col * (tile + gap)
            canvas.paste(image, (x, y))
            draw.rectangle((x, y, x + tile, y + tile), outline=(170, 170, 170), width=2)
            draw_centered(draw, path.name, (x, y + tile, x + tile, y + tile + footer_h), font_file)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "bab3_genimage_dataset_examples_grid.png"
    canvas.save(out_path, quality=95)

    print(out_path.relative_to(ROOT).as_posix())
    for file_path in used_files:
        print(f"  - {file_path}")


if __name__ == "__main__":
    main()
