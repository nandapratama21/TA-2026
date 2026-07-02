from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "raw" / "genimage"
OUT_DIR = ROOT / "assets" / "pics"

CLASS_INFO = [
    ("281", "n02123045", "tabby"),
    ("283", "n02123394", "Persian cat"),
    ("284", "n02123597", "Siamese cat"),
    ("285", "n02124075", "Egyptian cat"),
]

GENERATORS = ["BigGAN", "MidJourney"]
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
    if label == "ai":
        pattern = f"{numeric_id}_*.png"
    else:
        pattern = f"{wordnet_id}_*.JPEG"

    files = sorted(path for path in folder.glob(pattern) if path.name not in EXCLUDED_NAMES)
    if not files:
        raise FileNotFoundError(f"No file found for {generator} {label} {pattern}")

    # Use a later deterministic index so the examples differ from the earlier
    # one-off sample images while remaining reproducible.
    return files[min(12, len(files) - 1)]


def resize_crop(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    scale = max(size / width, size / height)
    new_size = (round(width * scale), round(height * scale))
    image = image.resize(new_size, Image.Resampling.LANCZOS)
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


def make_grid(label: str, output_name: str):
    tile = 260
    label_h = 56
    header_h = 70
    row_label_w = 150
    gap = 18
    margin = 36

    grid_w = row_label_w + 4 * tile + 3 * gap
    grid_h = header_h + 2 * (tile + label_h) + gap
    canvas = Image.new("RGB", (grid_w + 2 * margin, grid_h + 2 * margin), "white")
    draw = ImageDraw.Draw(canvas)

    font_header = get_font(24, bold=True)
    font_row = get_font(22, bold=True)
    font_label = get_font(18)
    font_class = get_font(18, bold=True)

    title = "Citra AI-generated" if label == "ai" else "Citra Nature"
    draw_centered(
        draw,
        title,
        (margin, 8, margin + grid_w, margin),
        font_header,
        fill=(0, 0, 0),
    )

    x0 = margin + row_label_w
    y0 = margin + header_h

    for col, (_, _, class_name) in enumerate(CLASS_INFO):
        x = x0 + col * (tile + gap)
        draw_centered(draw, class_name, (x, margin + 28, x + tile, margin + header_h), font_class)

    used_files = []
    for row, generator in enumerate(GENERATORS):
        y = y0 + row * (tile + label_h + gap)
        draw_centered(draw, generator, (margin, y, margin + row_label_w - 14, y + tile), font_row)

        for col, (numeric_id, wordnet_id, class_name) in enumerate(CLASS_INFO):
            path = pick_image(generator, label, numeric_id, wordnet_id)
            used_files.append(path.relative_to(ROOT).as_posix())

            image = resize_crop(Image.open(path), tile)
            x = x0 + col * (tile + gap)
            canvas.paste(image, (x, y))
            draw.rectangle((x, y, x + tile, y + tile), outline=(170, 170, 170), width=2)

            source_text = path.name
            draw_centered(draw, source_text, (x, y + tile, x + tile, y + tile + label_h), font_label)

    output_path = OUT_DIR / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)
    return output_path, used_files


def main():
    outputs = [
        make_grid("ai", "bab2_genimage_ai_examples_grid.png"),
        make_grid("nature", "bab2_genimage_nature_examples_grid.png"),
    ]

    for output_path, used_files in outputs:
        print(output_path.relative_to(ROOT).as_posix())
        for file_path in used_files:
            print(f"  - {file_path}")


if __name__ == "__main__":
    main()
