from pathlib import Path
import re


ROOT = Path("/home/nanda/folder belajar/TA-2026/skripsi_nanda_zip_work")
path = ROOT / "src/01-body/bab4.tex"
text = path.read_text(encoding="utf-8")

items = [
    ("fft-only", "examples_fft_only_xgboost.png", "examples-fft-only", "FFT \\textit{only}", "XGBoost"),
    ("clip-only", "examples_clip_only_mlp.png", "examples-clip-only", "CLIP \\textit{only}", "MLP"),
    ("iqa-only", "examples_iqa_only_mlp.png", "examples-iqa-only", "IQA \\textit{only}", "MLP"),
    ("fft-clip", "examples_fft_clip_xgboost.png", "examples-fft-clip", "FFT + CLIP", "XGBoost"),
    ("iqa-fft", "examples_iqa_fft_xgboost.png", "examples-iqa-fft", "IQA + FFT", "XGBoost"),
    ("iqa-clip", "examples_iqa_clip_mlp.png", "examples-iqa-clip", "IQA + CLIP", "MLP"),
    ("iqa-fft-clip", "examples_iqa_fft_clip_xgboost.png", "examples-iqa-fft-clip", "IQA + FFT + CLIP", "XGBoost"),
]

# Remove existing example blocks from a previous run.
for _, _, label, _, _ in items:
    pattern = (
        r"\n\\begin\{figure\}\[t\]\n"
        r"    \\centering\n"
        r"    \\includegraphics\[width=0\.82\\textwidth\]\{assets/pics/examples_[^}]+\}\n"
        r"    \\caption\{Contoh prediksi benar dan salah pada konfigurasi .*?\}\n"
        rf"    \\label\{{fig:{re.escape(label)}\}}\n"
        r"\\end\{figure\}\n\n"
        rf"Gambar \\ref\{{fig:{re.escape(label)}\}}.*?(?=\n\n(?:\\begin\{{figure\}}|\\subsection|\\section))"
    )
    text = re.sub(pattern, "", text, flags=re.S)

for cm_label, filename, example_label, config, model in items:
    marker = f"\\label{{fig:confusion-matrix-{cm_label}}}\n\\end{{figure}}"
    block = f"""\\label{{fig:confusion-matrix-{cm_label}}}
\\end{{figure}}

\\begin{{figure}}[t]
    \\centering
    \\includegraphics[width=0.82\\textwidth]{{assets/pics/{filename}}}
    \\caption{{Contoh prediksi benar dan salah pada konfigurasi {config} dengan model {model}.}}
    \\label{{fig:{example_label}}}
\\end{{figure}}

Gambar \\ref{{fig:{example_label}}} memperlihatkan contoh citra yang diklasifikasikan benar dan salah pada konfigurasi {config}. Contoh tersebut mencakup \\f{{true negative}}, \\f{{true positive}}, \\f{{false positive}}, dan \\f{{false negative}}. Informasi skor probabilitas AI, label sebenarnya, label prediksi, kelas objek, generator, serta nilai fitur yang relevan pada konfigurasi tersebut ditampilkan pada setiap contoh untuk membantu membaca pola prediksi model.
"""
    if marker not in text:
        raise RuntimeError(f"Marker not found: {marker}")
    text = text.replace(marker, block, 1)

path.write_text(text, encoding="utf-8")
print(f"Updated {path}")
