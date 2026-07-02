import nbformat as nbf
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "19_colab_prepare_genimage_multiclass_subset.ipynb"


def md(source: str):
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def code(source: str):
    return nbf.v4.new_code_cell(source.strip() + "\n")


nb = nbf.v4.new_notebook()
nb["cells"] = [
    md(
        """
# Notebook 19 - Menyiapkan Subset GenImage Multi-Kelas Seimbang di Colab

Notebook ini dibuat untuk mengambil subset GenImage dari archive BigGAN dan MidJourney di Google Drive. Targetnya adalah membuat dataset yang lebih beragam daripada empat kelas kucing, tetapi tetap seimbang berdasarkan:

- subset generator: BigGAN dan MidJourney
- label: `ai` dan `nature`
- kelas ImageNet

Notebook ini dirancang untuk Colab. File GenImage tetap diasumsikan berada di Google Drive dalam bentuk archive, misalnya `imagenet_ai_0419_biggan.zip` beserta part `.z01`, `.z02`, dan seterusnya.
"""
    ),
    md(
        """
## 1. Mount Google Drive dan Setup

Ubah `ARCHIVE_DIR` jika lokasi archive GenImage di Drive berbeda. Untuk split-ZIP, pastikan semua file part `.z01`, `.z02`, ..., `.zip` berada di folder yang sama.
"""
    ),
    code(
        """
from google.colab import drive
import subprocess

drive.mount("/content/drive")

subprocess.run(["apt-get", "-qq", "update"], check=True)
subprocess.run(["apt-get", "-qq", "install", "-y", "p7zip-full"], check=True)
"""
    ),
    code(
        """
from pathlib import Path
import json
import random
import re
import shutil
import subprocess
import urllib.request

import pandas as pd
from tqdm.auto import tqdm

RANDOM_STATE = 42
random.seed(RANDOM_STATE)

# Sesuaikan lokasi ini dengan folder Drive kamu.
ARCHIVES = {
    "BigGAN": {
        "archive_dirs": [
            Path("/content/drive/MyDrive/BigGAN"),
            Path("/content/drive/MyDrive/TA - Nanda Pratama/BigGAN"),
        ],
        "zip_globs": ["imagenet_ai_0419_biggan.zip", "*biggan*.zip"],
    },
    "MidJourney": {
        # Folder lama pada notebook Colab memakai nama "Midjourney".
        # Key manifest tetap "MidJourney" agar konsisten dengan proyek lokal.
        "archive_dirs": [
            Path("/content/drive/MyDrive/Midjourney"),
            Path("/content/drive/MyDrive/MidJourney"),
            Path("/content/drive/MyDrive/TA - Nanda Pratama/Midjourney"),
            Path("/content/drive/MyDrive/TA - Nanda Pratama/MidJourney"),
        ],
        "zip_globs": ["imagenet_midjourney.zip", "*midjourney*.zip"],
    },
}

WORK_ROOT = Path("/content/genimage_multiclass_work")
OUT_ROOT = Path("/content/drive/MyDrive/genimage_multiclass_balanced")
DATA_OUT = OUT_ROOT / "genimage"
MANIFEST_OUT = OUT_ROOT / "genimage_multiclass_manifest.csv"
CLASS_TABLE_OUT = OUT_ROOT / "genimage_multiclass_classes.csv"

WORK_ROOT.mkdir(parents=True, exist_ok=True)
DATA_OUT.mkdir(parents=True, exist_ok=True)

# Target jumlah kelas. Notebook akan memakai sebanyak mungkin kelas valid
# sampai batas ini. Jika kelas valid kurang dari 40, hasilnya bisa 10-40 kelas.
TARGET_NUM_CLASSES = 40
MIN_NUM_CLASSES = 10

# Jumlah citra per kombinasi generator-label-kelas.
# Total = jumlah_kelas * 2 generator * 2 label * N_PER_GROUP.
N_PER_GROUP = 50

print("Output dataset:", DATA_OUT)
print("Manifest:", MANIFEST_OUT)
"""
    ),
    md(
        """
## 2. Muat Mapping ImageNet

File AI GenImage memakai indeks kelas ImageNet, misalnya `281_biggan_00000.png`. File nature memakai WordNet ID, misalnya `n02123045_10001.JPEG`. Karena itu, pencocokan kelas dilakukan melalui mapping ImageNet resmi `class index -> WordNet ID`.
"""
    ),
    code(
        """
DRIVE_MAPPING_PATH = Path("/content/drive/MyDrive/imagenet_class_index.json")
MAPPING_PATH = WORK_ROOT / "imagenet_class_index.json"
CLASS_INDEX_URL = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"

if DRIVE_MAPPING_PATH.exists():
    shutil.copy2(DRIVE_MAPPING_PATH, MAPPING_PATH)
    print("Using ImageNet mapping from Drive:", DRIVE_MAPPING_PATH)
elif not MAPPING_PATH.exists():
    urllib.request.urlretrieve(CLASS_INDEX_URL, MAPPING_PATH)
    print("Downloaded ImageNet mapping:", MAPPING_PATH)

with open(MAPPING_PATH, "r", encoding="utf-8") as f:
    class_index_raw = json.load(f)

class_rows = []
for idx_str, value in class_index_raw.items():
    wnid, label = value
    class_rows.append({
        "class_id": int(idx_str),
        "class_id_str": f"{int(idx_str):03d}",
        "wnid": wnid,
        "label": label,
    })

imagenet_df = pd.DataFrame(class_rows).sort_values("class_id").reset_index(drop=True)
display(imagenet_df.head())
"""
    ),
    md(
        """
## 3. Daftar Kandidat Kelas yang Lebih Bervariasi

Daftar ini sengaja mencampur kategori hewan, kendaraan, objek, dan pemandangan/alam. Jika beberapa kelas tidak tersedia lengkap di BigGAN dan MidJourney, notebook akan melewatinya dan memilih kelas valid berikutnya.
"""
    ),
    code(
        """
# Candidate awal yang bervariasi. Angka mengikuti indeks ImageNet 0-999.
# Kelas dipilih agar tidak hanya kucing: ada hewan, kendaraan, objek, makanan,
# dan elemen alam/pemandangan.
CANDIDATE_CLASS_IDS = [
    # hewan air / alam
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    # burung
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    # mamalia dan hewan darat
    100, 101, 102, 105, 107, 108, 130, 145, 151, 207,
    # kucing/anjing sebagian tetap ada sebagai variasi objek hewan
    208, 209, 210, 281, 283, 284, 285,
    # kendaraan
    407, 436, 468, 511, 555, 569, 573, 575, 609, 656, 717, 751,
    # objek rumah tangga / alat
    504, 508, 530, 531, 533, 549, 550, 620, 621, 673, 760, 849,
    # makanan / buah / tanaman
    928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 948, 949,
    # pemandangan/alam dan struktur
    970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984,
]

candidate_df = imagenet_df[imagenet_df["class_id"].isin(CANDIDATE_CLASS_IDS)].copy()
display(candidate_df.head(60))
print("Jumlah kandidat:", len(candidate_df))
"""
    ),
    md(
        """
## 4. List Isi Archive Tanpa Ekstrak Penuh

Bagian ini memakai `7z l -slt` untuk membaca daftar path di dalam archive. Ini lebih hemat daripada mengekstrak seluruh GenImage terlebih dahulu.
"""
    ),
    code(
        """
def find_archive(generator_name):
    cfg = ARCHIVES[generator_name]
    for archive_dir in cfg["archive_dirs"]:
        if not archive_dir.exists():
            continue
        for zip_glob in cfg["zip_globs"]:
            matches = sorted(archive_dir.glob(zip_glob))
            matches = [m for m in matches if m.suffix.lower() == ".zip"]
            if matches:
                return matches[0]
    searched = []
    for archive_dir in cfg["archive_dirs"]:
        for zip_glob in cfg["zip_globs"]:
            searched.append(str(archive_dir / zip_glob))
    raise FileNotFoundError(
        f"Tidak menemukan archive untuk {generator_name}. Lokasi yang dicek: {searched}"
    )


def list_archive_paths(archive_path):
    cmd = ["7z", "l", "-slt", str(archive_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    paths = []
    for line in proc.stdout.splitlines():
        if line.startswith("Path = "):
            path = line.replace("Path = ", "", 1).strip()
            if path and not path.endswith("/"):
                paths.append(path)
    return paths


archive_paths = {}
for generator in ARCHIVES:
    archive = find_archive(generator)
    print(generator, "archive:", archive)
    paths = list_archive_paths(archive)
    archive_paths[generator] = paths
    print(generator, "files in archive:", len(paths))
    print(paths[:5])
"""
    ),
    md(
        """
## 5. Bangun Indeks File AI dan Nature

AI dikenali dari prefix angka kelas pada nama file, sedangkan nature dikenali dari WordNet ID pada nama file. Keduanya kemudian dicocokkan melalui mapping ImageNet.
"""
    ),
    code(
        """
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def is_image_path(path):
    return path.lower().endswith(IMAGE_EXTENSIONS)


def parse_ai_class_id(path):
    name = Path(path).name
    m = re.match(r"^(\\d{1,3})_", name)
    if not m:
        return None
    return int(m.group(1))


def parse_nature_wnid(path):
    name = Path(path).name
    m = re.match(r"^(n\\d{8})_", name)
    if not m:
        return None
    return m.group(1)


wnid_to_class_id = dict(zip(imagenet_df["wnid"], imagenet_df["class_id"]))
class_id_to_wnid = dict(zip(imagenet_df["class_id"], imagenet_df["wnid"]))
class_id_to_label = dict(zip(imagenet_df["class_id"], imagenet_df["label"]))


def build_archive_index(paths):
    ai = {}
    nature = {}
    for path in paths:
        if not is_image_path(path):
            continue
        lower = path.lower()
        if "/train/ai/" in lower or "\\\\train\\\\ai\\\\" in lower:
            cid = parse_ai_class_id(path)
            if cid is not None:
                ai.setdefault(cid, []).append(path)
        elif "/train/nature/" in lower or "\\\\train\\\\nature\\\\" in lower:
            wnid = parse_nature_wnid(path)
            cid = wnid_to_class_id.get(wnid)
            if cid is not None:
                nature.setdefault(cid, []).append(path)
    return ai, nature


indices = {}
for generator, paths in archive_paths.items():
    ai_idx, nature_idx = build_archive_index(paths)
    indices[generator] = {"ai": ai_idx, "nature": nature_idx}
    print(generator, "AI classes:", len(ai_idx), "Nature classes:", len(nature_idx))
"""
    ),
    md(
        """
## 6. Pilih Kelas Valid Secara Seimbang

Kelas valid berarti kelas tersebut punya minimal `N_PER_GROUP` citra untuk setiap kombinasi:

- BigGAN AI
- BigGAN Nature
- MidJourney AI
- MidJourney Nature
"""
    ),
    code(
        """
def unique_keep_order(values):
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def class_is_valid(class_id):
    for generator in ARCHIVES:
        if len(indices[generator]["ai"].get(class_id, [])) < N_PER_GROUP:
            return False
        if len(indices[generator]["nature"].get(class_id, [])) < N_PER_GROUP:
            return False
    return True


candidate_class_ids = unique_keep_order(CANDIDATE_CLASS_IDS)
valid_candidates = [cid for cid in candidate_class_ids if class_is_valid(cid)]

if len(valid_candidates) < MIN_NUM_CLASSES:
    print("Kandidat tematik kurang dari MIN_NUM_CLASSES. Mencari tambahan dari semua kelas ImageNet...")
    all_class_ids = imagenet_df["class_id"].tolist()
    for cid in all_class_ids:
        if cid not in valid_candidates and class_is_valid(cid):
            valid_candidates.append(cid)
        if len(valid_candidates) >= TARGET_NUM_CLASSES:
            break

selected_class_ids = unique_keep_order(valid_candidates)[:TARGET_NUM_CLASSES]

if len(selected_class_ids) < MIN_NUM_CLASSES:
    raise RuntimeError(
        f"Kelas valid hanya {len(selected_class_ids)}. Turunkan N_PER_GROUP atau cek archive."
    )

selected_classes = imagenet_df[imagenet_df["class_id"].isin(selected_class_ids)].copy()
selected_classes.to_csv(CLASS_TABLE_OUT, index=False)

print("Jumlah kelas dipilih:", len(selected_classes))
display(selected_classes)
"""
    ),
    md(
        """
## 7. Ekstrak Hanya File yang Dipilih

Bagian ini memilih `N_PER_GROUP` citra per kombinasi generator-label-kelas, lalu mengekstraknya ke struktur folder proyek.
"""
    ),
    md(
        """
### Bersihkan Output Lama Sebelum Ekstraksi Ulang

Jalankan cell ini jika proses ekstraksi sebelumnya gagal, berhenti di tengah, atau memakai daftar kelas yang belum diperbaiki. Cell ini hanya menghapus folder output subset yang dibuat notebook ini, bukan archive GenImage asli.
"""
    ),
    code(
        """
# Jalankan sebelum ekstraksi ulang agar file lama tidak tercampur.
for path in [
    DATA_OUT,
    WORK_ROOT / "BigGAN_raw_extract",
    WORK_ROOT / "MidJourney_raw_extract",
]:
    if path.exists():
        print("Removing:", path)
        shutil.rmtree(path)

DATA_OUT.mkdir(parents=True, exist_ok=True)

for path in [MANIFEST_OUT, CLASS_TABLE_OUT]:
    if path.exists():
        print("Removing:", path)
        path.unlink()

rows = []
selected_by_generator = {}
print("Cleanup selesai. Archive asli tidak dihapus.")
"""
    ),
    code(
        """
def write_include_list(paths, out_path):
    out_path.write_text("\\n".join(paths), encoding="utf-8")
    return out_path


def extract_selected_files(archive_path, include_list_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "7z",
        "x",
        "-y",
        str(archive_path),
        f"-o{output_dir}",
        f"-i@{include_list_path}",
    ]
    subprocess.run(cmd, check=True)


rows = []
selected_by_generator = {}

for generator in ARCHIVES:
    archive = find_archive(generator)
    generator_selected = []
    for class_id in tqdm(selected_class_ids, desc=f"Extract {generator}"):
        class_id_str = f"{class_id:03d}"
        wnid = class_id_to_wnid[class_id]
        label_name = class_id_to_label[class_id]

        for folder_label, y_ai in [("ai", 1), ("nature", 0)]:
            candidates = sorted(indices[generator][folder_label][class_id])
            sampled = random.sample(candidates, N_PER_GROUP)
            generator_selected.extend(sampled)

            for internal_path in sampled:
                final_rel = Path(generator) / "train" / folder_label / Path(internal_path).name
                final_path = DATA_OUT / final_rel

                rows.append({
                    "image_id": f"{generator}_{folder_label}_{class_id_str}_{len(rows):06d}",
                    "path": str(final_path),
                    "relative_path": str(final_rel),
                    "archive_path": str(archive),
                    "internal_path": internal_path,
                    "generator": generator,
                    "subset_name": generator,
                    "split": "train",
                    "class_name": folder_label,
                    "content_id": class_id_str if folder_label == "ai" else wnid,
                    "content_label": label_name,
                    "imagenet_id": class_id,
                    "wordnet_id": wnid,
                    "is_real": folder_label == "nature",
                    "y_ai": y_ai,
                })

    selected_by_generator[generator] = generator_selected
    include_list = WORK_ROOT / f"{generator.lower()}_selected_paths.txt"
    write_include_list(generator_selected, include_list)

    raw_extract_dir = WORK_ROOT / f"{generator}_raw_extract"
    if raw_extract_dir.exists():
        shutil.rmtree(raw_extract_dir)

    print(f"Extracting {len(generator_selected)} files from {generator} in one batch...")
    extract_selected_files(archive, include_list, raw_extract_dir)

    print(f"Flattening extracted files into {DATA_OUT / generator}...")
    for row in [r for r in rows if r["generator"] == generator]:
        extracted_matches = list(raw_extract_dir.rglob(Path(row["internal_path"]).name))
        if not extracted_matches:
            raise FileNotFoundError(f"File hasil ekstraksi tidak ditemukan: {row['internal_path']}")
        final_path = Path(row["path"])
        final_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(extracted_matches[0], final_path)

manifest_df = pd.DataFrame(rows)
manifest_df.to_csv(MANIFEST_OUT, index=False)

print("Total citra:", len(manifest_df))
display(manifest_df.head())
display(manifest_df.groupby(["generator", "class_name"]).size().reset_index(name="n"))
display(manifest_df.groupby(["generator", "class_name", "content_label"]).size().head(20).reset_index(name="n"))
"""
    ),
    md(
        """
## 8. Validasi Keseimbangan Dataset

Jika semua benar, jumlah per kombinasi generator-label-kelas harus sama dengan `N_PER_GROUP`.
"""
    ),
    code(
        """
balance = (
    manifest_df
    .groupby(["generator", "class_name", "imagenet_id", "wordnet_id", "content_label"])
    .size()
    .reset_index(name="n")
)

display(balance)

assert balance["n"].min() == N_PER_GROUP
assert balance["n"].max() == N_PER_GROUP

print("Dataset seimbang.")
print("Jumlah kelas:", manifest_df["imagenet_id"].nunique())
print("Jumlah total:", len(manifest_df))
print("Manifest:", MANIFEST_OUT)
print("Class table:", CLASS_TABLE_OUT)
"""
    ),
]

nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}

NOTEBOOK_PATH.write_text(nbf.writes(nb), encoding="utf-8")
print(NOTEBOOK_PATH)
