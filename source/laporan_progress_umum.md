# Laporan Progress Umum

## 1. Ringkasan Arah Penelitian

Penelitian saat ini berfokus pada deteksi citra `AI` vs `nature` menggunakan pendekatan klasifikasi biner berbasis feature vector. Setelah beberapa revisi arah, pipeline utama yang sedang diuji terdiri atas tiga keluarga fitur:

1. Fitur spektral sederhana dari FFT (`fft_mag_mean`, `fft_phase_mean`, `fft_phase_cos_mean`, `fft_phase_sin_mean`)
2. Fitur semantik dari CLIP image embedding
3. Fitur NR-IQA tradisional (`PIQE`, `BRISQUE`, `NIQE`)

Target utama saat ini bukan skor kontinu, tetapi klasifikasi biner yang kuat dan terukur pada beberapa skenario eksperimen bertahap.

## 2. Dataset dan Subset yang Sudah Digunakan

### 2.1 BigGAN - Single Class
- Kelas: `Persian cat`
- Tujuan: sanity check awal apakah FFT sederhana mampu membedakan `AI` vs `nature`
- Evaluasi: holdout `80:20` dari subset `train`

### 2.2 BigGAN - Multiple Classes, Single Generator
- Generator: `BigGAN`
- Jumlah kelas: `5` kelas acak dari ImageNet
- Tujuan: melihat apakah performa FFT tetap kuat ketika variasi konten diperluas
- Evaluasi: holdout `80:20` dari subset `train`

### 2.3 MidJourney - Multiple Classes, Single Generator
- Generator: `MidJourney`
- Jumlah kelas: `4` kelas
- Tujuan: menguji apakah hasil multiple-class juga bertahan pada generator berbeda
- Evaluasi: holdout `80:20` dari subset `train`

### 2.4 BigGAN + MidJourney - Same 4 Classes
- Generator: `BigGAN` dan `MidJourney`
- Kelas yang dipakai sama di kedua generator:
  - `114 -> n01945685`
  - `281 -> n02123045`
  - `654 -> n03769881`
  - `759 -> n04069434`
- Tujuan: menguji generalisasi awal ketika dua generator digabung namun kelas kontennya tetap dikontrol
- Evaluasi: holdout `80:20` dengan stratifikasi gabungan `generator + y_ai + content_label`

## 3. Notebook Eksperimen yang Sudah Dibuat

### 3.1 Notebook 05
File:
`\\wsl.localhost\Ubuntu\home\nanda\folder belajar\TA-2026\nr_iqa_genimage_scaffold\notebooks\05_persian_cat_fft_xgboost_mlp_baseline.ipynb`

Isi:
- eksperimen awal paling sederhana
- subset `Persian cat`
- fitur: `FFT mean`
- model: `XGBoost`, `MLP`

### 3.2 Notebook 06
File:
`\\wsl.localhost\Ubuntu\home\nanda\folder belajar\TA-2026\nr_iqa_genimage_scaffold\notebooks\06_biggan_random5class_fft_xgboost_mlp.ipynb`

Isi:
- eksperimen `multiple classes, single generator`
- subset `BigGAN/train`
- fitur: `FFT mean`
- model: `XGBoost`, `MLP`

### 3.3 Notebook 07
File:
`\\wsl.localhost\Ubuntu\home\nanda\folder belajar\TA-2026\nr_iqa_genimage_scaffold\notebooks\07_midjourney_4class_fft_xgboost_mlp.ipynb`

Isi:
- eksperimen `multiple classes, single generator`
- subset `MidJourney/train`
- fitur: `FFT mean`
- model: `XGBoost`, `MLP`

### 3.4 Notebook 08
File:
`\\wsl.localhost\Ubuntu\home\nanda\folder belajar\TA-2026\nr_iqa_genimage_scaffold\notebooks\08_biggan_midjourney_common4_fft_xgboost_mlp.ipynb`

Isi:
- eksperimen gabungan `BigGAN + MidJourney`
- 4 kelas yang sama
- fitur: `FFT mean`
- model: `XGBoost`, `MLP`

### 3.5 Notebook 09
File:
`\\wsl.localhost\Ubuntu\home\nanda\folder belajar\TA-2026\nr_iqa_genimage_scaffold\notebooks\09_biggan_midjourney_common4_fft_clip_xgboost_mlp.ipynb`

Isi:
- eksperimen gabungan `BigGAN + MidJourney`
- 4 kelas yang sama
- fitur: `FFT mean + CLIP`
- model: `XGBoost`, `MLP`

### 3.6 Notebook 10
File:
`\\wsl.localhost\Ubuntu\home\nanda\folder belajar\TA-2026\nr_iqa_genimage_scaffold\notebooks\10_biggan_midjourney_common4_clip_xgboost_mlp.ipynb`

Isi:
- eksperimen gabungan `BigGAN + MidJourney`
- 4 kelas yang sama
- fitur: `CLIP only`
- model: `XGBoost`, `MLP`

### 3.7 Notebook 11
File:
`\\wsl.localhost\Ubuntu\home\nanda\folder belajar\TA-2026\nr_iqa_genimage_scaffold\notebooks\11_biggan_midjourney_common4_iqa_xgboost_mlp.ipynb`

Isi:
- eksperimen gabungan `BigGAN + MidJourney`
- 4 kelas yang sama
- fitur: `PIQE + BRISQUE + NIQE`
- model: `XGBoost`, `MLP`

## 4. Hasil Penting yang Sudah Didapat

### 4.1 BigGAN - Persian Cat
- FFT mean memberikan hasil sangat tinggi pada eksperimen awal
- MLP mencapai performa sempurna pada satu split tertentu
- XGBoost sekitar `97%`
- interpretasi: FFT mean sangat informatif untuk setting `single class, single generator`

### 4.2 BigGAN - 5 Classes
- multiple-class, single-generator juga menghasilkan performa sangat tinggi
- hal ini menunjukkan bahwa fitur FFT mean bukan hanya kebetulan bekerja pada satu kelas saja

### 4.3 MidJourney - 4 Classes
- histogram fitur menunjukkan overlap distribusi `AI` dan `nature`
- meskipun begitu, model masih bisa mencapai akurasi tinggi
- interpretasi: walaupun fitur per-dimensi overlap secara univariat, kombinasi 4 fitur tetap cukup informatif untuk klasifikasi

### 4.4 BigGAN + MidJourney - FFT + CLIP
- hasil yang sudah dicatat:
  - model terbaik: `XGBoost`
  - accuracy: `0.98`
  - F1: `0.98`
- interpretasi: kombinasi spektral + semantik tetap kuat ketika dua generator digabung pada kelas yang sama

### 4.5 BigGAN + MidJourney - IQA Only
- hasil sekitar `76%`
- interpretasi: fitur IQA tradisional masih membawa sinyal di atas baseline acak, tetapi jauh lebih lemah dibanding FFT dan CLIP

## 5. Validasi dan Sanity Check yang Sudah Dilakukan

### 5.1 Validasi Split
Sudah dicek bahwa pada eksperimen holdout:
- `train_ids` dan `eval_ids` tidak overlap
- distribusi label `AI` dan `nature` tetap seimbang antara train dan eval

### 5.2 Random Feature Sanity Check
Sudah dilakukan uji dengan fitur acak pada eksperimen CLIP-only.
Hasil:
- akurasi sekitar `52%`
- F1 sekitar `52-54%`
- AUROC sekitar `0.51-0.52`

Interpretasi:
- pipeline evaluasi tidak bocor secara sederhana
- performa tinggi pada eksperimen utama memang berasal dari fitur yang diekstrak, bukan dari struktur split yang salah

## 6. Infrastruktur dan Implementasi

### 6.1 Environment
- WSL digunakan sebagai lingkungan kerja utama
- environment conda: `ai`
- CLIP model `openai/clip-vit-base-patch32` sudah berhasil diunduh dan tervalidasi di WSL

### 6.2 Catatan Teknis CLIP
- API `transformers` yang digunakan saat ini mengembalikan `BaseModelOutputWithPooling` pada `get_image_features()`
- notebook dan script sudah dipatch agar mengambil `pooler_output` terlebih dahulu sebelum normalisasi

### 6.3 Struktur File
Notebook dan file pendukung saat ini sudah tersusun rapi di:
- `nr_iqa_genimage_scaffold/notebooks`
- `nr_iqa_genimage_scaffold/data`
- `nr_iqa_genimage_scaffold/artifacts`
- `source`

## 7. Kesimpulan Sementara

Sampai tahap ini, beberapa hal sudah cukup jelas:

1. Fitur `FFT mean` adalah baseline yang sangat kuat pada berbagai skenario awal
2. Ketika generator diperluas dari `BigGAN` ke `MidJourney`, performa masih tetap tinggi
3. Kombinasi `FFT + CLIP` merupakan kandidat terkuat saat ini
4. `CLIP only` dan `IQA only` berguna sebagai pembanding, tetapi perlu dibandingkan langsung dengan hasil FFT-only dan FFT+CLIP untuk menyimpulkan kontribusi masing-masing
5. Untuk setting binary balanced, baseline acak memang berada di sekitar `50%`, sehingga hasil `76%` pada IQA-only berarti ada sinyal, tetapi belum cukup kuat sebagai pendekatan utama

## 8. Langkah Lanjut yang Disarankan

1. Merangkum hasil notebook `08`, `09`, `10`, dan `11` dalam satu tabel komparatif resmi
2. Membandingkan langsung:
   - `FFT only`
   - `CLIP only`
   - `FFT + CLIP`
   - `IQA only`
3. Menambahkan evaluasi multi-seed agar hasil lebih stabil secara statistik
4. Menyiapkan narasi metodologi dan hasil untuk Bab 3 dan Bab 4
5. Jika ingin naik level, eksperimen berikutnya bisa masuk ke lebih banyak generator AI atau skenario unseen generator
