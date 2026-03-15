# Progress Kerja - Pivot ke Klasifikasi Real vs AI

## Ringkasan Pivot
Problem utama sekarang adalah **klasifikasi biner**:
- `0 = real (nature)`
- `1 = AI`

Bukan lagi target utama skor kontinu MOS/HARMONI. Formula `Z/H` tetap disimpan sebagai catatan lanjutan (opsional), tetapi pipeline operasional berfokus pada klasifikasi.

## Status Saat Ini
1. Scaffold klasifikasi sudah ditambahkan ke kode (`src/pipeline_modular`).
2. Alur baru dari manifest -> fitur -> vector -> train/eval classifier sudah tersedia.
3. Notebook klasifikasi baru sudah dibuat.

## Vektor Fitur (Current Plan)
1. **FFT mean features**
- mean magnitude
- mean phase
- mean cos(phase)
- mean sin(phase)

2. **CLIP embedding features**
- vektor embedding image (default dim 64)
- mode auto (real CLIP bila dependency ada), fallback deterministic agar pipeline tetap jalan

3. **NR-IQA scalar features**
- PIQE, BRISQUE, NIQE
- engine auto: pyiqa jika tersedia, fallback proxy jika belum tersedia

4. **Periodic features (opsional)**
- periodic peak ratio
- top-k periodic energy ratio
- periodic score

## Alur Eksekusi
1. `make manifest`
2. `make fft_mean`
3. `make clip_vec`
4. `make nriqa`
5. `make periodic` (opsional)
6. `make vector`
7. `make train_cls`
8. `make eval_cls`

Atau sekali jalan:
- `make cls_pipeline`

## Output Utama
- `artifacts/feature_vector_classification.csv`
- `artifacts/results_classification.csv`
- `artifacts/results_classification_summary.md`
- `artifacts/predictions_classification.csv`

## Evaluasi
Model yang dibandingkan:
1. MLP
2. SVM
3. XGBoost (jika package tersedia)

Metrik:
1. Accuracy
2. F1
3. AUROC
4. AUPRC

## Catatan tentang Z/H (Legacy/Optional)
Formula berikut dipertahankan sebagai opsi riset lanjut jika nanti kembali ke continuous scoring:
\[
Z = a_1 D_{spec} + a_2 P_{periodic} + a_3 G_{phase} + a_4 S_{robust} + a_5 A_{sem} + b
\]
\[
H = 100(1-\sigma(Z))
\]
Saat ini belum dipakai sebagai objective utama eksperimen.
