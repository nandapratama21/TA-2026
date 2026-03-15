# Plan Implementasi - Klasifikasi Real vs AI (Vector-Based)

## 1) Objective
Membangun pipeline klasifikasi biner real-vs-AI dengan vektor fitur gabungan:
1. Spektral FFT (mean magnitude + mean phase)
2. Semantik CLIP (embedding)
3. NR-IQA scalar (PIQE/BRISQUE/NIQE)
4. Periodic artifact feature (opsional, jika referensi paper dipakai)

## 2) Dasar Metode Spektral (sesuai kutipan paper)
"To obtain spectral features, a two-dimensional FFT was applied. The magnitude and phase spectra are computed and their mean values across spatial dimensions are extracted."

Implementasi langsung:
1. FFT 2D pada grayscale image
2. Hitung magnitude dan phase
3. Ambil mean nilai magnitude dan mean nilai phase

## 3) Rekomendasi Referensi Periodic Artifact
Jika periodic dimasukkan sebagai komponen, pakai rujukan primer:
1. Tanaka et al. (2021) - checkerboard artifacts detector
- https://arxiv.org/abs/2108.01892
2. Dzanic et al. (2019) - Fourier spectrum discrepancies
- https://arxiv.org/abs/1911.06465
3. Durall et al. (2020) - up-convolution spectral failures
- https://arxiv.org/abs/2003.01826

## 4) NR-IQA Referensi
1. BRISQUE (TIP 2012)
- https://pubmed.ncbi.nlm.nih.gov/22910118/
2. NIQE documentation + reference
- https://www.mathworks.com/help/images/ref/niqe.html
3. PIQE documentation + reference
- https://www.mathworks.com/help/images/ref/piqe.html

## 5) Struktur Kode (sudah dibuat)
Di `nr_iqa_genimage_scaffold/src/pipeline_modular/`:
1. `build_manifest_genimage.py`
2. `features_fft_mean.py`
3. `features_clip_real.py`
4. `features_nriqa.py`
5. `features_periodic.py`
6. `build_feature_vector.py`
7. `train_classifiers.py`
8. `eval_classifiers.py`

## 6) Struktur Notebook (sudah dibuat)
Di `nr_iqa_genimage_scaffold/notebooks/`:
1. `01_genimage_manifest.ipynb`
2. `02_feature_extraction.ipynb`
3. `03_train_eval_classification.ipynb`
4. `full_pipeline_classification.ipynb`

## 7) Perubahan Markdown (sudah dibuat)
1. `source/progress_harmoni_pipeline.md` -> diubah untuk mencerminkan pivot klasifikasi
2. `source/plan_classification_vector.md` -> plan detail ini
3. `source/notes_harmoni_gendet.md` -> tetap jadi catatan konsep legacy Z/HARMONI

## 8) Protokol Eksperimen
1. In-domain: train/eval pada generator yang sama (split train/val/test)
2. Cross-generator: train pada subset generator A, eval di generator B (uji distribution gap)

## 9) Deliverables Teknis
1. CSV fitur per cabang
2. CSV merged vector
3. Tabel perbandingan model (MLP/SVM/XGBoost)
4. Prediksi per-sample untuk analisis error
