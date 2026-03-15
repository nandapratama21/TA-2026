# GenImage Classification Scaffold (FFT + CLIP + NR-IQA)

Scaffold ini sekarang berfokus pada **klasifikasi biner**: `real` vs `AI`.

## Objective
Membangun vektor fitur gabungan dari:
1. Spektral FFT (`mean magnitude`, `mean phase`)
2. Semantik CLIP (embedding image)
3. NR-IQA scalar (PIQE/BRISQUE/NIQE)
4. Periodic artifact indicator (opsional, berbasis referensi paper)

Lalu melatih beberapa classifier (`MLP`, `SVM`, `XGBoost` bila tersedia).

## Struktur Utama
- `data/raw/genimage/`: data hasil ekstrak GenImage
- `data/genimage_manifest.csv`: manifest auto-generated
- `src/pipeline_modular/build_manifest_genimage.py`
- `src/pipeline_modular/features_fft_mean.py`
- `src/pipeline_modular/features_clip_real.py`
- `src/pipeline_modular/features_nriqa.py`
- `src/pipeline_modular/features_periodic.py`
- `src/pipeline_modular/build_feature_vector.py`
- `src/pipeline_modular/train_classifiers.py`
- `src/pipeline_modular/eval_classifiers.py`

## Alur Eksekusi
1. `make manifest`
2. `make fft_mean`
3. `make clip_vec`
4. `make nriqa`
5. `make periodic` (opsional)
6. `make vector`
7. `make train_cls`
8. `make eval_cls`

Atau jalankan full:
- `make cls_pipeline`

Default Makefile memakai mode aman:
- `CLIP_MODE=dummy`
- `NRIQA_ENGINE=proxy`

Override mode bila perlu:
- `make clip_vec CLIP_MODE=auto` untuk mencoba CLIP nyata.
- `make nriqa NRIQA_ENGINE=pyiqa` untuk memaksa engine pyiqa.

## Output penting
- `artifacts/features_fft_mean.csv`
- `artifacts/features_clip.csv`
- `artifacts/features_nriqa.csv`
- `artifacts/features_periodic.csv`
- `artifacts/feature_vector_classification.csv`
- `artifacts/results_classification.csv`
- `artifacts/results_classification_summary.md`

## Catatan
- `features_clip_real.py` mode `auto` akan mencoba model CLIP nyata, dan fallback ke mode deterministic jika dependency belum tersedia.
- `features_nriqa.py` mode `auto` akan mencoba `pyiqa`, jika tidak ada akan fallback ke proxy scalar agar pipeline tetap berjalan.
