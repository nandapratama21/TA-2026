# Notebooks

## New classification notebooks
- `01_genimage_manifest.ipynb`: build/check manifest from extracted GenImage.
- `02_feature_extraction.ipynb`: extract FFT mean, CLIP embedding, NR-IQA, periodic features.
- `03_train_eval_classification.ipynb`: merge vector, train classifiers, evaluate results.
- `full_pipeline_classification.ipynb`: one-file end-to-end execution flow.
- `04_persian_cat_fft_clip_baseline.ipynb`: baseline nyata untuk subset BigGAN/Persian cat dengan visualisasi sample, FFT mean, CLIP, merge vector, dan train classifier.
- `05_persian_cat_fft_xgboost_mlp_baseline.ipynb`: baseline eksperimen awal paling sederhana, hanya `FFT mean` + `XGBoost` pada split 80:20 dari subset train Persian cat.
- `06_biggan_random5class_fft_xgboost_mlp.ipynb`: eksperimen `multiple classes, single generator` untuk subset BigGAN/train dengan audit class coverage, FFT mean, XGBoost, dan MLP.

## Legacy
- `full_pipeline_dummy.ipynb`: dummy flow lama (masih disimpan untuk referensi).
