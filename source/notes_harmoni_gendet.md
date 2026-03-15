# Notes: HARMONI Score dan Keterkaitan dengan GenDet (Legacy Notes)

## Status Note
Catatan ini dipertahankan sebagai **arsip konsep** untuk jalur continuous scoring (Z/HARMONI).

Pipeline aktif saat ini sudah dipivot ke:
- **binary classification** real vs AI
- fitur berbasis vektor (FFT mean + CLIP + NR-IQA + periodic optional)

## Inti Konsep (tetap relevan)
1. Masalah distribution gap (GenDet) tetap menjadi landasan evaluasi generalisasi.
2. Komponen spektral + semantik tetap dipakai, namun objective sementara adalah klasifikasi.

## Formula Legacy (opsional lanjutan)
\[
z = a_1 D_{spec} + a_2 P_{periodic} + a_3 G_{phase} + a_4 S_{robust} + a_5 A_{sem} + b
\]
\[
HARMONI = 100 * (1 - \sigma(z)), \quad \sigma(z)=\frac{1}{1 + e^{-z}}
\]

## Referensi Inti
1. GenDet: https://arxiv.org/abs/2312.08880
2. GenImage: https://arxiv.org/abs/2306.08571
3. CLIP: https://arxiv.org/abs/2103.00020
