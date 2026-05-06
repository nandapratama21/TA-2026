# Cat4 GenImage Extraction Notes

Target class kucing:

| Index | WNID | Label |
| --- | --- | --- |
| 281 | n02123045 | tabby |
| 283 | n02123394 | Persian_cat |
| 284 | n02123597 | Siamese_cat |
| 285 | n02124075 | Egyptian_cat |

Contoh ekstraksi dari archive GenImage per generator, jalankan di folder archive masing-masing:

```bash
ARCHIVE=imagenet_ai_0419_biggan.zip
OUT=/content/drive/MyDrive/BigGAN/cat4_train_subset

7z l -slt "$ARCHIVE" | sed -n 's/^Path = //p' > /tmp/all_paths.txt
(
  grep '/train/ai/281_' /tmp/all_paths.txt | head -100
  grep '/train/ai/283_' /tmp/all_paths.txt | head -100
  grep '/train/ai/284_' /tmp/all_paths.txt | head -100
  grep '/train/ai/285_' /tmp/all_paths.txt | head -100
  grep '/train/nature/n02123045_' /tmp/all_paths.txt | head -100
  grep '/train/nature/n02123394_' /tmp/all_paths.txt | head -100
  grep '/train/nature/n02123597_' /tmp/all_paths.txt | head -100
  grep '/train/nature/n02124075_' /tmp/all_paths.txt | head -100
) > /tmp/cat4_train_subset.txt

7z x "$ARCHIVE" -o"$OUT" -y -i@/tmp/cat4_train_subset.txt
```

Setelah diekstrak, salin file ke struktur canonical proyek:

```text
nr_iqa_genimage_scaffold/data/raw/genimage/<Generator>/train/ai/
nr_iqa_genimage_scaffold/data/raw/genimage/<Generator>/train/nature/
```
