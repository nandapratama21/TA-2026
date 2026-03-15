.PHONY: \
	fft clip train eval modular \
	manifest fft_mean clip_vec nriqa periodic vector train_cls eval_cls cls_pipeline

PYTHON ?= python3
CLIP_MODE ?= dummy
NRIQA_ENGINE ?= proxy

# Legacy dummy targets (kept for backward compatibility)
fft:
	$(PYTHON) src/pipeline/extract_fft.py

clip:
	$(PYTHON) src/pipeline/extract_clip.py

train:
	$(PYTHON) src/pipeline/fuse_regressor.py

eval:
	$(PYTHON) src/pipeline/run_eval.py

modular:
	$(PYTHON) src/pipeline_modular/run_pipeline_dummy.py

# New classification pipeline targets
manifest:
	$(PYTHON) src/pipeline_modular/build_manifest_genimage.py

fft_mean:
	$(PYTHON) src/pipeline_modular/features_fft_mean.py

clip_vec:
	$(PYTHON) src/pipeline_modular/features_clip_real.py --mode $(CLIP_MODE)

nriqa:
	$(PYTHON) src/pipeline_modular/features_nriqa.py --engine $(NRIQA_ENGINE)

periodic:
	$(PYTHON) src/pipeline_modular/features_periodic.py

vector:
	$(PYTHON) src/pipeline_modular/build_feature_vector.py --use-periodic

train_cls:
	$(PYTHON) src/pipeline_modular/train_classifiers.py

eval_cls:
	$(PYTHON) src/pipeline_modular/eval_classifiers.py

cls_pipeline: manifest fft_mean clip_vec nriqa periodic vector train_cls eval_cls
