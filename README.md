# Skin Cancer Classifier: Macro-Photo ML Infrastructure

This repository supports scientifically cautious classification of ordinary clinical/macro photographs comparable to smartphone photos. Dermoscopy, microscopy, and pathology-slide imagery are rejected by manifest validation.

## Data and safety

Download data only after reviewing terms; no automatic downloads occur. Place development data under `data/raw/PAD-UFES-20/`, `data/raw/MILK10k/`, `data/raw/Fitzpatrick17k/`, and `data/raw/SCIN/`. Use only MILK10k's ordinary close-up clinical image, never paired dermoscopy. Sources: [PAD-UFES-20](https://data.mendeley.com/datasets/zr7vgbcyr2/1), [MILK10k](https://doi.org/10.1038/s41597-024-03501-y), [Fitzpatrick17k](https://github.com/mattgroh/fitzpatrick17k), and [SCIN](https://github.com/google-research-datasets/scin).

DDI belongs only in `data/final_external_test/DDI/` and is an untouched final external test. It cannot be used for training, validation, tuning, calibration, threshold/model selection, or ensemble weights. Code requires explicit `allow_final_test=True` before accessing it.

The unified manifest preserves missing metadata and includes diagnosis, lesion presence, normal skin, image quality, localization, and support fields. Normal skin is not a benign lesion and cannot have a diagnosis or malignancy target. `select_task_manifest` provides independent `lesion_presence`, `diagnosis_binary`, `diagnosis_multiclass`, and `image_quality` views, allowing future normal-skin data without changing loaders.

## Workflow

Reusable code is in `src/`; notebooks call it. Harmonize labels explicitly, make seeded patient/lesion-safe 70/15/15 splits, inspect `leakage_report`, and begin in `notebooks/01_data_exploration.ipynb`. Training transforms are conservative and configurable; evaluation transforms are deterministic. Each run writes configuration, epoch history, metrics, timing, and checkpoints under `results/runs/`, `results/experiment_summary.csv`, and `models/<strategy>/`.
