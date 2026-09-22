# Skin Cancer Classifier: Clinical-Photo Research Pipeline

This project trains and evaluates PyTorch classifiers on ordinary clinical/macro photographs comparable to phone-camera images. It keeps lesion presence, diagnosis, localization, and image-quality tasks separate and protects DDI as a one-time final external test.

This is research software, not a medical device. A prediction is not a diagnosis and must not replace professional assessment.

## Quick start

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Then:

1. Place approved ordinary clinical photographs in the dataset folders described below.
2. Download/resume the official PAD-UFES-20 and SCIN releases, record provenance, and build
   `data/processed/development_manifest.csv` from approved local files:

   ```powershell
   python prepare_data.py --all-development
   ```
3. Assign leakage-safe splits with `make_group_splits`; inspect `leakage_report` and class coverage.
4. Validate the configured workflow:

   ```powershell
   python run_experiments.py --validate-only
   ```

5. Run the enabled development experiment(s):

   ```powershell
   python run_experiments.py
   ```

6. Inspect the new folder under `results/runs/` and `results/experiment_summary.csv`.
7. Classify one photo with the default checkpoint saved by the latest successful run:

   ```powershell
   python predict.py path\to\photo.jpg
   ```

   You may also choose a checkpoint explicitly:

   ```powershell
   python predict.py path\to\photo.jpg --checkpoint models\efficientnet\your_checkpoint.pt
   ```

Normal runs and single-photo inference never access DDI.

## Data rules

Only ordinary clinical/macro photographs are allowed. Dermoscopic, microscopic, and pathology-slide inputs are rejected by manifest validation. Review every dataset's license and terms. SCIN is downloaded only from Google's official `dx-scin-public-data` bucket and PAD-UFES-20 from the official Mendeley record; gated sources remain manual/local acquisitions.

Suggested development locations are:

- `data/raw/PAD-UFES-20/`
- `data/raw/MILK10k/` — use only the ordinary close-up clinical image, never paired dermoscopy
- `data/raw/Fitzpatrick17k/`
- `data/raw/SCIN/dataset/` — official SCIN metadata and extensionless image objects; downloads are resumable and valid existing images are reused

Sources: [PAD-UFES-20](https://data.mendeley.com/datasets/zr7vgbcyr2/1), [MILK10k](https://doi.org/10.1038/s41597-024-03501-y), [Fitzpatrick17k](https://github.com/mattgroh/fitzpatrick17k), and [SCIN](https://github.com/google-research-datasets/scin).

To rebuild strictly from already-downloaded SCIN files, pass
`--skip-scin-download`. Acquisition details, including Google's documented
missing object, are written to `data/raw/SCIN/acquisition_report.json`.

The manifest preserves null metadata and supports these independent tasks:

- `lesion_presence`: target focal-lesion gate; internally `0 = no target focal lesion`, `1 = target focal lesion present`
- `diagnosis_binary`: benign versus malignant lesion
- `diagnosis_multiclass`: harmonized lesion diagnosis
- `image_quality`: only when appropriate labels and trained components exist

Normal skin is not a benign lesion. Acne, rash, dermatitis, tinea, and diffuse pigmentary conditions are preserved as OTHER and excluded from the default binary gate; the binary model does not have a third class. Ambiguous diagnoses remain unmapped rather than being forced into a binary target. Patient, lesion, duplicate, or image groups must never cross train, validation, and development-test splits. Normalization statistics and metadata vocabularies are fit on training rows only. Validation and test transforms are deterministic.

## Repository layout

- `run_experiments.py`: one-command development runner
- `predict.py`: one-photo inference CLI
- `configs/experiments.yaml`: enabled strategies, staged search, paths, and seed
- `configs/*.yaml`: strategy settings
- `src/data/`: manifests, labels, group-safe splits, and transforms
- `src/strategies/`: EfficientNetV2, ConvNeXt, DINOv2, multimodal, lesion presence, and crop support
- `src/training/`: orchestration, early stopping, optimization, checkpoints, and run contracts
- `src/evaluation/`: metrics, calibration, ensembles, reports, plots, and protected external testing
- `src/inference.py`: unified programmatic inference API
- `notebooks/`: exploration and review; production training does not require notebooks
- `models/<strategy>/`: best checkpoints
- `results/runs/<run_id>/`: invocation and per-strategy artifacts

## Configure and run experiments

Edit YAML rather than Python to choose work. The default `configs/experiments.yaml` enables only EfficientNet so a normal Run action does not launch every expensive strategy. Enable or disable `efficientnet`, `convnext`, `dinov2`, `multimodal`, and `lesion_presence` independently, or select enabled strategies at the command line:

```powershell
python run_experiments.py --strategies efficientnet convnext
```

Every normal invocation performs:

1. configuration validation;
2. manifest and DDI checks;
3. patient/lesion/image leakage checks;
4. split and class-coverage validation;
5. training and validation each epoch;
6. validation-only early stopping and model selection;
7. restoration of the best checkpoint;
8. independent development-test evaluation;
9. runtime and batch-size-1 inference timing;
10. prediction, metric, plot, environment, and configuration persistence.

The runner writes `results/latest_inference.yaml` after success so `python predict.py photo.jpg` uses the configured `default_inference_strategy`. Use `--checkpoint` or `--frozen-config` to override it.

### Staged parameter search

Set `parameter_search.enabled: true` in `configs/experiments.yaml` to use the practical two-stage search:

- screening: smaller images and fewer epochs across a small set of plausible configurations;
- serious training: only the validation-ranked top `top_k` candidates at the larger settings.

Selection uses validation metrics only. Development-test metrics are recorded for reporting and never rank candidates. DDI is never involved. Supported configurable dimensions include learning rates, weight decay, optimizer (`adamw` or momentum `sgd`), scheduler (`reduce_on_plateau`, `cosine`, or `none`), batch size, image size, dropout, loss (`cross_entropy`, weighted cross-entropy, or focal), weighted sampling, focal gamma, augmentation, freeze depth, epoch limit, scheduler settings, and early-stopping patience/minimum improvement. DINOv2 candidates use head/backbone learning rates and unfreeze depth; multimodal candidates additionally use metadata width, fusion width, and metadata dropout. Crop margin and full/crop/fusion modes apply only when a legitimate localization source exists.

## Multimodal training and optional metadata

The multimodal strategy now follows the same lifecycle as image-only models: build, train, validate each epoch, early stop, restore the best state, evaluate the development test, save, reload, and infer.

Its checkpoint preserves:

- image backbone and fusion architecture;
- selected metadata feature allow-list;
- categorical vocabularies and unknown/missing IDs;
- age mean, standard deviation, and missing indicator policy;
- class order and task;
- image preprocessing and resolution;
- all training parameters, early-stopping details, threshold, and calibration fields.

Metadata is optional at inference. Missing age is represented by the training mean plus an explicit missing indicator; this is a neutral numerical encoding learned during training, not an assertion about the person's age. Missing categories receive a dedicated missing ID. No medically meaningful value is invented.

Examples:

```powershell
python predict.py photo.jpg --checkpoint models\multimodal\model.pt
python predict.py photo.jpg --checkpoint models\multimodal\model.pt --age 55 --sex female --site torso
python predict.py photo.jpg --checkpoint models\multimodal\model.pt --metadata metadata.json
```

The programmatic interface is:

```python
from src.inference import predict_image

result = predict_image("photo.jpg", checkpoint="models/efficientnet/model.pt")
result_with_metadata = predict_image(
    "photo.jpg",
    checkpoint="models/multimodal/model.pt",
    metadata={"age": 55, "sex": "female", "anatomical_site": "torso"},
)
```

Image-only models ignore supplied metadata with an explicit warning. Frozen ensembles use the persisted missing-member policy and renormalize only when that policy allows it. Crop-dependent members cannot run from one photo unless a real serialized automatic localization provider exists.

## Single-photo output and limitations

Inference loads the persisted class order and preprocessing, selects CUDA when available, switches models to evaluation mode, uses inference mode, and reports normalized probabilities. Output includes the task, predicted class, confidence, task-appropriate probability fields, metadata usage, model/checkpoint identity, device, timing, and warnings.

The CLI validates file existence, supported extension, readability, corruption, and channel count. It does not claim image-quality, OOD, localization, or lesion-presence results unless an appropriate trained component is actually loaded.

Current data-dependent limitations are intentional:

- no serialized automatic lesion-localization provider is bundled;
- the current normal-skin negatives are weak SCIN self-reports; expert-labelled same-source ImageQX/Muhaba negatives still require access;
- image-quality and OOD behavior needs suitable labelled datasets and trained components;
- no performance metrics are included until real experiments are run.

These optional limitations do not prevent full-image diagnosis models from training, saving, reloading, or performing photo-only inference.

## Run outputs

A run is organized as `results/runs/<run_id>/<strategy>/`. Depending on task and search mode it contains:

- complete YAML configuration and invocation summary;
- seed, environment, package, CUDA, and GPU information;
- dataset/split/class summaries;
- epoch history with train/validation loss, performance, learning rate, and epoch time;
- best epoch, stop epoch, stop reason, patience, `min_delta`, and monitored value;
- best checkpoint path;
- validation and development-test metrics;
- row-aligned development-test predictions;
- training, validation, evaluation, total invocation, and batch-size-1 inference timing;
- training-history, confusion-matrix, ROC, precision-recall, and calibration plots where applicable;
- staged-search comparison CSV/JSON/plot when search is enabled.

`results/experiment_summary.csv` provides a sortable cross-run index. Checkpoints under `models/<strategy>/` contain the inference contract, so users do not reconstruct class order or preprocessing manually.

## Calibration, ensembling, and freezing

Calibration, thresholds, and ensemble weights may be fit only on validation predictions. Compatible ensemble members must share task, class order, and sample order. `notebooks/30_ensemble.ipynb` supports development-only selection and creates an immutable frozen configuration before final testing. Ground-truth crops are oracle ablations and must never be described as deployable end-to-end performance.

The reproducible completed selection command is `\.venv\Scripts\python.exe run_final_selection.py`. It writes the validation leaderboard, ID-aligned component predictions, equal-weight ensemble comparison, validation threshold scan, and frozen system configuration. The current frozen system is in `configs/final_model.yaml`; it averages the retained ConvNeXt-Tiny, EfficientNetV2-S, and age/sex/anatomical-site multimodal DINOv2 checkpoints at a threshold of 0.51. It never opens DDI.

## Protected DDI final external test

DDI belongs only in `data/final_external_test/DDI/`. It is never training, validation, development test, parameter-search, calibration, threshold-selection, or ensemble-selection data.

## Final DDI external evaluation

DDI has not been accessed by this repository. Obtain it individually from the
official Stanford AIMI portal at https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50dca81df965,
accept the Stanford DDI Research Use Agreement, and place the official
`ddi_metadata.csv` plus images under `data/final_external_test/ddi/` (normally
`images/`). Do not use mirrors or share the download link. The official DDI
code specifies a direct `malignant` / `malignancy(malig=1)` field; that field,
not a guessed diagnosis mapping, drives this project's binary evaluation.

Once the registered user has completed that manual access step, run exactly
once:

```powershell
.\.venv\Scripts\python.exe run_ddi_final_evaluation.py --allow-final-test
```

It preserves the immutable download outside Git, writes the DDI manifest,
checks image hashes and identifiers against the development manifest, verifies
frozen checkpoint hashes, and refuses a non-empty output directory. Results go
to `results/final_evaluation/ddi/`; no threshold, preprocessing, metadata
vocabulary, ensemble weight, or checkpoint is fit after DDI access.

Do not add DDI to `configs/experiments.yaml`. Normal `python run_experiments.py` and `python predict.py` cannot consume it. After all development choices are frozen, use `notebooks/91_final_external_test.ipynb` deliberately. Access requires both `allow_final_test=True` and an existing immutable `results/final_model/frozen_config.yaml`; the external-test workflow verifies checkpoint hashes and prevents automatic reruns.

## Notebooks

- `01_data_exploration.ipynb`: development-manifest review
- `10_efficientnet.ipynb`, `11_convnext.ipynb`, `20_dinov2.ipynb`: strategy review/examples
- `12_lesion_presence.ipynb`: separate lesion-presence task
- `21_multimodal_metadata.ipynb`: multimodal training and ablation review
- `30_ensemble.ipynb`: validation-only ensemble/calibration/freeze workflow
- `90_general_model_testing.ipynb`: universal development evaluation
- `91_final_external_test.ipynb`: intentional one-time DDI evaluation

Notebooks are optional interfaces around shared `src/` code. The production workflow is `run_experiments.py`.

## Tests and reproducibility

Run:

```powershell
python -m pytest -q
```

Tests use synthetic data and tiny offline models; they do not require medical datasets or pretrained downloads. Seeds are applied to Python, NumPy, PyTorch, and CUDA. The default configuration allows cuDNN benchmarking for practical GPU speed; enable deterministic mode when repeatability is more important than throughput.

## Live Demo

The Clinical Cobalt Streamlit demo accepts one ordinary clinical/macro skin-lesion photograph and optional age, sex, and anatomical-site metadata. Missing values use the multimodal checkpoint's persisted missing-value preprocessing; uploads are processed in memory and are not saved.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-app.txt
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

The app caches exactly the frozen ConvNeXt-Tiny, EfficientNetV2-S, and multimodal DINOv2 members in `configs/final_model.yaml`, using its 0.51 threshold. The standalone DINOv2 development model is never loaded or displayed. This is a research and educational demonstration, not a medical device or diagnostic tool.

### Demo architecture

```text
Image + optional metadata
          ↓
Frozen preprocessing
          ↓
ConvNeXt-Tiny ───────┐
EfficientNetV2-S ────┼─→ Frozen Ensemble → Prediction
Multimodal DINOv2 ───┘
```

### Docker

The CPU-ready image intentionally excludes checkpoint files. Mount the frozen `models` directory at runtime; the container reads it through `MODEL_ROOT=/models`. `FROZEN_CONFIG_PATH` can override the default configuration when needed.

```powershell
docker build -t skin-cancer-classifier .
docker run --rm -p 8501:8501 -v "${PWD}\models:/models:ro" skin-cancer-classifier
```

Screenshot placeholder: add a verified local demo screenshot after launch; no synthetic prediction screenshots are included. The Docker image has not been built in this environment.
