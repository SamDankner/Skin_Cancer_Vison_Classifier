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
2. Record provenance and build `data/processed/development_manifest.csv` from local files:

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

Only ordinary clinical/macro photographs are allowed. Dermoscopic, microscopic, and pathology-slide inputs are rejected by manifest validation. Review every dataset's license and terms; this repository performs no automatic dataset downloads.

Suggested development locations are:

- `data/raw/PAD-UFES-20/`
- `data/raw/MILK10k/` — use only the ordinary close-up clinical image, never paired dermoscopy
- `data/raw/Fitzpatrick17k/`
- `data/raw/SCIN/`

Sources: [PAD-UFES-20](https://data.mendeley.com/datasets/zr7vgbcyr2/1), [MILK10k](https://doi.org/10.1038/s41597-024-03501-y), [Fitzpatrick17k](https://github.com/mattgroh/fitzpatrick17k), and [SCIN](https://github.com/google-research-datasets/scin).

The manifest preserves null metadata and supports these independent tasks:

- `lesion_presence`: true normal skin versus lesion present
- `diagnosis_binary`: benign versus malignant lesion
- `diagnosis_multiclass`: harmonized lesion diagnosis
- `image_quality`: only when appropriate labels and trained components exist

Normal skin is not a benign lesion. Ambiguous diagnoses remain unmapped rather than being forced into a binary target. Patient, lesion, or image groups must never cross train, validation, and development-test splits. Normalization statistics and metadata vocabularies are fit on training rows only. Validation and test transforms are deterministic.

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
- lesion-presence training/evaluation needs genuine normal-skin negatives;
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

## Protected DDI final external test

DDI belongs only in `data/final_external_test/DDI/`. It is never training, validation, development test, parameter-search, calibration, threshold-selection, or ensemble-selection data.

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
