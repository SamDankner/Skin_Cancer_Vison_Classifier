# Skin Lesion Classification Model

A PyTorch research project for classifying an **already-present focal skin lesion** in an ordinary clinical or macro photograph as benign or malignant. It combines convolutional and transformer-based models, supports optional structured metadata, preserves leakage-aware development workflows, and includes a Streamlit demonstration interface.

> **Research and educational demonstration only.** This project is not a medical device, diagnostic system, or substitute for professional medical assessment.

## Live Demo

Try the public Streamlit demo: [Skin Lesion Classification Model](https://skin-lesion-classifier-422830547981.us-east1.run.app/)

The demo accepts a PNG, JPG, or JPEG photograph of one focal lesion, optionally accepts age, sex, and anatomical-site information, and shows the selected deployment ensemble's malignant probability, member outputs, and optional image attribution. Uploads are processed in memory.

## Project Overview

This project addresses binary benign-versus-malignant classification for suitable close-up clinical/macro photographs. It is intentionally **not** a general skin-image classifier or lesion detector: it assumes a relevant focal lesion is already visible in the image. It is not designed for blank or normal skin, diffuse rashes, acne without a focal lesion, wounds, non-skin photographs, or arbitrary photos containing multiple unrelated regions.

The repository keeps four tasks distinct in its manifest and strategy interfaces:

- `diagnosis_binary` — benign versus malignant lesion classification.
- `diagnosis_multiclass` — harmonized lesion diagnoses when labels support it.
- `lesion_presence` — a separate focal-lesion-versus-no-focal-lesion gate.
- `image_quality` — available only when appropriate labelled data and trained components exist.

Ordinary clinical/macro photos are the only permitted image modality. Dermoscopic, microscopic, and pathology-slide inputs are rejected during manifest validation. Normal skin is deliberately distinct from a benign lesion, and unclear label mappings remain null rather than being forced into a binary class.

## Key Features

- EfficientNetV2-S and ConvNeXt-Tiny image classifiers, plus a DINOv2 ViT-S/14 strategy.
- A multimodal DINOv2 model that can combine image features with optional age, sex, and anatomical-site metadata.
- Two versioned deployment ensembles: an immutable V1 frozen ensemble and a V2 metadata-aware adaptive ensemble.
- Validation-only model, ensemble, calibration, and threshold-selection utilities with patient/lesion/image grouping checks.
- Explicit protected handling for DDI as final external-test data, including manifests, overlap auditing, checkpoint-hash verification, and guarded execution.
- Individual-model image attribution: Grad-CAM for CNNs and gradient-weighted patch-token attribution for DINOv2.
- A Streamlit interface, Docker packaging, command-line inference, reproducibility utilities, and a synthetic-data test suite.

## Model Architecture and Approach

All deployed systems predict a two-class probability distribution in the persisted class order: `benign`, then `malignant`. The selected decision threshold is stored with each deployment configuration rather than inferred at runtime.

| Component | Role |
| --- | --- |
| EfficientNetV2-S | Image-only CNN member trained through the shared CNN workflow. |
| ConvNeXt-Tiny | Image-only CNN member trained through the same shared workflow. |
| DINOv2 ViT-S/14 | Standalone transformer strategy used for development experiments. |
| Multimodal DINOv2 | DINOv2 image backbone plus encoded age, sex, and anatomical-site features fused before the classifier head. |

### V1 frozen ensemble

V1 is the frozen three-member research configuration in [`configs/deployments/v1_frozen/ensemble.yaml`](configs/deployments/v1_frozen/ensemble.yaml), with its source contract retained in [`configs/final_model.yaml`](configs/final_model.yaml). It averages ConvNeXt-Tiny, EfficientNetV2-S, and multimodal DINOv2 probabilities equally and uses a 0.51 malignant threshold. The configuration records exact checkpoint hashes, 224-pixel ImageNet preprocessing, class order, and the checkpoint-persisted metadata preprocessor.

### V2 adaptive ensemble

V2 is an experimental deployment configuration in [`configs/deployments/v2_adaptive/ensemble.yaml`](configs/deployments/v2_adaptive/ensemble.yaml). If no optional metadata is supplied, it uses the ConvNeXt and EfficientNet members with equal probability averaging. When at least one metadata field is supplied, it activates multimodal DINOv2 and applies a saved closest-pair-consensus policy that can downweight a clearly separated third prediction. V2's policy search used validation prediction exports; its later DDI comparison is explicitly post-hoc, not a new untouched external validation.

## Data

The development pipeline builds a validated manifest from locally obtained, approved ordinary clinical/macro image releases. Repository documentation and configuration identify PAD-UFES-20 and MILK10k as the datasets used by the frozen V1 contract; data-preparation support also covers SCIN and several audit/download paths. Raw images, processed manifests, and checkpoints are intentionally Git-ignored because of licensing, privacy, size, and reproducibility constraints.

Development data are split with patient, lesion, duplicate, or image-group safeguards. Training-only rows fit normalization and metadata vocabularies; validation and development-test transforms are deterministic. The DDI dataset is protected final external-test data: normal development commands do not read it, and the dedicated workflow requires an explicit `allow_final_test` opt-in and frozen configuration.

See [DATA_ACCESS_REQUESTS.md](DATA_ACCESS_REQUESTS.md) for access notes and [data/normal_skin/REQUIRES_DATA.md](data/normal_skin/REQUIRES_DATA.md) for the normal-skin data requirement.

## Results

Results below are repository artifacts, not claims of clinical utility.

| Evaluation | Configuration | Summary |
| --- | --- | --- |
| Internal development test | Frozen V1 system, 968 images | ROC-AUC 0.9222, macro-F1 0.8477, sensitivity 0.9253, specificity 0.7673. |
| External final test (DDI) | Frozen V1 system, 656 images | ROC-AUC 0.7210, macro-F1 0.6513, sensitivity 0.5731, specificity 0.7588. |
| Post-hoc DDI analysis | V2 adaptive system, 656 images | Sensitivity 0.6842 and specificity 0.5876; descriptive only because DDI had already been inspected and V2 differs in active members. |

The validation leaderboard selected ConvNeXt-Tiny, EfficientNetV2-S, and multimodal DINOv2 as V1 members. The frozen V1 validation threshold scan selected 0.51, with validation macro-F1 0.8558. Detailed artifacts are in [`results/final_selection/`](results/final_selection/), [`results/final_evaluation/`](results/final_evaluation/), and [`results/posthoc_evaluation/`](results/posthoc_evaluation/). External results and the post-hoc comparison should not be used to retune V1 or to characterize V2 as independently externally validated.

## Model Interpretability

The demo can generate attribution for an individual active model after inference:

- ConvNeXt-Tiny and EfficientNetV2-S use Grad-CAM from the final spatial feature stage.
- Multimodal DINOv2 uses gradient-weighted attribution over image patch tokens entering the final transformer block.

Attribution maps identify image regions that influenced a model's malignant output. They are not a medical explanation, a lesion segmentation, or evidence of causal model reasoning. The DINOv2 visualization is image-only; it does not display metadata contributions.

## Repository Structure

```text
app/             Streamlit entry point, rendering helpers, stylesheet, and demo asset
configs/         Strategy, experiment, inference, and versioned deployment contracts
data/            Data-access guidance and protected/placeholder data locations
notebooks/       Optional exploration, training-review, ensemble, and DDI workflows
results/         Persisted selection, evaluation, ablation, and post-hoc artifacts
scripts/         Validation-prediction-based adaptive ensemble policy search
src/             Reusable data, strategies, training, evaluation, inference, and attribution code
tests/           Offline synthetic-data test suite
```

For a file-by-file onboarding guide, see [TECHNICAL_OVERVIEW.md](TECHNICAL_OVERVIEW.md).

## Running Locally

Create an environment and install the training/development dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

To prepare approved local development data, build a manifest, and validate the configured workflow:

```powershell
python prepare_data.py --all-development
python run_experiments.py --validate-only
```

Run the enabled strategies with:

```powershell
python run_experiments.py
```

The default experiment configuration expects `data/processed/development_manifest.csv`; it controls enabled strategies and their YAML settings through [`configs/experiments.yaml`](configs/experiments.yaml). Training writes checkpoints below `models/<strategy>/` and run artifacts below `results/runs/`.

For the Streamlit demo, install the runtime dependency set and launch the app:

```powershell
python -m pip install -r requirements-app.txt
python -m streamlit run app\streamlit_app.py
```

The app loads the checkpoint paths recorded in the selected V1 or V2 deployment YAML. Those checkpoint files are not tracked by Git; place the expected checkpoint hierarchy under `models/`, or provide `MODEL_ROOT` for a different root. Select `MODEL_VARIANT=v1_frozen` or `MODEL_VARIANT=v2_adaptive` (the application default is V2).

For one-photo CLI inference after training, use:

```powershell
python predict.py path\to\photo.jpg
```

`--checkpoint`, `--frozen-config`, and optional metadata flags are supported; run `python predict.py --help` for the complete interface.

## Docker

The Dockerfile uses `python:3.11-slim`, installs the Streamlit runtime requirements plus system libraries, copies the application, source, configuration, and named model checkpoints, then starts Streamlit on port 8501.

```powershell
docker build -t skin-cancer-classifier .
docker run --rm -p 8501:8501 skin-cancer-classifier
```

The current Dockerfile copies the three named deployment checkpoint files into `/models`; its `MODEL_ROOT` environment variable is set to that location. When using different checkpoint packaging, ensure the paths in the frozen deployment configuration resolve under the configured model root.

## Deployment

The live demo is a containerized Streamlit application. At startup, the UI lazily caches immutable predictor bundles keyed by deployment variant. The predictor resolves versioned YAML, verifies model/checkpoint compatibility, loads the requested members on CUDA when available (otherwise CPU), applies persisted preprocessing, and returns the ensemble result to the UI.

## Limitations

- The classifier expects a suitable focal clinical/macro lesion image and does not locate lesions in arbitrary photographs.
- Normal skin is not treated as benign lesion data; the optional lesion-presence task is separate and requires suitable data/checkpoints.
- Dataset source/domain shift, class imbalance, demographic coverage, acquisition differences, and limited external evidence materially limit generalization.
- Optional metadata can be absent or source-specific; missing values use persisted model preprocessing, not invented information.
- No automatic lesion-localization provider is bundled for crop-dependent workflows.
- V2's DDI comparison is post-hoc and cannot serve as independent external validation.

## Medical / Research Disclaimer

This is research software and an educational portfolio demonstration. It is not a medical device, does not diagnose disease, does not provide treatment advice, and must not be used to make medical decisions or replace evaluation by a qualified healthcare professional.

## Technical Documentation

[TECHNICAL_OVERVIEW.md](TECHNICAL_OVERVIEW.md) provides a detailed guide to the repository architecture, configuration, data workflow, model strategies, inference, attribution, evaluation, deployment, and generated artifacts.
