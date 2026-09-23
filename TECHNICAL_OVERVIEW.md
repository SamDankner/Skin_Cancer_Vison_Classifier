# Technical Repository Overview

This is the detailed companion to the [README](README.md). It documents the repository as checked in: a PyTorch research pipeline for ordinary clinical/macro focal-lesion photographs, versioned ensemble inference, and a Streamlit demonstration. It does not reproduce raw data or checkpoint binaries, which are intentionally untracked.

## 1. System Architecture

```text
approved clinical/macro images
  → download/preparation and validated manifest
  → duplicate/group-aware development splits
  → strategy training and validation-only selection
  → checkpoints and persisted run/evaluation artifacts
  → frozen V1 or adaptive V2 deployment configuration
  → image validation, preprocessing, members, ensemble, threshold
  → Streamlit result and optional attribution
```

The repository enforces four important boundaries. Only ordinary clinical/macro images are accepted; dermoscopic, microscopic, and pathology-slide rows are rejected. `lesion_presence`, `diagnosis_binary`, `diagnosis_multiclass`, and `image_quality` are distinct tasks. Normal skin is distinct from benign lesion and uncertain binary mappings remain null. DDI is final-test data: normal development commands reject it, while external evaluation requires `allow_final_test=True` and a frozen contract.

`src/` is reusable implementation code. Notebooks are optional orchestration/review interfaces. Models and raw/processed data are ignored by Git; their expected paths are configuration contracts rather than tracked repository assets.

## 2. Repository Map

```text
app/                  Streamlit app, presentation helpers, CSS, image asset
configs/              strategy, runner, inference, and deployment YAML contracts
data/                 protected-data placeholder and access documentation
notebooks/            optional exploration/training/evaluation notebooks
results/              tracked selection, evaluation, ablation, and analysis outputs
scripts/              adaptive ensemble policy-search command
src/data/             manifests, labels, metadata, downloads, transforms, splits
src/strategies/       CNN, DINOv2, multimodal, crop/localization implementations
src/training/         orchestration, checkpoints, early stopping, optimization
src/evaluation/       metrics, calibration, evaluation, freezing, reporting
src/ensemble/         V2 active-member and aggregation policies
src/explainability/   Grad-CAM, transformer attribution, heatmap rendering
src/inference.py      unified programmatic/runtime inference API
tests/                synthetic-data unit and integration tests
```

Excluded from the map are `.git`, virtual environments, caches, raw/processed data, DDI downloads, mutable run outputs, and `.pt`/`.pth` checkpoints. `.gitignore` explicitly excludes those files.

## 3. Root-Level Files and Commands

| File | Purpose and integration |
| --- | --- |
| `prepare_data.py` | CLI wrapper around `src.data.preparation.build_development_manifest`; prepares selected/all development datasets and provenance. |
| `predict.py` | One-image CLI. Parses image/checkpoint/frozen-config/metadata arguments and calls `src.inference.predict_image`. |
| `run_experiments.py` | Normal development runner; delegates to `src.training.experiment_runner.run_experiments`. |
| `run_final_selection.py` | Aligns development validation exports, selects the retained V1 system/threshold, and writes frozen-selection artifacts. |
| `run_ddi_final_evaluation.py` | Explicit guarded frozen V1 DDI external-final-test command. |
| `run_ddi_posthoc_analysis.py` | Analysis-only reporting over existing DDI outputs: errors, subgroups, confidence, disagreement, image summaries, figures. |
| `run_ddi_v2_posthoc_evaluation.py` | Runs V2 against already inspected DDI and writes a descriptive—not fresh external-test—comparison. |
| `run_multimodal_metadata_ablation.py` | Runs named development image-only/metadata ablations. |
| `audit_multimodal_metadata.py` | Creates development metadata availability/normalization/source audit output. |
| `Dockerfile` | Python 3.11 slim Streamlit container; installs runtime requirements, copies source/config/app and three named checkpoints, exposes 8501. |
| `requirements.txt` | Training/development dependencies: torch, torchvision, pandas, NumPy, scikit-learn, YAML, Pillow, plotting, Jupyter, pytest. |
| `requirements-app.txt` | Runtime dependency set with Streamlit. |
| `pytest.ini` | Sets `tests` as root and excludes data/models/results/caches from discovery. |
| `.gitignore` | Excludes raw/processed data, checkpoints, mutable runs, environments, DDI files, and OS/cache artifacts. |
| `DATA_ACCESS_REQUESTS.md` | Source/access constraints for data not bundled with the repository. |
| `PROGRESS.md` | Historical audit/integration record; documentation only, not runtime input. |
| `AGENTS.md` | Project conventions and public-interface/data-safety constraints. |

`data/final_external_test/.gitkeep` reserves the protected final-test location without shipping DDI. `data/normal_skin/REQUIRES_DATA.md` explains the outstanding normal-skin data need.

## 4. Configuration: `configs/`

`experiments.yaml` is the development-runner root: development manifest, output/model roots, seed/determinism, default inference strategy, enabled strategy YAML files, candidate overrides, and staged parameter search. `inference.yaml` provides nullable default checkpoint/frozen-config values overridden by CLI flags. `ensemble.yaml` is a generic development ensemble schema covering compatible members, averaging, validation-only weight fitting, threshold selection, temperature calibration, and bootstrap settings.

| File(s) | Important fields and consumer |
| --- | --- |
| `efficientnet.yaml`, `convnext.yaml` | `strategy_name`, `task`, architecture, image size, optimizer/LRs, loss, sampling, early stopping, scheduler, augmentation. Consumed by shared CNN training. |
| `dinov2.yaml` | DINOv2 backbone, head/backbone learning rates, unfreeze depth, and training controls. Consumed by `train_dinov2`. |
| `multimodal.yaml` | DINOv2 settings plus `metadata_fields`, embedding/fusion widths, metadata dropout. Consumed by `multimodal_strategy.train`. |
| `lesion_presence.yaml` | Separate gate task and class/source/subtype/label-strength sampling controls. |
| `efficientnet_crop.yaml`, `efficientnet_full_plus_crop.yaml` | Crop/fusion experiment settings; require a genuine box provider and are not standalone deployed localization. |
| `final_model.yaml` | Original frozen V1 contract: checkpoint paths/hashes, class order, equal weights, threshold 0.51, preprocessing, selection metadata. |
| `deployments/v1_frozen/{ensemble,manifest}.yaml` | Versioned V1 deployment contract and variant identity. |
| `deployments/v2_adaptive/{ensemble,manifest}.yaml` | Experimental V2 contract: no-metadata two-model equal policy; metadata-available three-model closest-pair policy. |
| `deployments/v2_adaptive/weighting_search_report.json` | Machine-readable search result referenced by V2. |

All runtime checkpoint paths can be remapped through `MODEL_ROOT`, allowing Windows-authored YAML paths to resolve in a Linux container.

## 5. Application: `app/`

`streamlit_app.py` is the UI entry point. `load_predictor` caches a `SkinCancerPredictor` per `v1_frozen`/`v2_adaptive` variant. `main` sets page settings, accepts one image plus optional age/sex/site, validates bytes through `validate_uploaded_image`, runs prediction, and manages session-state invalidation through `_input_signature` and `_invalidate_if_inputs_changed`.

`ui.py` contains rendering functions: `render_hero`, `render_version_comparison`, `render_usage_guide`, `render_result`, `render_model_outputs`, `render_attribution`, `render_about`, `render_next_steps`, and `render_disclaimer`. `render_attribution` asks the cached predictor for an individual active-member heatmap and displays it with `overlay_heatmap`. `styles.py` exposes the CSS string `CLINICAL_COBALT_CSS`. `assets/pad_ufes_20_close_up_examples.jpg` is the published visual example used in the UI; `__init__.py` marks the package.

## 6. Data: `src/data/`

### Manifest, targets, transforms, and splits

`datasets.py` contains public `MANIFEST_COLUMNS`, `validate_manifest`, `select_task_manifest`, and `ManifestImageDataset`. It normalizes nullable fields, checks modality/DDI safety, selects task-eligible rows without inventing labels, writes manifests, and computes exact/perceptual duplicate groups through `file_sha256`, `image_dhash`, `add_exact_hashes`, `add_perceptual_duplicate_groups`, and `find_perceptual_duplicates`. `build_lesion_presence_manifest` builds the distinct gate view.

`splits.py` provides `make_group_splits`, `leakage_report`, and split class-coverage checks, using groups so related records do not bridge development splits. `targets.py` provides `canonical_target`, `TargetEncoding`, `build_target_encoding`, and `target_encoding_for_manifest`, separating canonical labels from presentation class names. `transforms.py` provides `build_transforms`, with augmented training and deterministic validation/test/inference transforms.

### Labels and metadata

`harmonize_labels.py` defines `LabelDecision`, normalizes source labels, maps only explicit benign/malignant diagnoses, and leaves unclear labels unmapped. `metadata.py` defines conservative sex/anatomical-site/skin-tone aliases, preserves unknown categories as `unmapped:...`, and provides `normalize_metadata_value`, `normalize_metadata_frame`, and `metadata_audit_report`. Raw metadata is not overwritten.

### Preparation and acquisition

`preparation.py` is the central development-data builder. It inventories supported local release layouts, confirms image eligibility, reads source metadata, joins files to records, harmonizes labels, computes hashes/duplicate groups, assigns safe splits, and writes manifest/provenance/readiness reports. Its main interfaces are `build_dataset_manifest`, `build_development_manifest`, and `write_provenance`; it contains source-specific builders for SCIN, PAD, manual gate sources, hard negatives, arsenic-skin data, MCVSLD, monkeypox aggregate audit, and skin-disease archive audit.

`scin_download.py`, `pad_download.py`, `mendeley_download.py`, `mcsi_download.py`, and `msld_download.py` are source-specific acquisition helpers with safe extraction/validation or external-command support. They do not bypass source terms. `ddi.py` is isolated final-test support: `build_ddi_manifest`, `validate_ddi_evaluation_manifest`, and `audit_ddi_overlap` construct/check the official DDI contract and compare hashes/IDs against development data.

## 7. Model Strategies: `src/strategies/`

`cnn_common.py` is the shared CNN implementation. It builds EfficientNet/ConvNeXt models, optional full/crop fusion (`FullImageCropFusion`), datasets/loaders, losses, weighted sampling, optimizer/scheduler loops, validation early stopping, development-test evaluation, persistence, and checkpoint loading. `train_cnn_strategy` is its main orchestration function; `load_cnn_checkpoint` restores saved CNN contracts.

`efficientnet_strategy.py` and `convnext_strategy.py` are thin `build_model`/`train` wrappers around this shared workflow. `lesion_presence_strategy.py` adapts it to the separately constructed lesion-presence manifest.

`dinov2_strategy.py` loads the DINOv2 backbone, determines embedding width, controls last-block trainability, and defines `DinoV2Classifier`. `make_image_loaders`, `_phase`, `train_dinov2`, and `load_dinov2_checkpoint` implement the standalone transformer strategy.

`multimodal_strategy.py` adds structured data to DINOv2. `MetadataPreprocessor` fits categorical vocabularies and age statistics only on training rows, retaining missing/unknown identifiers and an age-missing indicator. `MetadataEncoder` produces metadata embeddings. `MultimodalClassifier` fuses DINOv2 image features with metadata; `ImageOnlyMultimodalClassifier` supports ablation. `train`, `load_multimodal_checkpoint`, `make_metadata_tensors`, and `ablation_config` implement training/loading/controlled variants. Checkpoints persist preprocessor state, so inference reuses training-only statistics and vocabularies.

`localization.py` parses/expands bounding boxes, crops images, calculates IoU/localization metrics, and documents annotation requirements. It is a crop utility, not an included automatic-localization model.

## 8. Training and Utilities

`training/experiment_runner.py` reads/validates YAML, rejects DDI references, creates run folders, resolves named strategy trainers, merges candidate overrides, runs staged screening/serious search, persists environment/config summaries, and writes `results/latest_inference.yaml`. Public `persist_run` saves the common run contract.

`checkpointing.py` creates strategy-scoped checkpoint paths and saves/updates model, optimizer, epoch, and metadata payloads. `early_stopping.py` supplies the `EarlyStopping` dataclass and retained best state. `optimization.py` builds AdamW or momentum SGD plus plateau/cosine/no schedulers. `gate_sampling.py` implements source/class/subtype/label-confidence sampling for the gate. `trainer.py` defines serializable `TrainingResult`.

`utils/device.py`, `seed.py`, `timing.py`, and `reporting.py` respectively choose/summarize compute device, seed Python/NumPy/PyTorch/CUDA, benchmark inference/capture environment, and flatten/append run summaries. Package `__init__.py` files expose selected shared interfaces.

## 9. Evaluation and Ensemble

`evaluation/metrics.py` computes classification and lesion-presence metrics, calibration error, source/gate partitions, and bootstrap intervals. `calibration.py` selects validation-only thresholds, fits/applies temperature scaling, and builds reliability tables. `reporting.py` reads/normalizes run records, compares/selects compatible development models, and writes tables plus confusion/history/ROC/PR/calibration/localization plots.

`evaluator.py` defines `CheckpointBundle`, loaders, checkpoint restoration, prediction collection/export, generic evaluation, frozen-config creation, guarded external evaluation, and external-manifest validation. Its external-test guards require both explicit permission and frozen configuration. `final_selection.py` aligns predictions by stable sample ID, calculates binary metrics, scans/selects thresholds, and serializes selection artifacts. `ddi_analysis.py` derives DDI error, subgroup, probability, false-negative, and disagreement tables. `gate_audit.py` produces categorized lesion-presence audit output.

`evaluation/ensemble.py` validates compatible member predictions and supports equal/fitted probability averaging with missing-member policies. `ensemble/adaptive_ensemble.py` provides `AdaptiveEnsemble`, `EnsembleDecision`, and metadata availability selection for V2. `ensemble/weighting.py` implements equal, static, closest-pair-consensus, and robust MAD/Huber aggregation. Runtime applies the policy already named in YAML; it does not search at prediction time.

## 10. Runtime Inference Flow

1. `predict.py` or Streamlit validates one image with `validate_uploaded_image` and converts it to RGB.
2. `SkinCancerPredictor.from_frozen_config` in `src/inference.py` resolves V1/V2 YAML, model-root paths, checkpoints, class order, task, and persisted preprocessing contract.
3. `SkinCancerPredictor.predict` prepares optional metadata, transforms the image for each active member, and obtains normalized member class probabilities in evaluation/inference mode.
4. V1 combines all three members by the persisted equal-weight contract. V2 uses ConvNeXt+EfficientNet if no metadata exists; otherwise it activates multimodal DINOv2 and the saved closest-pair policy.
5. Any persisted validation-fitted calibration is applied after combination; the stored threshold returns benign/malignant.
6. A `PredictionResult` returns ensemble probability/class, individual predictions, active/inactive members, weights/policy, metadata information, timing, and warnings for the UI or CLI.

`predict_image` is the convenience API, `metadata_from_json` parses optional JSON, and `predict_with_lesion_routing` optionally invokes a separate lesion-presence model before diagnosis.

## 11. Attribution

`explainability/service.py` dispatches individual-member attribution. `gradcam.py` registers a temporary hook on a CNN's final spatial features, backpropagates the malignant logit, and returns a normalized Grad-CAM map. `transformer_attribution.py` hooks the final DINOv2 block input, validates CLS/register/patch token layout, and maps absolute activation-times-gradient patch influence to image space. Metadata remains fixed and is not visualized. `visualization.py` overlays a warm heatmap using `overlay_heatmap`.

These maps are interpretability aids, not lesion segmentation, clinical explanation, or causal evidence.

## 12. Notebooks

`01_data_exploration.ipynb` reviews development manifest quality/transforms. `10_efficientnet.ipynb`, `11_convnext.ipynb`, `12_lesion_presence.ipynb`, `20_dinov2.ipynb`, and `21_multimodal_metadata.ipynb` review corresponding development strategies and ablations. `30_ensemble.ipynb` is validation-only ensemble/freeze exploration. `90_general_model_testing.ipynb` evaluates supported checkpoints on development data. `91_final_external_test.ipynb` is the deliberate frozen DDI workflow. None replaces shared `src/` code.

## 13. Results and Generated Artifacts

`results/experiment_summary.csv` is the flattened cross-run index. `results/final_selection/` contains leaderboard, aligned validation prediction exports, ensemble/threshold scans, frozen config, plots, and `model_selection_report.md`. `results/final_evaluation/` contains development-test metrics/predictions/plots; `ddi/` contains frozen V1 DDI export/report; `ddi_analysis/` contains descriptive errors, subgroups, confidence/disagreement outputs, figures, examples, and its report. `ddi_failed_attempt_*` preserve prior final-test readiness/audit artifacts.

`results/ensemble_search/v2_adaptive/` contains cross-validation, hyperparameter/strategy comparisons, selected policy JSON, and `weighting_report.md`. `results/posthoc_evaluation/ddi/v2_adaptive/` contains V2 predictions/report, completion marker, and V1-versus-V2 descriptive comparison. `results/ablations/multimodal_metadata/screening_20260921/` contains A image-only, B core metadata, C core-plus-skin-tone, and D all-missing experiment configs, histories, predictions, figures, summary, and ablation report. `results/multimodal_metadata_audit.json` is the metadata audit. `results/runs/.gitkeep` reserves the ignored mutable run folder.

PNG files are generated visual diagnostics; CSV/JSON files are machine-readable outputs. Reports distinguish frozen V1 external evaluation from V2 post-hoc results.

## 14. Scripts and Tests

`scripts/search_ensemble_weighting.py` evaluates row-aligned validation prediction exports using repeated stratified cross-validation and a one-standard-error simplicity rule. It compares equal, static, robust, and closest-pair policies and writes the V2 selection artifacts; it does not use DDI for selection.

`tests/conftest.py` provides synthetic fixtures. `test_data_preparation.py`, `test_target_encoding.py`, and `test_shared_infrastructure.py` cover data contracts. `test_strategies.py`, `test_cnn_loss.py`, `test_gate_sampling.py`, and `test_multimodal_metadata.py` cover models/training. `test_evaluation_framework.py`, `test_final_selection.py`, and `test_adaptive_ensemble.py` cover evaluation/selection. `test_inference.py`, `test_explainability.py`, `test_streamlit_app.py`, and `test_productization.py` cover runtime behavior. `test_ddi_analysis.py`, `test_ddi_integration.py`, `test_ddi_output_guard.py`, and `test_ddi_v2_posthoc_evaluation.py` cover final-test and post-hoc boundaries. Run `python -m pytest -q`.

## 15. End-to-End Developer Mental Model

Treat this as two separated systems sharing strict contracts. The **development system** builds an approved manifest, protects grouping and label/null semantics, trains strategies, selects only on validation, persists runs, and freezes exact checkpoint/preprocessing/threshold artifacts. DDI is outside that system until the deliberate final test.

The **runtime system** does no learning or selection. It loads a versioned immutable contract, validates a focal-lesion image, applies checkpoint-persisted preprocessing and optional metadata rules, combines only the configured models, thresholds the result, and optionally provides individual-model attribution. V1 owns the frozen external-test result; V2 is a later adaptive deployment configuration whose DDI comparison is post-hoc. Preserving those distinctions is central to maintaining the repository correctly.
