# Target focal-lesion gate: independent data audit and integration record

## DDI final external-test readiness (2026-09-21)

- Inspected the frozen `configs/final_model.yaml`, model checkpoints,
  manifests, selection code, and final-test directory. `external_test_consumed`
  remains false and `data/final_external_test/` contains only `.gitkeep`; no
  DDI image, manifest, prediction, or evaluation result exists.
- The authentic source is Stanford AIMI's DDI portal:
  `https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50dca81df965`.
  The associated project is `https://ddi-dataset.github.io/` and the source
  paper is Daneshjou et al., *Science Advances* (2022),
  doi:10.1126/sciadv.abq6147. Its Research Use Agreement requires individual
  registration, non-commercial research use, and prohibits redistributing the
  dataset or download link.
- Automatic download is therefore intentionally not possible without a
  registered user's manual agreement. Added `src/data/ddi.py` and
  `run_ddi_final_evaluation.py` to build a manifest only from the official
  `ddi_metadata.csv` direct malignancy flag, preserve diagnoses/unknowns,
  audit exact hash and ID overlap, verify frozen checkpoints, and refuse rerun.
  This is readiness work only: **DDI has not been accessed or evaluated.**

## DDI integration attempt (2026-09-21)

- The registered user placed the authentic official DDI export locally: 656 PNGs
  and `ddi_metadata.csv`. DDI metadata was accessed for integration. Its actual
  columns include `DDI_file`, `skin_tone`, `malignant`, and `disease`; the
  direct official `malignant` flag contains 485 false (benign) and 171 true
  (malignant) rows.
- The first evaluation stopped before inference because the intentional final
  external split has no `train` rows, while the generic loader incorrectly
  tried to fit a target encoding from that empty split. No model outputs,
  performance metrics, subgroup results, or tuning decisions were produced.
- The external loader now uses the frozen diagnosis-binary contract (canonical
  0=benign, 1=malignant) for DDI only, after explicit pre-inference validation.
  Checkpoints, architecture, preprocessing, metadata state, ensemble weights,
  threshold, training data, and validation choices remain unchanged.
- A second pre-inference plumbing issue was identified: the common diagnosis
  selector correctly requires `lesion_present=true`, but the DDI manifest had
  left that required eligibility field null despite each official DDI row being
  a clinical lesion image selected from pathology reports. The DDI builder now
  explicitly sets that eligibility field for rows with a valid official
  malignancy flag. This changes neither binary labels nor any frozen model
  decision; no inference was run while correcting it.
- A third attempt reached model initialization but stopped before prediction
  collection. The runner had written only `ddi_dataset_report.json` and
  `ddi_overlap_audit.json` into `results/final_evaluation/ddi/`; the evaluator
  then treated those same-invocation preflight files as an unsafe rerun. The
  lifecycle guard now admits exactly that preflight-only state, while any
  prediction, metric, report, consumption marker, or unknown artifact still
  blocks rerunning final DDI evaluation.

## 2026-09-20 independent audit outcome

The official archives for `SkinDiseaseClassification` (`10.17632/schhndjbjp.1`),
`MCVSLD` (`10.17632/dfztdtfsxz.1`), and `MonkeyPox`
(`10.17632/st6kggjr23.1`) were downloaded from Mendeley, CRC-validated, kept in
separate raw directories, and quarantined during audit. None passed admission.
No DDI files were accessed, no dermoscopic image was admitted, and no synthetic
or offline-augmented file was used as an independent sample.

### Candidate decisions

| Dataset | Downloaded | Finding | Decision |
|---|---:|---|---|
| Skin Disease Classification | 177,058,379-byte official outer archive | The nested release has 878 images and nine actual classes, not the advertised ten: there is **no Healthy class**. Seven hundred files are ISIC-named dermoscopy. The remaining 178 web/clinical images are only atopic dermatitis and tinea; 70 have supplied augmentation names, 15 are screenshots, and 68 files form 34 exact-copy groups including train/validation copies. No patient IDs or original-image provenance are supplied. | Rejected: provenance, modality, derived-copy, and split-integrity problems; zero rows admitted. |
| MCVSLD | 164,960,513-byte official outer archive | The extracted release contains 11,325 files but only 755 are explicitly marked original; 10,570 are derived copies. The advertised healthy material expands to 1,710 files across supplied splits but only **114** are marked originals, leaving 1,596 derived healthy copies. The full release contains 1,870 exact-duplicate files in 929 groups. One original perceptually overlaps MCSI. Visual review of every healthy original found stock/web imagery, privacy-bar edits, and images with visible focal marks; there is no image-level source traceability or patient grouping. | Rejected: web-derived labels are not defensible no-target-lesion ground truth; zero rows admitted. |
| MonkeyPox aggregate | 190,214,902-byte official outer archive | The 3,000-file release contains 400 files named as originals and 2,600 explicit augmentations. The originals comprise 100 normal, 100 chickenpox, 100 monkeypox, and 100 acne images. **All 400 originals**, including all 100 normal images, perceptually match the already integrated 400-image MCSI collection after resizing/recompression (418 cross-source perceptual pairs). | Rejected as a derived duplicate aggregate; zero rows admitted. |

Additional primary-source search did not identify a safe immediate replacement.
The Zenodo `Skin Lesions Dataset for 2-Class Object Segmentation`
(`10.5281/zenodo.21416759`) advertises 150 healthy backgrounds and 1,000 lesion
images, but the record provides no usable original-image provenance, patient
grouping, or stated license; it was not downloaded or admitted. The new
`DermatoNet-TelessaudeRS` record (`10.17632/g5bj76p2z3.1`) has strong
specialist-validated smartphone provenance but contains only focal-lesion
positives, so it does not address the current healthy/source-balance bottleneck.
`PoxNetX` was rejected at discovery because it is a multi-source aggregate that
explicitly includes synthetic images.

The acquisition utility now records CRC validation and reuses complete local
downloads without redownloading them. Preparation performs these three
source-specific audits on every rebuild and persists the rejection counts and
reasons in `data/processed/preparation_report.json`.

## Phase-one manifest audit

The full development manifest was rebuilt after the candidate audits. Its
approved composition is unchanged: **24,871 total rows** and **15,253
gate-eligible rows**.

| Dataset | Gate positive | Healthy negative | Hard negative | Excluded/not gate-eligible |
|---|---:|---:|---:|---:|
| PAD-UFES-20 | 2,298 | 0 | 0 | 0 |
| MILK10k | 5,240 | 0 | 0 | 5,240 dermoscopic images |
| SCIN | 101 | 249 | 6,419 | 3,637 ambiguous/unsupported rows |
| MCSI | 0 | 100 | 300 | 0 |
| ArsenicSkinImageBD | 0 | 546 | 0 | 741 affected originals; 8,892 augmentations excluded before manifest construction |
| SkinDiseaseClassification | 0 | 0 | 0 | 878 |
| MCVSLD | 0 | 0 | 0 | 11,325 |
| MonkeyPox aggregate | 0 | 0 | 0 | 3,000 |

### Healthy negatives

- Total before audit: **895**.
- Total after audit: **895**; no questionable image was admitted to reach a numeric target.
- By source: ArsenicSkinImageBD 546, SCIN 249, MCSI 100.
- By confidence: moderate 646, weak 249.
- By split: train 652, validation 117, test 126.

### Hard negatives

- Total: **6,719**.
- SCIN: 6,419 = RASH 6,170, ACNE 162, PIGMENTARY_PROBLEM 87.
- MCSI: 300 = acne 100, mpox 100, chickenpox 100.

### Positives

- Total: **7,639**.
- MILK10k 5,240, PAD-UFES-20 2,298, SCIN 101.

### Source/target correlation

| Source | Gate samples | P(target=1 \| source) |
|---|---:|---:|
| ArsenicSkinImageBD | 546 | 0.0000 |
| MCSI | 400 | 0.0000 |
| MILK10k | 5,240 | 1.0000 |
| PAD-UFES-20 | 2,298 | 1.0000 |
| SCIN | 6,769 | 0.0149 |

The rejected candidates therefore did not improve source/target confounding.

### Duplicate and split safety

- Approved manifest: 74 exact-duplicate rows in 31 groups.
- Approved manifest: 635 perceptual pairs in 141 groups.
- Patient/lesion/duplicate groups crossing splits: **0**.
- SkinDiseaseClassification quarantine: 68 exact-copy files in 34 groups.
- MCVSLD quarantine: 1,870 exact-copy files in 929 groups.
- MonkeyPox quarantine: 2 exact-copy files in one exact group, plus all 400 alleged originals matching MCSI perceptually.

### Patient/group and metadata coverage

- PAD-UFES-20: patient and lesion IDs on all 2,298 rows.
- MILK10k: lesion IDs on all 10,480 rows; no patient IDs.
- SCIN: case/participant grouping on all 10,406 rows.
- MCSI and ArsenicSkinImageBD: no source patient/case/lesion IDs; image identity and duplicate groups provide the available leakage protection.
- All three rejected candidates lack usable patient grouping.
- Approved metadata coverage: age 7,518; age group 10,405; sex 17,139; skin tone 6,969; Monk skin tone 10,349; anatomical site 14,288.
- Predictive metadata continues to exclude `dataset` and `source_dataset`.

## Code and test changes

- Added reproducible source-specific quarantine audits for all three candidate Mendeley datasets.
- Added exact-copy, offline-derived-file, screenshot, actual-class, and MCSI perceptual-overlap reporting.
- Kept all rejected rows out of task manifests and downstream diagnosis selection.
- Updated resumable Mendeley acquisition reporting to record CRC validation and reuse existing complete files.
- Added regression tests for MCVSLD original/derived handling, MonkeyPox-to-MCSI overlap, and SkinDiseaseClassification dermoscopy/augmentation/train-validation-copy rejection.
- `python -m compileall -q src tests`: passed.
- `python -m pytest -q`: **81 passed in 77.08s** using a project-writable pytest temporary root because the host Temp directory denies access.

## Bounded verification reference

No candidate was admitted and the gate manifest composition, split counts, seed,
sampler, or training code changed. Repeating the same CPU-only one-epoch run
would therefore be duplicate expensive work, so the existing completed bounded
verification remains the applicable post-audit reference:

- Run: `results/runs/20260921T011247485453Z/lesion_presence/`.
- Pretrained EfficientNetV2-S, 224px, seed 42, one epoch, CPU (CUDA unavailable).
- Confidence-weighted, class/subtype/source-aware replacement sampling with a 1,024-draw epoch budget.
- Training 316.0s; evaluation 361.7s; total experiment 948.6s.
- Balanced accuracy 0.6620; macro-F1 0.6283; ROC-AUC 0.8848; PR-AUC 0.8868.
- Positive sensitivity 0.9701; healthy specificity 0.4683; hard-negative rejection 0.3394.
- Confusion matrix: `[[395, 721], [34, 1105]]`.

### Source-specific held-out performance

| Source | Samples | Relevant held-out result |
|---|---:|---|
| ArsenicSkinImageBD | 82 | healthy/negative specificity 0.4268 |
| MCSI | 66 | negative specificity 0.3788 |
| MILK10k | 804 | positive sensitivity 0.9764 |
| PAD-UFES-20 | 312 | positive sensitivity 0.9712 |
| SCIN | 991 | balanced accuracy 0.5426; sensitivity 0.7391; specificity 0.3461 |
| SkinDiseaseClassification | 0 | rejected; not evaluated |
| MCVSLD | 0 | rejected; not evaluated |
| MonkeyPox aggregate | 0 | rejected; not evaluated |

This remains a source-correlated operating point: excellent focal-lesion recall
does not compensate for the poor healthy and hard-negative rejection rates.
Threshold tuning was not used to conceal the problem.

### Fixed-seed prediction audit (seed 42)

| Category | Source / label | Target | Confidence | P(target) | Output / routing |
|---|---|---:|---|---:|---|
| Healthy skin | MCSI / Healthy | 0 | moderate | 0.6156 | target lesion; routed downstream (incorrect) |
| Mole or nevus | SCIN / GROWTH_OR_MOLE | 1 | weak | 0.6248 | target lesion; routed downstream |
| Benign focal lesion | MILK10k / Nevus | 1 | strong | 0.6342 | target lesion; routed downstream |
| Malignant focal lesion | PAD-UFES-20 / SCC | 1 | strong | 0.7316 | target lesion; routed downstream |
| Acne | MCSI / Acne | 0 | moderate | 0.6395 | target lesion; routed downstream (incorrect) |
| Rash | SCIN / RASH | 0 | weak | 0.4463 | no target lesion; rejected |
| Pox-like | MCSI / Mpox | 0 | moderate | 0.6560 | target lesion; routed downstream (incorrect) |
| Chickenpox | MCSI / Chickenpox | 0 | moderate | 0.4344 | no target lesion; rejected |

## Readiness decision

Healthy data remains below the 1,500-2,000 defensible-image target, only three
approved sources contribute healthy negatives, and source label ratios remain
near-deterministic. Serious gate training is not justified.

NEEDS BOTH MORE HEALTHY NEGATIVES AND SOURCE-BALANCE IMPROVEMENT

## Multimodal preprocessing and reporting audit (2026-09-21)

- Fixed multimodal epoch metrics to pass the persisted `TargetEncoding` class names, so binary diagnostic reports retain numeric labels `0/1` while displaying `benign/malignant` in per-class metrics, confusion matrices, JSON, prediction exports, and run records.
- Added `src/data/metadata.py`: explicit case/whitespace-aware sex mappings; conservative anatomical ontology (upper/lower extremity, hand/foot, head/neck, trunk); Fitzpatrick 1--6 normalization; and explicit `unmapped:<raw>` categories for nonempty values not covered by the audited rules. Missing values remain missing and use the existing ID 0; unseen held-out categories use ID 1.
- Metadata preprocessing still fits age mean/std and categorical vocabularies exclusively on the training split. Raw manifest metadata is preserved; the normalization helper exposes raw, normalized, and source fields for audit use.
- Generated `results/multimodal_metadata_audit.json` from the approved development manifest (6,606 diagnostic rows; MILK10k 5,038 and PAD-UFES-20 1,568). It records raw-to-canonical mappings, source availability, and source/label association without accessing DDI.
- Added focused multimodal regression tests. `python -m compileall -q src tests` passed through `.venv\\Scripts\\python.exe`; no training, hyperparameter search, DDI access, checkpoint/result deletion, or gate changes were performed.

## Final multimodal metadata readiness pass (2026-09-21)

- Canonical metadata normalization is audited in `results/multimodal_metadata_audit.json`. Sex collapses case/whitespace variants to `female`/`male`; site mappings use the conservative upper/lower-extremity, hand/foot, head/neck, and trunk ontology; nonempty unrecognized values remain `unmapped:<value>`; and both numeric and string Fitzpatrick representations such as `1`, `1.0`, `"1"`, and `"1.0"` map to `fitzpatrick_1` (through 6). Raw manifest values are never rewritten.
- Metadata fields are checkpoint-persisted and configurable, including image-only (`[]`), core (`age + sex + anatomical_site`), and experimental skin-tone-enabled subsets. Inference reconstructs the training-only preprocessor and accepts omitted optional fields as explicit missing values.
- Serious multimodal tuning now defaults to `age + sex + anatomical_site`. Skin tone is supported only for controlled ablations because its availability is source-specific and can encode dataset identity; this is not a claim about clinical usefulness.
- `run_multimodal_metadata_ablation.py` performs the bounded validation-only A/B/C comparison (and optional all-missing skin-tone D control): fixed seed 42, DINOv2 ViT-S/14, 196px, at most five epochs, identical base setup, no development-test scoring, no DDI, and a new non-overwriting `results/ablations/multimodal_metadata/` subdirectory. Its report includes overall/source/present-vs-missing validation metrics and availability associations.
- Completed compact controlled screening run: `results/ablations/multimodal_metadata/screening_20260921/ablation_report.json`. It used a fixed seed-42 source/label-stratified cap of 64 records per split/source/label stratum (754 records total), not the full baseline dataset, so it is diagnostic evidence only. Validation macro-F1: image 0.698, core metadata 0.805, core + skin tone 0.823, and skin-tone architecture with every tone missing 0.774. C was much stronger on PAD-UFES-20 (macro-F1 0.937) than MILK10k (0.707); skin tone was present only for PAD rows (100% of present rows) and C's mean malignant probability was 0.716 with tone present versus 0.454 missing. This is source-shortcut evidence, so retain core metadata as the serious default.

Current intended workflow: serious baseline comparison complete -> metadata pipeline corrected -> short multimodal metadata ablation -> ConvNeXt parameter search -> review -> potential multimodal parameter search -> ensemble experiments -> final untouched DDI external evaluation.

## Final validation-only model selection (2026-09-21)

- Architecture winners selected solely by validation macro-F1: EfficientNetV2-S `models/efficientnet/efficientnet_efficientnet_v2_s_20260921T053334483303Z_efficientnet.pt` (0.7983); ConvNeXt-Tiny historical baseline `models/convnext/convnext_convnext_tiny_20260921T060517413359Z_convnext.pt` (0.8410); frozen DINOv2 ViT-S/14 `models/dinov2/dinov2_dinov2_vits14_20260921T064521050559Z_dinov2.pt` (0.7820); and core-metadata DINOv2 `models/multimodal/multimodal_dinov2_vits14_20260921T224823181902Z_multimodal_serious_01.pt` (0.8187). The older ConvNeXt baseline beat the newer serious search winner and was retained.
- The selected frozen system is equal probability averaging of ConvNeXt, EfficientNet, and core-metadata multimodal DINOv2. Its validation macro-F1 was 0.8528 at threshold 0.50; the validation-only threshold scan selected 0.51 with macro-F1 0.8558.
- Frozen specification: `configs/final_model.yaml` and `results/final_selection/frozen_final_model.yaml`. Selection artifacts and aligned predictions are under `results/final_selection/`.
- Frozen development-test evaluation: macro-F1 0.8477, balanced accuracy 0.8463, ROC-AUC 0.9222, sensitivity 0.9253, specificity 0.7673. Outputs are under `results/final_evaluation/`.
- DDI was not accessed. This repository currently has no DDI manifest or images, so final external evaluation remains intentionally pending rather than being approximated or used for any decision.
- The diagnostic model expects an appropriate focal clinical/macro lesion image; it is not a general skin-condition classifier or a clinical diagnostic device. Domain shift, class imbalance, and demographic/acquisition limitations remain material.
