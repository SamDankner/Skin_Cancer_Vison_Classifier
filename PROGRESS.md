# Target focal-lesion gate: independent data audit and integration record

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
