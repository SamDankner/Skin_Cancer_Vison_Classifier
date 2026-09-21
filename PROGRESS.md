# Gate-pipeline progress note

## Completed

- Added MCSI official Zenodo download/verification and ingested 400 labelled clinical images.
- Added MSLD v2 original-only ingestion (normal Kaggle authentication required; not downloaded on this host).
- Added healthy versus hard-negative manifest subtype handling, confidence weights, source-aware sampling, and partition metrics.
- Rebuilt `data/processed/development_manifest.csv` with duplicate-safe grouping and zero reported split leakage.
- Fixed the strategy-level hard-negative propagation bug discovered by the first verification audit.
- Focused tests passed: `28 passed in 8.64s`.

## Completed data preparation

`python prepare_data.py --all-development` completed successfully after the validation fix. The preparation report is at `data/processed/preparation_report.json`.

## Datasets downloaded

- `data/raw/MCSI/`: official Zenodo MCSI archive, verified against its published MD5; 400 valid images.
- Existing PAD/SCIN/MILK assets were reused. No DDI or dermoscopy data was used.

## Commands run

- `python -m compileall -q src tests`
- `python -m pytest -q` (completed during the session, terminal output was not retained by the host)
- `python prepare_data.py --all-development`
- Focused tests: `python -m pytest -q tests/test_data_preparation.py tests/test_gate_sampling.py tests/test_productization.py`
- `python run_experiments.py --strategies lesion_presence` twice.

## Verification status

- The first one-epoch run completed but exposed that hard negatives were pre-filtered in `lesion_presence_strategy.train`; it is invalid as a hard-negative verification.
- That selector propagation was fixed.
- The corrected one-epoch hard-negative-inclusive run was stopped after exceeding the user-imposed 30-minute completion window before it saved a run record.

## Remaining steps

1. Run the full test suite again and retain its final output.
2. Re-run one bounded epoch of `lesion_presence` now that hard negatives are actually selected; confirm `dataset_split_summary` includes `other_skin_condition` examples.
3. Read saved validation/development-test partition and source metrics, create the fixed-seed prediction audit, and report them.
4. Do not perform serious multi-epoch training: the manifest's source-target correlation remains severely confounded.
