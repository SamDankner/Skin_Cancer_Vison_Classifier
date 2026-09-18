# Project conventions

Keep reusable data, training, evaluation, and utility code in `src/`; notebooks only orchestrate or explore it. Public shared interfaces include `MANIFEST_COLUMNS`, `validate_manifest`, `select_task_manifest`, `make_group_splits`, `build_transforms`, `classification_metrics`, and `persist_run`.

Only ordinary clinical/macro photos are allowed. Never use dermoscopic, microscopic, or pathology-slide inputs. DDI is untouchable final test data unless code receives `allow_final_test=True`; never use it for development decisions.

Normal skin is distinct from benign lesion. Preserve null metadata and unclear binary mappings. Strategies select a task (`lesion_presence`, `diagnosis_binary`, `diagnosis_multiclass`, or `image_quality`), preserve patient/lesion grouping, save weights in `models/<strategy>/`, and use `persist_run` for results.
