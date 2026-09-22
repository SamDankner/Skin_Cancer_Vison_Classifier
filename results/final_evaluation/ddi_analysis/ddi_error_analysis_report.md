# DDI post-hoc error analysis

## Executive summary

This is an analysis-only report of the already-consumed DDI final test. The frozen equal-weight ensemble and 0.51 threshold were not changed. DDI contained 656 images: 171 malignant and 485 benign. It produced 73 malignant false negatives.

## Performance degradation

Development-test sensitivity was 0.925; DDI sensitivity was 0.573 (change -0.352). Development ROC-AUC was 0.922; DDI ROC-AUC was 0.721.

## Malignant false negatives and model disagreement

Of 73 false negatives, ConvNeXt voted malignant for 40, EfficientNet for 4, and multimodal for 1. 30 were missed by every member. ConvNeXt was positive while the fixed ensemble remained benign for 40 false negatives; this is direct evidence that averaging sometimes suppressed its positive vote, not a basis to alter weights.

## Multimodal missing metadata

Official DDI metadata has no age, sex, or anatomical-site values: all 656 rows received the checkpoint's missing-value representation. The persisted preprocessor replaces age with its training mean and a missing indicator, and encodes categorical missing values with dedicated ID 0. This is direct evidence of a uniform missing-metadata condition. Comparison with development complete/partial metadata is unavailable because the retained development manifest is a placeholder; output shifts alone do not establish causation.

## Dataset and image domain shift

DDI image statistics are saved in `ddi_image_statistics.csv`. Comparable development image statistics cannot be calculated from the retained files, so no numeric image-domain difference is claimed. The substantial prediction-metric shift is observed; its image-level causes require data retained for a future study.

## Skin tone and diagnosis findings

See `skin_tone_subgroup_metrics.csv` and `disease_subgroup_metrics.csv`. Small groups are flagged. These descriptive subgroup results do not establish causal bias. Diagnosis labels are retained exactly as supplied by DDI.

## Representative images and interpretability

Contact sheets under `examples/` are qualitative review aids. CNN saliency maps were not generated: doing so reliably requires reconstructing each exact inference preprocessing/model graph; this post-hoc script intentionally does not risk changing or rerunning the frozen final-test system. No causal conclusion about attention is made.

## Evidence for domain-shift drivers

**Strong evidence:** DDI sensitivity degraded materially; all multimodal auxiliary fields were missing; and the fixed ensemble overruled some ConvNeXt-positive malignant cases. **Moderate evidence:** diagnosis-level and skin-tone subgroup differences are descriptive but may be unstable in small groups. **Hypotheses requiring future study:** broader source-diverse data, metadata-robust multimodal training, lesion localization, acquisition augmentation, calibration, and diagnosis/source-balanced evaluation. Any modified system needs a new untouched external test set, not DDI.

## Limitations

This analysis cannot infer medical mechanisms, causal demographic bias, or model reasoning from probabilities/contact sheets. Standalone DINOv2 was not run because no stored DDI prediction artifact exists and this tool does not rerun protected inference.
