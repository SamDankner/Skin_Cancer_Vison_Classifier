# Frozen final-model evaluation

The system was frozen before this evaluation: equal-weight ConvNeXt-Tiny + EfficientNetV2-S + age/sex/anatomical-site multimodal DINOv2, threshold 0.51. The full immutable inference contract is [`../../configs/final_model.yaml`](../../configs/final_model.yaml).

## Development-test results

On 968 internal development-test images, the frozen system achieved accuracy 0.8853, balanced accuracy 0.8463, macro-F1 0.8477, weighted F1 0.8851, ROC-AUC 0.9222, PR-AUC 0.9668, sensitivity 0.9253, specificity 0.7673, PPV 0.9215, NPV 0.7769, Brier score 0.0867, and calibration error 0.0148. The confusion matrix was `[[188, 57], [54, 669]]` for `[benign, malignant]`.

No architecture, checkpoint, metadata field, probability weight, preprocessing setting, or threshold was changed after seeing these results.

## DDI external final test

DDI has not been accessed. No DDI manifest or images are present in this workspace, so its protected one-time external evaluation cannot be performed. It remains pending and must use the existing frozen configuration with the guarded `allow_final_test=True` external-evaluation workflow; it must not tune any part of the system.

## Limitations

The system is a research/portfolio diagnostic classifier for suitable focal clinical/macro lesion images, not a general skin-image classifier or a clinical diagnostic device. Dataset source/domain shift, class imbalance, and demographic/acquisition limitations require further external and clinical validation.
