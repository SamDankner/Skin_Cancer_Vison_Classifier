# Final model selection report

All checkpoint, ensemble, and threshold decisions used the internal validation split only. DDI was not accessed.

## Architecture winners

| Architecture | Validation macro-F1 | Retained checkpoint |
| --- | ---: | --- |
| EfficientNetV2-S | 0.7983 | `models/efficientnet/efficientnet_efficientnet_v2_s_20260921T053334483303Z_efficientnet.pt` |
| ConvNeXt-Tiny | 0.8410 | `models/convnext/convnext_convnext_tiny_20260921T060517413359Z_convnext.pt` |
| DINOv2 ViT-S/14 | 0.7820 | `models/dinov2/dinov2_dinov2_vits14_20260921T064521050559Z_dinov2.pt` |
| Multimodal DINOv2 | 0.8187 | `models/multimodal/multimodal_dinov2_vits14_20260921T224823181902Z_multimodal_serious_01.pt` |

The historical ConvNeXt checkpoint beat the serious parameter-search winner (0.8206) and was retained. DINOv2 frozen and last-two-block tuning were essentially tied; frozen marginally led on macro-F1 and was retained. The multimodal component uses age, sex, and anatomical site only; skin tone remains excluded because the prior ablation showed source confounding.

## Ensemble and threshold

Equal-weight ConvNeXt + EfficientNet + multimodal DINOv2 was the strongest validation system, with macro-F1 0.8528 at the default 0.50 threshold. The predefined validation-only threshold grid selected 0.51, yielding macro-F1 0.8558, sensitivity 0.9384, and specificity 0.7576. The complete comparisons, relationships, and threshold rows are in the adjacent CSV/JSON artifacts.

## Frozen system

`configs/final_model.yaml` and `frozen_final_model.yaml` fix the three exact checkpoint hashes, equal weights, 224px ImageNet preprocessing, checkpoint-persisted multimodal preprocessing, class order, and threshold. No development-test result was used to change this system.

## Limitations

This research model expects an appropriate focal clinical/macro lesion photograph. It is not intended to classify arbitrary healthy skin, acne, rashes, cuts, or other non-lesion inputs. Source/domain shift, class imbalance, demographic and acquisition limitations remain; clinical deployment would require substantially more external and prospective validation. This is not a clinical diagnostic device.
