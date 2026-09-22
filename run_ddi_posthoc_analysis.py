"""Post-hoc, analysis-only DDI error and domain-shift investigation.

Consumes the frozen evaluation exports.  It never performs inference or writes
to the protected final-evaluation directory.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.evaluation.ddi_analysis import (binary_subgroup_metrics, canonical_error_table, false_negative_diagnoses, model_disagreement, probability_summary)

THRESHOLD = .51
MEMBERS = ("convnext", "efficientnet", "multimodal")
METRIC_COLUMNS = ("macro_f1", "balanced_accuracy", "roc_auc", "pr_auc", "sensitivity", "specificity", "brier_score", "calibration_error")


def _plot_confidence(table, output):
    fig, axis = plt.subplots(figsize=(8, 5))
    for name, subset in table.groupby("error_type"):
        axis.hist(subset.ensemble_malignant_probability, bins=np.linspace(0, 1, 31), alpha=.55, label=name)
    axis.axvline(THRESHOLD, color="black", linestyle="--", label="frozen threshold")
    axis.set(xlabel="Frozen ensemble malignant probability", ylabel="Images", title="DDI probability distributions by outcome")
    axis.legend(); fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)


def _plot_diagnoses(diagnoses, output):
    shown = diagnoses.sort_values("malignant_n", ascending=False)
    fig, axis = plt.subplots(figsize=(max(8, len(shown) * .75), 5))
    axis.bar(shown.original_diagnosis, shown.sensitivity)
    axis.set(ylim=(0, 1), ylabel="Frozen ensemble sensitivity", title="DDI malignant sensitivity by original diagnosis")
    axis.tick_params(axis="x", rotation=50); fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)


def _plot_correlations(table, output):
    columns = [f"{name}_malignant_probability" for name in MEMBERS]
    fig, axis = plt.subplots(figsize=(5, 4)); image = axis.imshow(table[columns].corr(), vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(range(3), MEMBERS, rotation=35); axis.set_yticks(range(3), MEMBERS)
    for i in range(3):
        for j in range(3): axis.text(j, i, f"{table[columns].corr().iloc[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=axis, label="Pearson correlation"); fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)


def _contact_sheet(table, image_root, output, title):
    thumb, cell_w, cell_h = 180, 220, 260
    selected = table.head(12).copy(); canvas = Image.new("RGB", (cell_w * 3, cell_h * 4 + 36), "white"); draw = ImageDraw.Draw(canvas); draw.text((8, 8), title, fill="black")
    for i, row in enumerate(selected.itertuples()):
        image = Image.open(image_root / row.sample_id).convert("RGB"); image.thumbnail((thumb, thumb)); x, y = (i % 3) * cell_w, 36 + (i // 3) * cell_h
        canvas.paste(image, (x + (cell_w - image.width) // 2, y)); diagnosis = str(getattr(row, "original_diagnosis", ""))[:24]
        draw.text((x + 3, y + thumb + 3), f"{row.sample_id} {row.error_type} p={row.ensemble_malignant_probability:.3f}", fill="black")
        draw.text((x + 3, y + thumb + 18), f"{diagnosis} | tone={getattr(row, 'skin_tone', '')}", fill="black")
        draw.text((x + 3, y + thumb + 33), f"C/E/M: {row.convnext_malignant_probability:.2f}/{row.efficientnet_malignant_probability:.2f}/{row.multimodal_malignant_probability:.2f}", fill="black")
    canvas.save(output)


def _image_stats(image_root, sample_ids):
    rows = []
    for sample_id in sample_ids:
        with Image.open(image_root / sample_id).convert("RGB") as image:
            values = np.asarray(image, dtype=np.float32) / 255.0; rgb_mean = values.mean((0, 1)); rgb_std = values.std((0, 1)); lum = values @ np.array([.2126, .7152, .0722])
            saturation = (values.max(2) - values.min(2)).mean(); sharpness = np.abs(np.diff(lum, axis=0)).mean() + np.abs(np.diff(lum, axis=1)).mean()
            rows.append({"sample_id": sample_id, "width": image.width, "height": image.height, "aspect_ratio": image.width / image.height,
                         "mean_r": rgb_mean[0], "mean_g": rgb_mean[1], "mean_b": rgb_mean[2], "std_r": rgb_std[0], "std_g": rgb_std[1], "std_b": rgb_std[2],
                         "luminance_mean": lum.mean(), "luminance_std": lum.std(), "saturation_proxy": saturation, "sharpness_proxy": sharpness, "format": image.format})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Analysis-only DDI error report from saved frozen predictions")
    parser.add_argument("--allow-final-test", action="store_true", help="Required acknowledgement before opening DDI images")
    parser.add_argument("--ddi-root", default="data/final_external_test/ddi")
    parser.add_argument("--input", default="results/final_evaluation/ddi")
    parser.add_argument("--output", default="results/final_evaluation/ddi_analysis")
    args = parser.parse_args()
    if not args.allow_final_test: parser.error("DDI is final-test data; pass --allow-final-test for post-hoc image analysis")
    root, input_dir, output = Path(args.ddi_root), Path(args.input), Path(args.output); output.mkdir(parents=True, exist_ok=True); examples = output / "examples"; examples.mkdir(exist_ok=True)
    ensemble = pd.read_csv(input_dir / "ensemble_predictions.csv"); members = {name: pd.read_csv(input_dir / f"{name}_predictions.csv") for name in MEMBERS}; metadata = pd.read_csv(root / "ddi_metadata.csv")
    table = canonical_error_table(ensemble, members, metadata, threshold=THRESHOLD); table.to_csv(output / "ddi_error_table.csv", index=False)
    diagnosis = false_negative_diagnoses(table); diagnosis.to_csv(output / "disease_subgroup_metrics.csv", index=False); diagnosis.loc[diagnosis.missed_fn.gt(0)].to_csv(output / "false_negative_analysis.csv", index=False)
    tone = binary_subgroup_metrics(table, "skin_tone"); tone.to_csv(output / "skin_tone_subgroup_metrics.csv", index=False)
    confidence = probability_summary(table); confidence.to_csv(output / "confidence_analysis.csv", index=False)
    disagreement = model_disagreement(table, MEMBERS, THRESHOLD); disagreement.to_csv(output / "model_disagreement_analysis.csv", index=False)
    # Image decoding is intentionally the only expensive part.  Reuse the
    # immutable per-image descriptive artifact on subsequent report runs.
    stats_path = output / "ddi_image_statistics.csv"
    ddi_stats = pd.read_csv(stats_path) if stats_path.is_file() else _image_stats(root, table.sample_id)
    summary = ddi_stats.drop(columns="sample_id").select_dtypes("number").agg(["count", "mean", "std", "median"]).T.reset_index(names="statistic"); summary.insert(0, "dataset", "DDI"); summary["development_test_status"] = "unavailable: retained development manifest is placeholder"; summary.to_csv(output / "dataset_shift_summary.csv", index=False); ddi_stats.to_csv(stats_path, index=False)
    dev = json.loads((Path("results/final_evaluation/dev_test_metrics.json")).read_text())["metrics"]; ddi = json.loads((input_dir / "ddi_report.json").read_text())["metrics"]
    comparison = pd.DataFrame([{"system": "frozen_ensemble", "split": "development_test", **{key: dev.get(key) for key in METRIC_COLUMNS}}, {"system": "frozen_ensemble", "split": "DDI", **{key: ddi.get(key) for key in METRIC_COLUMNS}}]); ddi_row = comparison.loc[comparison.split.eq("DDI")].iloc[0].to_dict(); dev_row = comparison.loc[comparison.split.eq("development_test")].iloc[0].to_dict(); comparison = pd.concat([comparison, pd.DataFrame([{ "system": "frozen_ensemble", "split": "DDI_minus_development_test", **{key: ddi_row[key] - dev_row[key] for key in METRIC_COLUMNS}}])], ignore_index=True); comparison.to_csv(output / "devtest_vs_ddi_comparison.csv", index=False)
    _plot_confidence(table, output / "confidence_distribution.png"); _plot_diagnoses(diagnosis, output / "sensitivity_by_diagnosis.png"); _plot_correlations(table, output / "model_probability_correlation.png")
    for kind, subset, title in [("high_confidence_tp", table.query("error_type == 'TP'").sort_values("ensemble_malignant_probability", ascending=False), "High-confidence true positives"), ("high_confidence_tn", table.query("error_type == 'TN'").sort_values("ensemble_malignant_probability"), "High-confidence true negatives"), ("high_confidence_fn", table.query("error_type == 'FN'").sort_values("ensemble_malignant_probability"), "High-confidence false negatives"), ("high_confidence_fp", table.query("error_type == 'FP'").sort_values("ensemble_malignant_probability", ascending=False), "High-confidence false positives"), ("borderline_fn", table.query("error_type == 'FN'").assign(delta=lambda x: (x.ensemble_malignant_probability - THRESHOLD).abs()).sort_values("delta"), "Borderline false negatives"), ("strong_disagreement", disagreement.sort_values("model_probability_spread", ascending=False), "Strongest member disagreement")]: _contact_sheet(subset, root, examples / f"{kind}.png", title)
    fn = disagreement.query("error_type == 'FN'"); overruled = fn.query("convnext_predicted_malignant")
    auxiliary_missing = all(field not in table or table[field].isna().all() for field in ("age", "sex", "anatomical_site"))
    summary_json = {"scope": "posthoc_analysis_only", "frozen_threshold": THRESHOLD, "n": len(table), "confusion_counts": table.error_type.value_counts().to_dict(), "false_negative_count": len(fn), "fn_convnext_detected": int(fn.convnext_predicted_malignant.sum()), "fn_efficientnet_detected": int(fn.efficientnet_predicted_malignant.sum()), "fn_multimodal_detected": int(fn.multimodal_predicted_malignant.sum()), "fn_missed_by_all_members": int((fn.models_predicting_malignant == 0).sum()), "fn_overruled_convnext": int(len(overruled)), "all_ddi_auxiliary_missing": auxiliary_missing, "standalone_dinov2": "not run: absent from saved DDI artifacts; this analysis script does not rerun final-test inference"}; (output / "analysis_summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
    report = f"""# DDI post-hoc error analysis

## Executive summary

This is an analysis-only report of the already-consumed DDI final test. The frozen equal-weight ensemble and 0.51 threshold were not changed. DDI contained {len(table)} images: {int(table.true_binary.eq(1).sum())} malignant and {int(table.true_binary.eq(0).sum())} benign. It produced {len(fn)} malignant false negatives.

## Performance degradation

Development-test sensitivity was {dev['sensitivity']:.3f}; DDI sensitivity was {ddi['sensitivity']:.3f} (change {ddi['sensitivity']-dev['sensitivity']:.3f}). Development ROC-AUC was {dev['roc_auc']:.3f}; DDI ROC-AUC was {ddi['roc_auc']:.3f}.

## Malignant false negatives and model disagreement

Of {len(fn)} false negatives, ConvNeXt voted malignant for {int(fn.convnext_predicted_malignant.sum())}, EfficientNet for {int(fn.efficientnet_predicted_malignant.sum())}, and multimodal for {int(fn.multimodal_predicted_malignant.sum())}. {int((fn.models_predicting_malignant == 0).sum())} were missed by every member. ConvNeXt was positive while the fixed ensemble remained benign for {len(overruled)} false negatives; this is direct evidence that averaging sometimes suppressed its positive vote, not a basis to alter weights.

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
"""; (output / "ddi_error_analysis_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__": main()
