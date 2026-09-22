# v2 adaptive ensemble weighting search

Validation-only repeated stratified 5-fold CV (3 repeats, seed 42) evaluated fixed neural-network prediction rows. DDI/external-test files were not read.

## Selection
Three-model policy: `{'strategy': 'closest_pair_consensus', 'pair_max_distance': 0.1, 'separation_ratio': 3.0, 'outlier_weight_multiplier': 0.2, 'weights': {'convnext': 0.3427816148634098, 'efficientnet': 0.32914912409374214, 'multimodal': 0.3280692610428481}}`.
Two-model policy: `{'strategy': 'equal_probability_average'}`.
Threshold fixed analysis at 0.51: {'accuracy': 0.8862512363996043, 'balanced_accuracy': 0.8409775465498357, 'macro_f1': 0.8490294331900218, 'sensitivity': 0.9357429718875502, 'specificity': 0.7462121212121212, 'roc_auc': 0.9044257839438562, 'pr_auc': 0.9548487573584474, 'brier_score': 0.09253964762046686}; tuned analysis: {'accuracy': 0.884272997032641, 'balanced_accuracy': 0.8420880491663625, 'macro_f1': 0.8475889216021029, 'sensitivity': 0.9303882195448461, 'specificity': 0.7537878787878788, 'roc_auc': 0.9044257839438562, 'pr_auc': 0.9548487573584474, 'brier_score': 0.09253964762046686}; deployed threshold: 0.51.

## Candidate theory
- Equal averaging: transparent baseline.
- Static validation weighting: normalized per-model validation balanced accuracy.
- Robust MAD/Huber: median-centered influence reduction for large deviations.
- Closest-pair consensus: downweights one member only when a close pair and separation-ratio gates are both satisfied.
- Logistic stacking was intentionally not deployed: it needs fitted coefficients and an out-of-fold serving contract; the simple policies were the production comparison.

## Example behavior
{"probabilities": [0.86, 0.92, 0.05], "weights": {"convnext": 0.4647605296943856, "efficientnet": 0.44627691401479536, "multimodal": 0.08896255629081903}, "ensemble_probability": 0.8147169442453243}
{"probabilities": [0.2, 0.4, 0.65], "weights": {"convnext": 0.3427816148634098, "efficientnet": 0.32914912409374214, "multimodal": 0.3280692610428481}, "ensemble_probability": 0.4134609922880301}

See strategy_comparison.csv and cross_validation_results.csv for every fixed-threshold fold result.
