import pandas as pd

from src.evaluation.ddi_analysis import canonical_error_table, false_negative_diagnoses, model_disagreement


def _prediction(ids, labels, probs):
    return pd.DataFrame({"sample_id": ids, "true_label": labels, "predicted_label": ["malignant" if p >= .51 else "benign" for p in probs], "class_probabilities": [f'{{"benign": {1-p}, "malignant": {p}}}' for p in probs]})


def test_canonical_table_preserves_fixed_predictions_and_votes():
    ids, labels = ["a.png", "b.png"], ["malignant", "benign"]
    ensemble = _prediction(ids, labels, [.4, .7])
    members = {"convnext": _prediction(ids, labels, [.8, .6]), "efficientnet": _prediction(ids, labels, [.2, .8]), "multimodal": _prediction(ids, labels, [.2, .7])}
    meta = pd.DataFrame({"DDI_file": ids, "DDI_ID": [1, 2], "disease": ["x", "y"], "skin_tone": [1, 2]})
    result = canonical_error_table(ensemble, members, meta, threshold=.51)
    assert result.error_type.tolist() == ["FN", "FP"]
    assert result.models_predicting_malignant.tolist() == [1, 3]
    disagreement = model_disagreement(result, members, .51)
    assert disagreement.ensemble_overruled_member_positive.tolist() == [True, False]
    assert false_negative_diagnoses(result).loc[0, "missed_fn"] == 1
