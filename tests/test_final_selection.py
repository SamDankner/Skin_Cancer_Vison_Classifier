import numpy as np
import pytest

from src.evaluation.final_selection import align_prediction_collections, binary_metrics, select_threshold, threshold_search


def _collection(ids, targets, probabilities):
    return {"metadata": [{"image_id": item} for item in ids], "targets": np.asarray(targets), "probabilities": np.asarray(probabilities)}


def test_prediction_alignment_reorders_by_stable_id_and_rejects_bad_labels():
    first = _collection(["a", "b"], [0, 1], [[.9, .1], [.2, .8]])
    second = _collection(["b", "a"], [1, 0], [[.3, .7], [.8, .2]])
    aligned = align_prediction_collections({"one": first, "two": second})
    assert aligned["two"]["sample_ids"] == ("a", "b")
    assert np.allclose(aligned["two"]["probabilities"][:, 1], [.2, .7])
    second["targets"] = np.array([0, 0])
    with pytest.raises(ValueError, match="labels disagree"):
        align_prediction_collections({"one": first, "two": second})


def test_threshold_search_is_validation_only_and_selects_macro_f1():
    rows = threshold_search([0, 0, 1, 1], [.1, .4, .6, .9], split="validation", thresholds=[.4, .5, .6])
    assert select_threshold(rows)["threshold"] == .5
    assert binary_metrics([0, 1], [.2, .8], .5)["macro_f1"] == 1.0
    with pytest.raises(ValueError, match="validation"):
        threshold_search([0, 1], [.2, .8], split="test")
