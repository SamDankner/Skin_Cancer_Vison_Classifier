"""Shared evaluation interfaces."""

from .calibration import TemperatureCalibration, fit_temperature, select_binary_threshold
from .ensemble import EnsembleMember, equal_weight_ensemble, fit_validation_weights
from .evaluator import (
    PREDICTION_COLUMNS,
    build_evaluation_loader,
    evaluate_frozen_external_test,
    evaluate_model,
    export_predictions,
    freeze_final_configuration,
    load_checkpoint_bundle,
)
from .metrics import classification_metrics, lesion_presence_metrics

__all__ = [
    "EnsembleMember", "PREDICTION_COLUMNS", "TemperatureCalibration",
    "build_evaluation_loader", "classification_metrics", "equal_weight_ensemble",
    "evaluate_frozen_external_test", "evaluate_model",
    "export_predictions", "fit_temperature", "fit_validation_weights",
    "freeze_final_configuration", "lesion_presence_metrics",
    "load_checkpoint_bundle", "select_binary_threshold",
]
