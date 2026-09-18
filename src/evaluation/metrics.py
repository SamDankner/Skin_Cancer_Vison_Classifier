"""Classification metric calculations."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, average_precision_score

def classification_metrics(targets, predictions, probabilities=None) -> dict:
    """Return a common classification metric contract; supports binary tasks."""
    y_true, y_pred = np.asarray(targets), np.asarray(predictions)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    result = {"accuracy": float(accuracy_score(y_true, y_pred)), "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)), "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()), "macro_f1": float(f1.mean()), "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(), "class_metrics": [{"class": int(label), "precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)} for label, p, r, f, s in zip(np.unique(y_true), precision, recall, f1, support)]}
    if len(np.unique(y_true)) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel(); result.update({"sensitivity": tp / (tp + fn) if tp + fn else None, "specificity": tn / (tn + fp) if tn + fp else None})
        if probabilities is not None and len(np.unique(y_true)) == 2:
            scores = np.asarray(probabilities); scores = scores[:, 1] if scores.ndim == 2 else scores
            result["roc_auc"] = float(roc_auc_score(y_true, scores)); result["pr_auc"] = float(average_precision_score(y_true, scores))
    return result

def lesion_presence_metrics(targets, predictions, probabilities=None) -> dict:
    """Metrics for lesion=1 versus normal-skin=0, including normal-skin FPR."""
    result = classification_metrics(targets, predictions, probabilities)
    matrix = np.asarray(result["confusion_matrix"])
    if matrix.shape != (2, 2): raise ValueError("Lesion-presence metrics require both lesion and normal-skin classes")
    tn, fp, fn, tp = matrix.ravel(); result.update({"lesion_sensitivity": tp / (tp + fn) if tp + fn else None, "normal_skin_specificity": tn / (tn + fp) if tn + fp else None, "normal_skin_false_positive_rate": fp / (fp + tn) if fp + tn else None, "precision": tp / (tp + fp) if tp + fp else 0., "recall": tp / (tp + fn) if tp + fn else 0., "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.})
    return result
