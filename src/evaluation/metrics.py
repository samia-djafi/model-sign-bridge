"""
Evaluation metrics module for Algerian Sign Language recognition.
"""

import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None,
    latencies_ms: Optional[List[float]] = None,
) -> Dict:
    """
    Compute comprehensive multi-class evaluation metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred)

    # Per-class metrics
    per_class_p = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_r = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))

    per_class_dict = {}
    for i, lbl_id in enumerate(unique_labels):
        name = target_names[lbl_id] if target_names and lbl_id < len(target_names) else str(lbl_id)
        support = int(np.sum(y_true == lbl_id))
        per_class_dict[name] = {
            "class_id": int(lbl_id),
            "precision": float(per_class_p[i]) if i < len(per_class_p) else 0.0,
            "recall": float(per_class_r[i]) if i < len(per_class_r) else 0.0,
            "f1": float(per_class_f1[i]) if i < len(per_class_f1) else 0.0,
            "support": support,
        }

    metrics = {
        "num_samples": int(len(y_true)),
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class_dict,
        "mean_latency_ms": float(np.mean(latencies_ms)) if latencies_ms else None,
    }
    return metrics
