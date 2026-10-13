"""Evaluation helpers that refuse to score unreviewed label mappings."""

import json
from pathlib import Path

import numpy as np


def load_mapping(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "reviewed":
        raise ValueError("Taxonomy mapping is not marked reviewed; metrics would be scientifically invalid.")
    return payload["mapping"]


def multilabel_metrics(scores: np.ndarray, targets: np.ndarray, threshold: float = 0.50) -> dict:
    """Return sklearn-free thresholded multilabel metrics for saved outputs."""
    if scores.shape != targets.shape or scores.ndim != 2:
        raise ValueError("scores and targets must be equally shaped [samples, labels] arrays")
    predicted = scores >= threshold
    true_positive = np.logical_and(predicted, targets).sum(axis=0)
    false_positive = np.logical_and(predicted, ~targets.astype(bool)).sum(axis=0)
    false_negative = np.logical_and(~predicted, targets).sum(axis=0)
    precision = np.divide(true_positive, true_positive + false_positive, out=np.zeros_like(true_positive, dtype=float), where=(true_positive + false_positive) != 0)
    recall = np.divide(true_positive, true_positive + false_negative, out=np.zeros_like(true_positive, dtype=float), where=(true_positive + false_negative) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) != 0)
    tp, fp, fn = true_positive.sum(), false_positive.sum(), false_negative.sum()
    micro_precision = tp / (tp + fp) if tp + fp else 0.0
    micro_recall = tp / (tp + fn) if tp + fn else 0.0
    average_precision = [_average_precision(scores[:, i], targets[:, i].astype(bool)) for i in range(scores.shape[1])]
    defined_ap = [value for value in average_precision if value is not None]
    return {"threshold": threshold, "macro_f1": float(f1.mean()), "micro_f1": float(2 * micro_precision * micro_recall / (micro_precision + micro_recall)) if micro_precision + micro_recall else 0.0, "macro_average_precision": float(np.mean(defined_ap)) if defined_ap else None, "micro_average_precision": _average_precision(scores.ravel(), targets.astype(bool).ravel()), "per_class_precision": precision.tolist(), "per_class_recall": recall.tolist(), "per_class_f1": f1.tolist(), "per_class_average_precision": average_precision, "support": targets.sum(axis=0).astype(int).tolist()}


def _average_precision(scores: np.ndarray, targets: np.ndarray) -> float | None:
    """Non-interpolated AP; undefined rather than zero when no positives exist."""
    positives = int(targets.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    ranked_true = targets[order]
    precision_at_rank = np.cumsum(ranked_true) / np.arange(1, len(ranked_true) + 1)
    return float(precision_at_rank[ranked_true].sum() / positives)
