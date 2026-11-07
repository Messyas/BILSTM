"""Binary evaluation and grouped uncertainty. / Avaliação binária e incerteza por grupos."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _labels(values: ArrayLike, name: str) -> NDArray[np.int_]:
    array = np.asarray(values)
    if array.ndim != 1 or not len(array) or not np.isin(array, [0, 1]).all():
        raise ValueError(f"{name} must be a nonempty one-dimensional sequence of 0 and 1.")
    return array.astype(int)


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def binary_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict:
    """Evaluate labels 0/1. / Avalia rótulos 0/1; 1 é a classe positiva."""
    true, pred = _labels(y_true, "y_true"), _labels(y_pred, "y_pred")
    if len(true) != len(pred):
        raise ValueError("y_true and y_pred must have the same length.")
    tn = int(np.sum((true == 0) & (pred == 0)))
    fp = int(np.sum((true == 0) & (pred == 1)))
    fn = int(np.sum((true == 1) & (pred == 0)))
    tp = int(np.sum((true == 1) & (pred == 1)))

    def class_metrics(class_tp: int, class_fp: int, class_fn: int, support: int) -> dict:
        return {
            "precision": _ratio(class_tp, class_tp + class_fp),
            "recall": _ratio(class_tp, class_tp + class_fn),
            "f1": _ratio(2 * class_tp, 2 * class_tp + class_fp + class_fn),
            "support": support,
        }

    classes = {"0": class_metrics(tn, fn, fp, tn + fp), "1": class_metrics(tp, fp, fn, tp + fn)}
    f1_values = [classes["0"]["f1"], classes["1"]["f1"]]
    macro = sum(f1_values) / 2 if all(value is not None for value in f1_values) else None
    weighted = sum(item["f1"] * item["support"] for item in classes.values() if item["f1"] is not None) / len(true)
    return {
        "confusion_matrix": [[tn, fp], [fn, tp]], "positive_class": 1,
        "total": len(true), "accuracy": _ratio(tn + tp, len(true)),
        "specificity": _ratio(tn, tn + fp), "false_positive_rate": _ratio(fp, tn + fp),
        "classes": classes, "f1_macro": macro, "f1_weighted": float(weighted),
    }


def _metric_value(metrics: dict, name: str) -> float | None:
    supported = {"accuracy", "specificity", "false_positive_rate", "f1_macro", "f1_weighted", "positive_precision", "positive_recall", "positive_f1"}
    if name not in supported:
        raise ValueError(f"Unsupported metric: {name}. Choose from {sorted(supported)}.")
    if name.startswith("positive_"):
        return metrics["classes"]["1"][name.removeprefix("positive_")]
    return metrics[name]


def _validate_bootstrap_groups(true, pred, groups, paired_pred):
    """Validate inputs and groups. / Valida entradas e grupos."""
    group_array = np.asarray(groups)
    if group_array.ndim != 1 or len(true) != len(pred) or len(true) != len(group_array):
        raise ValueError("Labels, predictions and groups must have equal lengths.")
    if any(not isinstance(group, str) or not group.strip() for group in group_array):
        raise ValueError("Groups must be nonempty strings.")
    other = None if paired_pred is None else _labels(paired_pred, "paired_pred")
    if other is not None and len(other) != len(true):
        raise ValueError("paired_pred must have the same length as y_true.")
    unique = np.unique(group_array)
    if len(unique) < 2:
        raise ValueError("At least two independent groups are required.")
    return group_array, unique, other


def _validate_bootstrap_options(n_resamples, seed, confidence):
    """Validate resampling options. / Valida opções de reamostragem."""
    if type(n_resamples) is not int or n_resamples < 2 or not 0 < confidence < 1:
        raise ValueError("n_resamples must be at least 2 and confidence must be between 0 and 1.")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")


def _comparison_value(true, pred, other, metric):
    """Compute a metric or paired difference. / Calcula métrica ou diferença pareada."""
    value = _metric_value(binary_metrics(true, pred), metric)
    if value is None or other is None:
        return value
    value_b = _metric_value(binary_metrics(true, other), metric)
    if value_b is None:
        return None
    return value_b - value


def _bootstrap_samples(true, pred, other, indices, metric, n_resamples, seed):
    """Resample groups as units. / Reamostra grupos como unidades."""
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_resamples):
        chosen = rng.integers(0, len(indices), size=len(indices))
        selected = np.concatenate([indices[index] for index in chosen])
        other_selected = other[selected] if other is not None else None
        value = _comparison_value(true[selected], pred[selected], other_selected, metric)
        if value is not None:
            samples.append(value)
    return samples


def grouped_bootstrap(y_true, y_pred, groups, *, metric="accuracy", n_resamples=1000, seed=42, confidence=0.95, paired_pred=None) -> dict:
    """Resample whole groups; paired result is B minus A. / Reamostra grupos; comparação pareada é B menos A."""
    true, pred = _labels(y_true, "y_true"), _labels(y_pred, "y_pred")
    group_array, unique, other = _validate_bootstrap_groups(true, pred, groups, paired_pred)
    _validate_bootstrap_options(n_resamples, seed, confidence)
    indices = [np.flatnonzero(group_array == group) for group in unique]
    estimate = _comparison_value(true, pred, other, metric)
    samples = _bootstrap_samples(true, pred, other, indices, metric, n_resamples, seed)
    if len(samples) < 2:
        raise ValueError("Fewer than two valid bootstrap samples; the metric cannot be estimated.")
    alpha = (1 - confidence) * 50
    lower, upper = np.percentile(samples, [alpha, 100 - alpha])
    return {
        "metric": metric, "estimate": estimate, "lower": float(lower), "upper": float(upper),
        "confidence": confidence, "method": "percentile grouped bootstrap", "groups": len(unique),
        "resamples": n_resamples, "valid_resamples": len(samples), "skipped_resamples": n_resamples - len(samples),
        "seed": seed, "comparison": "B - A" if other is not None else None,
    }
