"""Store evaluation artifacts. / Armazena artefatos da avaliação."""

import csv
import json
from pathlib import Path
import numpy as np

from .metrics import binary_metrics


def save_evaluation(path: str | Path, *, sample_ids, y_true, y_pred, source: str, unit: str, threshold: float | None = None, scores=None) -> dict:
    """Save predictions and metrics once. / Salva uma vez predições e métricas."""
    metrics = binary_metrics(y_true, y_pred)
    ids = list(sample_ids)
    if len(ids) != metrics["total"] or any(not isinstance(item, str) or not item.strip() for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("sample_ids must contain one unique nonempty string per observation.")
    if not isinstance(source, str) or not source.strip() or not isinstance(unit, str) or not unit.strip():
        raise ValueError("source and unit must be nonempty strings.")
    if threshold is not None and (not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("threshold must be within [0, 1].")
    if scores is not None:
        scores = np.asarray(scores, dtype=float)
        if scores.shape != (len(ids),) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError("scores must contain one probability per observation.")
        if threshold is not None and not np.array_equal((scores >= threshold).astype(int), np.asarray(y_pred)):
            raise ValueError("Predictions do not match scores and threshold.")
    target = Path(path)
    target.mkdir(parents=True, exist_ok=False)
    metadata = {"source": source, "unit": unit, "threshold": threshold, "metrics": metrics}
    with (target / "evaluation.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    with (target / "predictions.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "true_label", "predicted_label", "score"])
        for index, sample_id in enumerate(ids):
            writer.writerow([sample_id, int(y_true[index]), int(y_pred[index]), "" if scores is None else float(scores[index])])
    return metadata
