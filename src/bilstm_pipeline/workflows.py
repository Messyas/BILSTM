"""Reusable file workflows. / Fluxos reutilizáveis de arquivos."""

import csv
import json
from pathlib import Path

from .dataset import group_split, save_manifest, validate_split
from .plots import plot_binary_metrics, plot_confusion_matrix
from .results import save_evaluation


def evaluate_csv(input_path: str | Path, output_dir: str | Path, *, source: str, unit: str, threshold: float | None = None) -> dict:
    """Evaluate a prediction table. / Avalia uma tabela de predições."""
    with Path(input_path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"sample_id", "true_label", "predicted_label"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain columns: {', '.join(sorted(required))}.")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV must contain at least one prediction.")
    scores_raw = [row.get("score", "") or "" for row in rows]
    if any(scores_raw) and not all(scores_raw):
        raise ValueError("The score column must be populated for every row or left empty.")
    scores = [float(value) for value in scores_raw] if all(scores_raw) else None
    result = save_evaluation(
        output_dir, sample_ids=[row["sample_id"] for row in rows],
        y_true=[int(row["true_label"]) for row in rows],
        y_pred=[int(row["predicted_label"]) for row in rows],
        scores=scores, threshold=threshold, source=source, unit=unit,
    )
    plot_confusion_matrix(result["metrics"], Path(output_dir) / "figures")
    plot_binary_metrics(result["metrics"], Path(output_dir) / "figures")
    return result


def split_manifest(input_path: str | Path, output_path: str | Path, *, proportions=(0.7, 0.15, 0.15), seed=42) -> dict:
    """Split a dataset manifest by group. / Divide um manifesto de dados por grupo."""
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    required = {"dataset_name", "source", "license", "records"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(f"Manifest must contain: {', '.join(sorted(required))}.")
    records = payload["records"]
    assignments = group_split(records, proportions=proportions, seed=seed)
    metadata = {"seed": seed, "proportions": list(proportions), "algorithm": "group_split", "stratified": False}
    save_manifest(output_path, records, dataset_name=payload["dataset_name"], source=payload["source"], license_name=payload["license"], assignments=assignments, split_metadata=metadata)
    return validate_split(records, assignments)
