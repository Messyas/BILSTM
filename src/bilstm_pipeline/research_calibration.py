"""Calibrate and freeze the decision threshold on validation. / Calibra e congela o limiar na validação."""

from datetime import datetime, timezone
import hashlib
import json

import numpy as np

from .execution import write_json
from .hardware import hardware_metadata
from .metrics import binary_metrics
from .research_config import resolve_project_path
from .research_data import _hash_file, load_manifest, video_path
from .research_evaluation import THRESHOLD_FILENAME, _collect_scores, _load_run
from .results import save_evaluation

RULE = (
    "Smallest-margin threshold with validation FPR <= target_fpr (maximizes validation recall under the FPR cap); "
    "placed halfway between the highest rejected negative score and the next higher validation score; positive when score >= threshold."
)


def select_threshold(labels, scores, target_fpr: float) -> dict:
    """Choose the threshold for a validation FPR cap. / Escolhe o limiar para um teto de FPR na validação."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError("labels and scores must be one-dimensional and aligned.")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Calibration requires both classes.")
    if not np.isfinite(scores).all() or scores.min() < 0 or scores.max() > 1:
        raise ValueError("Scores must be probabilities in [0, 1].")
    if not 0 <= target_fpr < 1:
        raise ValueError("target_fpr must be in [0, 1).")
    negatives = np.sort(scores[labels == 0])[::-1]
    allowed = int(np.floor(target_fpr * len(negatives) + 1e-9))
    boundary = float(negatives[allowed])
    higher = scores[scores > boundary]
    if higher.size:
        threshold = (boundary + float(higher.min())) / 2
        if not threshold > boundary:
            threshold = float(higher.min())
    else:
        threshold = min(1.0, float(np.nextafter(boundary, 2.0)))
    predictions = (scores >= threshold).astype(int)
    false_positives = int(((predictions == 1) & (labels == 0)).sum())
    true_positives = int(((predictions == 1) & (labels == 1)).sum())
    fpr = false_positives / len(negatives)
    return {
        "threshold": float(threshold),
        "target_fpr": float(target_fpr),
        "target_attained": bool(fpr <= target_fpr + 1e-12),
        "allowed_false_positives": allowed,
        "validation_negatives": int(len(negatives)),
        "validation_positives": int((labels == 1).sum()),
        "validation_false_positives": false_positives,
        "validation_true_positives": true_positives,
        "validation_fpr": fpr,
        "validation_recall": true_positives / int((labels == 1).sum()),
        "tied_scores_at_threshold": int((scores == threshold).sum()),
    }


def calibrate_threshold(run_id: str, *, target_fpr: float | None = None, dataset_root=None, device: str | None = None) -> str:
    """Score validation, freeze the threshold and write threshold.json. / Pontua a validação e grava threshold.json."""
    run_dir, config, selected, model = _load_run(run_id, device)
    target_file = run_dir / THRESHOLD_FILENAME
    if target_file.exists():
        raise FileExistsError(f"Threshold already frozen for this run: {target_file}. It must not be recalibrated.")
    target_fpr = config["evaluation"]["target_fpr"] if target_fpr is None else target_fpr
    manifest = load_manifest(run_dir / "dataset_manifest.json")
    records = [record for record in manifest["records"] if manifest["assignments"][record["sample_id"]] == "validation"]
    root = resolve_project_path(dataset_root or config["data"]["root"])
    for record in records:
        video_path(root, record, verify_hash=True)
    data = _collect_scores(records, root, config, selected, model, False, None)
    selection = select_threshold(data["labels"], data["scores"], target_fpr)
    predictions = (np.asarray(data["scores"]) >= selection["threshold"]).astype(int)
    save_evaluation(run_dir / "calibration", sample_ids=data["ids"], y_true=data["labels"], y_pred=predictions, scores=data["scores"], threshold=selection["threshold"], source=manifest["source"], unit="one video; one uniformly sampled sequence (validation split)")
    reference = binary_metrics(data["labels"], (np.asarray(data["scores"]) >= 0.5).astype(int))
    ids_digest = hashlib.sha256("\n".join(sorted(data["ids"])).encode()).hexdigest()
    payload = {
        **selection,
        "rule": RULE,
        "split": "validation",
        "dataset": manifest["dataset_name"],
        "validation_ids_sha256": ids_digest,
        "checkpoint_sha256": _hash_file(run_dir / "best.pt"),
        "reference_metrics_at_0_5": reference,
        "note": "The validation split also selected best.pt (EarlyStopping/ReduceLROnPlateau), so FPR on unseen data may exceed the target.",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware_metadata(selected),
    }
    with target_file.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    write_json(run_dir / "calibration" / "threshold_copy.json", payload)
    return str(target_file)
