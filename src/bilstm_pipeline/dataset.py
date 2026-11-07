"""Dataset manifests and group splits. / Manifestos de dados e divisão por grupos."""

from collections.abc import Sequence
import json
from pathlib import Path
import re
import numpy as np

SPLITS = ("train", "validation", "test")


def _validate_record_fields(record: dict) -> None:
    """Validate required fields. / Valida campos obrigatórios."""
    if not isinstance(record, dict):
        raise ValueError("Each record must be a dictionary.")
    for field in ("sample_id", "video_id", "group_id", "source"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError(f"{field} must be a nonempty string.")
    label = record.get("label")
    if label is not None and (type(label) is not int or label not in (0, 1)):
        raise ValueError("label must be 0, 1 or None.")


def _track_record(record, sample_ids, video_groups, hash_groups):
    """Check relationships and duplicates. / Verifica relações e duplicatas."""
    sample_id, video_id, group_id = record["sample_id"], record["video_id"], record["group_id"]
    if sample_id in sample_ids:
        raise ValueError(f"Duplicate sample_id: {sample_id}.")
    sample_ids.add(sample_id)
    if video_id in video_groups and video_groups[video_id] != group_id:
        raise ValueError(f"Video {video_id} belongs to multiple groups.")
    video_groups[video_id] = group_id
    content_hash = record.get("sha256")
    if content_hash is None:
        return
    if not isinstance(content_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", content_hash):
        raise ValueError("sha256 must contain 64 hexadecimal characters.")
    content_hash = content_hash.lower()
    if content_hash in hash_groups and hash_groups[content_hash] != group_id:
        raise ValueError("Known duplicate content belongs to different groups.")
    hash_groups[content_hash] = group_id


def validate_records(records: Sequence[dict]) -> None:
    """Check identifiers and known duplicates. / Verifica identificadores e duplicatas conhecidas."""
    if not records:
        raise ValueError("The dataset must contain at least one record.")
    sample_ids, video_groups, hash_groups = set(), {}, {}
    for record in records:
        _validate_record_fields(record)
        _track_record(record, sample_ids, video_groups, hash_groups)


def group_split(records: Sequence[dict], *, proportions=(0.7, 0.15, 0.15), seed=42) -> dict[str, str]:
    """Split whole groups, not stratified. / Divide grupos inteiros, sem estratificação."""
    validate_records(records)
    fractions = np.asarray(proportions, dtype=float)
    if fractions.shape != (3,) or not np.isfinite(fractions).all() or np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1):
        raise ValueError("Provide three positive proportions summing to one.")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    groups = np.array(sorted({record["group_id"] for record in records}))
    if len(groups) < 3:
        raise ValueError("At least three independent groups are required.")
    groups = np.random.default_rng(seed).permutation(groups)
    remaining = len(groups) - 3
    desired = fractions * remaining
    counts = np.floor(desired).astype(int) + 1
    for index in np.argsort(-(desired - np.floor(desired)))[:len(groups) - int(counts.sum())]:
        counts[index] += 1
    group_assignments, offset = {}, 0
    for name, count in zip(SPLITS, counts, strict=True):
        for group in groups[offset:offset + count]:
            group_assignments[str(group)] = name
        offset += count
    assignments = {record["sample_id"]: group_assignments[record["group_id"]] for record in records}
    validate_split(records, assignments)
    return assignments


def _track_split(mapping, field, value, name):
    """Prevent a shared entity crossing splits. / Impede entidade comum entre divisões."""
    if value is None:
        return
    if field == "sha256":
        value = value.lower()
    if value in mapping and mapping[value] != name:
        raise ValueError(f"{field} appears in multiple splits: {value}.")
    mapping[value] = name


def _update_summary(item, record):
    """Count samples, groups and labels. / Conta amostras, grupos e rótulos."""
    item["samples"] += 1
    item["groups"].add(record["group_id"])
    item["videos"].add(record["video_id"])
    label = str(record["label"]) if record.get("label") is not None else "unknown"
    item["labels"][label] += 1


def validate_split(records: Sequence[dict], assignments: dict[str, str]) -> dict:
    """Reject overlap and report counts. / Rejeita vazamento e informa contagens."""
    validate_records(records)
    expected = {record["sample_id"] for record in records}
    if set(assignments) != expected or any(value not in SPLITS for value in assignments.values()):
        raise ValueError("Every sample must have exactly one valid split assignment.")
    group_splits, video_splits, hash_splits = {}, {}, {}
    summary = {name: {"samples": 0, "groups": set(), "videos": set(), "labels": {"0": 0, "1": 0, "unknown": 0}} for name in SPLITS}
    for record in records:
        name = assignments[record["sample_id"]]
        for mapping, field in ((group_splits, "group_id"), (video_splits, "video_id"), (hash_splits, "sha256")):
            _track_split(mapping, field, record.get(field), name)
        _update_summary(summary[name], record)
    for item in summary.values():
        item["groups"] = len(item["groups"])
        item["videos"] = len(item["videos"])
    return summary


def save_manifest(path: str | Path, records: Sequence[dict], *, dataset_name: str, source: str, license_name: str, assignments: dict | None = None, split_metadata: dict | None = None) -> None:
    """Save an explicit manifest. / Salva um manifesto explícito."""
    validate_records(records)
    if any(not isinstance(value, str) or not value.strip() for value in (dataset_name, source, license_name)):
        raise ValueError("Dataset name, source and license must be nonempty strings.")
    summary = validate_split(records, assignments) if assignments is not None else None
    payload = {"dataset_name": dataset_name, "source": source, "license": license_name, "records": list(records), "assignments": assignments, "split_summary": summary, "split_metadata": split_metadata}
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
