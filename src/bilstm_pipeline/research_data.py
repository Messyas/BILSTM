"""Prepare RLVS records and video sequences. / Prepara registros e sequências do RLVS."""

import csv
import hashlib
import json
from pathlib import Path
import numpy as np

from .dataset import group_split, save_manifest, validate_records, validate_split
from .research_config import load_research_config, resolve_project_path

_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg"}
_CLASS_NAMES = {"nonviolence": 0, "violence": 1}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _label_from_path(path: Path) -> int:
    labels = {_CLASS_NAMES[part.casefold().replace(" ", "")] for part in path.parts if part.casefold().replace(" ", "") in _CLASS_NAMES}
    if len(labels) != 1:
        raise ValueError(f"Cannot uniquely infer class from video path: {path}.")
    return labels.pop()


def inventory_videos(root: str | Path) -> list[dict]:
    """Hash and inventory original videos. / Calcula hashes e inventaria vídeos originais."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.casefold() not in _VIDEO_EXTENSIONS:
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"Video path escapes dataset root: {path}")
        relative = path.relative_to(root)
        digest = _hash_file(path)
        records.append({
            "sample_id": relative.as_posix(), "video_id": relative.as_posix(),
            "group_id": f"sha256-{digest}", "sha256": digest,
            "source": "RLVS", "label": _label_from_path(relative), "path": relative.as_posix(),
        })
    if not records:
        raise ValueError(f"No supported video files found under {root}.")
    validate_records(records)
    return records


def _apply_groups(records: list[dict], groups_path: str | Path) -> None:
    """Apply a reviewed mapping. / Aplica um mapeamento revisado."""
    with Path(groups_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["sample_id", "group_id"]:
            raise ValueError("Group CSV must have exactly the columns sample_id,group_id.")
        groups = {}
        for row in reader:
            sample, group = row.get("sample_id"), row.get("group_id")
            if not sample or not group or not sample.strip() or not group.strip() or None in row:
                raise ValueError("Each group mapping must contain a sample_id and group_id.")
            if sample in groups:
                raise ValueError(f"Duplicate group mapping: {sample}.")
            groups[sample] = group.strip()
    if set(groups) != {record["sample_id"] for record in records}:
        raise ValueError("Group mapping must cover exactly all inventoried sample IDs.")
    for record in records:
        record["group_id"] = groups[record["sample_id"]]
    validate_records(records)


def _validate_group_labels(records: list[dict]) -> None:
    """Ensure each split group has one label. / Garante um rotulo por grupo."""
    labels_by_group = {}
    for record in records:
        group_id, label = record["group_id"], record["label"]
        if group_id in labels_by_group and labels_by_group[group_id] != label:
            raise ValueError("A group contains different class labels; stratified group splitting requires a reviewed alternative protocol.")
        labels_by_group[group_id] = label


def _create_assignments(records: list[dict], fractions: tuple, seed: int) -> dict:
    """Create stratified group assignments. / Cria divisoes estratificadas por grupo."""
    assignments = {}
    for label in (0, 1):
        class_records = [record for record in records if record["label"] == label]
        group_count = len({record["group_id"] for record in class_records})
        if group_count < 3:
            raise ValueError(f"At least three independent video groups are required for class {label}.")
        assignments.update(group_split(class_records, proportions=fractions, seed=seed + label))
    return assignments


def _validate_assignment_summary(records: list[dict], assignments: dict) -> None:
    """Require both classes in every split. / Exige as duas classes em cada divisao."""
    for name, item in validate_split(records, assignments).items():
        if not item["labels"]["0"] or not item["labels"]["1"]:
            raise ValueError(f"Split {name} lacks one of the classes.")


def _read_download_metadata(root: Path) -> dict | None:
    """Read available download provenance. / Le a origem disponivel do download."""
    metadata_file = root / "download_metadata.json"
    if not metadata_file.is_file():
        return None
    payload = json.loads(metadata_file.read_text(encoding="utf-8"))
    return {key: payload.get(key) for key in ("dataset", "version", "archive_sha256")}


def prepare_dataset(config_path: str | Path, *, dataset_root: str | Path | None = None, manifest_path: str | Path | None = None, dataset_name: str = "RLVS", source: str = "mohamedmustafa/real-life-violence-situations-dataset", groups_path: str | Path | None = None) -> Path:
    """Build a stratified grouped manifest. / Cria manifesto estratificado por grupos."""
    config = load_research_config(config_path)
    root = resolve_project_path(dataset_root or config["data"]["root"])
    target = resolve_project_path(manifest_path or config["data"]["manifest"])
    if target.exists():
        raise FileExistsError(f"Manifest already exists: {target}")
    records = inventory_videos(root)
    if groups_path is not None:
        groups_path = resolve_project_path(groups_path)
        _apply_groups(records, groups_path)
    _validate_group_labels(records)
    fractions = tuple(config["split"][name] for name in ("train", "validation", "test"))
    assignments = _create_assignments(records, fractions, config["split"]["seed"])
    _validate_assignment_summary(records, assignments)
    target.parent.mkdir(parents=True, exist_ok=True)
    for record in records:
        record["source"] = source
    split_metadata = {
        "algorithm": "stratified_by_label_group",
        "seed": config["split"]["seed"],
        "proportions": list(fractions),
        "download": _read_download_metadata(root),
        "group_mapping_sha256": _hash_file(groups_path) if groups_path else None,
        "independence_limit": "Exact hashes checked; source-scene and near-duplicate groups require review.",
    }
    save_manifest(target, records, dataset_name=dataset_name, source=source, license_name="Verify source dataset license and original authors", assignments=assignments, split_metadata=split_metadata)
    return target


def load_manifest(path: str | Path) -> dict:
    """Load and validate a prepared manifest. / Lê e valida um manifesto preparado."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_records(payload["records"])
    validate_split(payload["records"], payload["assignments"])
    return payload


def video_path(root: Path, record: dict, *, verify_hash: bool = False) -> Path:
    """Resolve a video within its root. / Resolve um vídeo dentro da raiz."""
    root = root.resolve()
    path = (root / record["path"]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError(f"Invalid or missing video for {record['sample_id']}.")
    if verify_hash and _hash_file(path) != record["sha256"].lower():
        raise ValueError(f"Video content differs from manifest: {record['sample_id']}.")
    return path


def _open_video_capture(path: Path):
    """Open and validate a video capture. / Abre e valida um video."""
    import cv2
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Cannot open video: {path}")
    return capture


def _count_video_frames(path: Path) -> int:
    """Count decodable frames with bounded memory. / Conta quadros com memoria limitada."""
    capture = _open_video_capture(path)
    count = 0
    try:
        while capture.read()[0]:
            count += 1
    finally:
        capture.release()
    return count


def _sample_video_frames(path: Path, indices: np.ndarray, count: int, *, frames: int, width: int, height: int) -> np.ndarray:
    """Decode selected frames in a second pass. / Decodifica quadros selecionados em uma segunda passagem."""
    import cv2
    capture = _open_video_capture(path)
    images = []
    try:
        for index in range(count):
            ok, image = capture.read()
            if not ok:
                raise ValueError(f"Video decoding changed between passes: {path}")
            if len(images) < frames and index == indices[len(images)]:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                while len(images) < frames and index == indices[len(images)]:
                    images.append(image)
    finally:
        capture.release()
    return np.stack(images).astype(np.float32) / 255.0


def decode_video(path: Path, *, frames: int, width: int, height: int) -> np.ndarray:
    """Sample in two passes with bounded memory. / Amostra em duas passagens com memoria limitada."""
    if any(type(value) is not int or value <= 0 for value in (frames, width, height)):
        raise ValueError("frames, width and height must be positive integers.")
    count = _count_video_frames(path)
    if not count:
        raise ValueError(f"Video contains no decodable frames: {path}")
    indices = np.arange(frames, dtype=int) * (count // frames) if count >= frames else np.linspace(0, count - 1, frames).astype(int)
    return _sample_video_frames(path, indices, count, frames=frames, width=width, height=height)
