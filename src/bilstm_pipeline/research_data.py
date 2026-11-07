"""Prepare RLVS records and video sequences. / Prepara registros e sequências do RLVS."""

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


def prepare_dataset(config_path: str | Path, *, dataset_root: str | Path | None = None, manifest_path: str | Path | None = None, dataset_name: str = "RLVS", source: str = "mohamedmustafa/real-life-violence-situations-dataset") -> Path:
    """Build a stratified grouped manifest. / Cria manifesto estratificado por grupos."""
    config = load_research_config(config_path)
    root = resolve_project_path(dataset_root or config["data"]["root"])
    target = resolve_project_path(manifest_path or config["data"]["manifest"])
    if target.exists():
        raise FileExistsError(f"Manifest already exists: {target}")
    records = inventory_videos(root)
    groups = {}
    for record in records:
        group, label = record["group_id"], record["label"]
        if group in groups and groups[group] != label:
            raise ValueError("Identical videos have conflicting class labels.")
        groups[group] = label
    fractions = tuple(config["split"][name] for name in ("train", "validation", "test"))
    assignments = {}
    for label in (0, 1):
        class_records = [record for record in records if record["label"] == label]
        if len({record["group_id"] for record in class_records}) < 3:
            raise ValueError(f"At least three independent video groups are required for class {label}.")
        assignments.update(group_split(class_records, proportions=fractions, seed=config["split"]["seed"] + label))
    summary = validate_split(records, assignments)
    for name, item in summary.items():
        if not item["labels"]["0"] or not item["labels"]["1"]:
            raise ValueError(f"Split {name} lacks one of the classes.")
    target.parent.mkdir(parents=True, exist_ok=True)
    for record in records:
        record["source"] = source
    download_metadata = None
    download_file = root / "download_metadata.json"
    if download_file.is_file():
        download = json.loads(download_file.read_text(encoding="utf-8"))
        download_metadata = {key: download.get(key) for key in ("dataset", "version", "archive_sha256")}
    save_manifest(target, records, dataset_name=dataset_name, source=source, license_name="Verify source dataset license and original authors", assignments=assignments, split_metadata={"algorithm": "stratified_by_label_group_sha256", "seed": config["split"]["seed"], "proportions": list(fractions), "download": download_metadata, "independence_limit": "Exact hashes only; source scene and near-duplicates remain unverified."})
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


def decode_video(path: Path, *, frames: int, width: int, height: int) -> np.ndarray:
    """Uniformly sample decoded video frames. / Amostra quadros uniformemente do vídeo decodificado."""
    import cv2
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Cannot open video: {path}")
    images = []
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA))
    finally:
        capture.release()
    if not images:
        raise ValueError(f"Video contains no decodable frames: {path}")
    indices = np.linspace(0, len(images) - 1, frames).astype(int)
    return np.stack([images[index] for index in indices]).astype(np.float32) / 255.0
