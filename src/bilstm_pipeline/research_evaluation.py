"""Evaluate trained runs. / Avalia execuções treinadas."""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import tomllib
from uuid import uuid4

import numpy as np
import torch

from .execution import run_path, write_json
from .metrics import binary_metrics, grouped_bootstrap
from .plots import plot_binary_metrics, plot_confusion_matrix, plot_robustness_comparison
from .research_config import resolve_project_path, validate_research_config
from .research_data import _hash_file, decode_video, load_manifest, video_path
from .research_model import ConflictModel
from .research_training import select_device, set_seed, video_tensor
from .results import save_evaluation
from .robustness import brightness, contrast, gaussian_noise, jpeg_compression, occlusion

_RUN_MANIFEST_FILENAME = "manifest.json"


def _load_run(run_id: str, device: str | None):
    target = run_path(run_id)
    manifest = json.loads((target / _RUN_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("type") != "proposed_reconstruction":
        raise ValueError("A completed proposed reconstruction run is required.")
    with (target / "effective_config.toml").open("rb") as stream:
        config = validate_research_config(tomllib.load(stream))
    if device is not None:
        if device not in ("cuda", "cpu", "auto"):
            raise ValueError("device must be cuda, cpu or auto.")
        config["training"]["device"] = device
    selected = select_device(config["training"]["device"])
    model_config = dict(config["model"], pretrained=False)
    model = ConflictModel(**model_config).to(selected)
    checkpoint = torch.load(target / "best.pt", map_location=selected, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return target, config, selected, model


def _predict(model, frames, device):
    start_preprocess = time.perf_counter()
    tensor = video_tensor(frames).unsqueeze(0).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    preprocess = time.perf_counter() - start_preprocess
    start_inference = time.perf_counter()
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        score = float(probabilities[0, 1].item())
    return score, preprocess, time.perf_counter() - start_inference


def _intervals(labels, predictions, groups, *, seed, resamples):
    result = {}
    for metric in ("accuracy", "specificity", "false_positive_rate", "positive_precision", "positive_recall", "positive_f1", "negative_precision", "negative_recall", "negative_f1", "f1_macro", "f1_weighted"):
        try:
            result[metric] = grouped_bootstrap(labels, predictions, groups, metric=metric, seed=seed, n_resamples=resamples)
        except ValueError as error:
            result[metric] = {"unavailable": str(error)}
    return result


def _variants(frames, config, seed):
    height, width = frames.shape[1:3]
    fraction = config["occlusion_fraction"]
    mask_height = max(1, min(height, round(height * fraction**0.5)))
    mask_width = max(1, min(width, round(width * fraction**0.5)))
    return {
        "brightness": brightness(frames, config["brightness_delta"]),
        "contrast": contrast(frames, config["contrast_factor"]),
        "noise": gaussian_noise(frames, config["noise_sigma"], seed=seed),
        "occlusion": occlusion(frames, top=(height - mask_height) // 2, left=(width - mask_width) // 2, height=mask_height, width=mask_width),
        "jpeg_frames": jpeg_compression(frames, quality=config["jpeg_quality"]),
    }


def _compressed_video_frames(path: Path, *, ffmpeg: str, crf: int, video_config: dict):
    """Re-encode one video with H.264. / Recodifica um vídeo com H.264."""
    with tempfile.TemporaryDirectory(prefix="bilstm-video-") as temporary:
        output = Path(temporary) / "compressed.mp4"
        subprocess.run([ffmpeg, "-nostdin", "-y", "-loglevel", "error", "-i", str(path), "-an", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), str(output)], check=True)
        return decode_video(output, **video_config)


def _timing_summary(decode_times, preprocess_times, inference_times, frames):
    def stats(values):
        return {"median_ms": float(np.median(values) * 1000), "p95_ms": float(np.percentile(values, 95) * 1000)}
    combined = np.array(decode_times) + np.array(preprocess_times) + np.array(inference_times)
    return {"unit": "one sampled video sequence", "warmup_inferences": 2, "decode": stats(decode_times), "preprocess_and_transfer": stats(preprocess_times), "inference": stats(inference_times), "offline_total": stats(combined), "inference_throughput_sampled_frames_per_second": float(frames * len(inference_times) / sum(inference_times)), "note": "Offline video processing; this does not measure live acquisition latency."}


def _ffmpeg_details(video_compression, robustness):
    """Validate optional FFmpeg support. / Valida suporte opcional ao FFmpeg."""
    if video_compression and not robustness:
        raise ValueError("video_compression requires robustness=True.")
    ffmpeg = shutil.which("ffmpeg") if video_compression else None
    if video_compression and ffmpeg is None:
        raise FileNotFoundError("FFmpeg with libx264 is required for video compression.")
    if ffmpeg is None:
        return None, None
    info = subprocess.run([ffmpeg, "-version"], check=True, capture_output=True, text=True, timeout=10)
    return ffmpeg, info.stdout.splitlines()[0]


def _evaluation_records(run_dir, external_manifest, dataset_root):
    """Select test or external records. / Seleciona registros de teste ou externos."""
    baseline_manifest = load_manifest(run_dir / "dataset_manifest.json")
    if external_manifest is None:
        records = [record for record in baseline_manifest["records"] if baseline_manifest["assignments"][record["sample_id"]] == "test"]
        return baseline_manifest, records, "test"
    if dataset_root is None:
        raise ValueError("dataset_root is required for external evaluation.")
    manifest = load_manifest(resolve_project_path(external_manifest))
    if manifest["source"] == baseline_manifest["source"]:
        raise ValueError("External evaluation requires a different dataset source.")
    baseline_hashes = {record["sha256"].lower() for record in baseline_manifest["records"]}
    if any(record.get("sha256") is None or record["sha256"].lower() in baseline_hashes for record in manifest["records"]):
        raise ValueError("External dataset has missing hashes or exact overlap with the original dataset.")
    return manifest, manifest["records"], "external"


def _collect_scores(records, root, config, selected, model, robustness, ffmpeg):
    """Score videos and perturbations. / Calcula scores de vídeos e perturbações."""
    labels, scores, groups, ids = [], [], [], []
    decode_times, preprocess_times, inference_times = [], [], []
    variant_scores = {name: [] for name in ("brightness", "contrast", "noise", "occlusion", "jpeg_frames")}
    if ffmpeg is not None:
        variant_scores["h264_video"] = []
    for index, record in enumerate(records):
        start = time.perf_counter()
        frames = decode_video(video_path(root, record), **config["video"])
        decode_times.append(time.perf_counter() - start)
        if index == 0:
            _predict(model, frames, selected)
            _predict(model, frames, selected)
        score, preprocessing, inference = _predict(model, frames, selected)
        labels.append(record["label"])
        scores.append(score)
        groups.append(record["group_id"])
        ids.append(record["sample_id"])
        preprocess_times.append(preprocessing)
        inference_times.append(inference)
        if robustness:
            variants = _variants(frames, config["robustness"], config["training"]["seed"] + index)
            if ffmpeg is not None:
                variants["h264_video"] = _compressed_video_frames(video_path(root, record), ffmpeg=ffmpeg, crf=config["robustness"]["video_crf"], video_config=config["video"])
            for name, variant in variants.items():
                variant_scores[name].append(_predict(model, variant, selected)[0])
    return {"labels": labels, "scores": scores, "groups": groups, "ids": ids, "decode_times": decode_times, "preprocess_times": preprocess_times, "inference_times": inference_times, "variants": variant_scores}


def _save_robustness(target, config, source, unit, data, predictions):
    """Save paired comparisons. / Salva comparações pareadas."""
    comparisons = {}
    for name, values in data["variants"].items():
        changed = (np.asarray(values) >= config["evaluation"]["threshold"]).astype(int)
        save_evaluation(target / name, sample_ids=data["ids"], y_true=data["labels"], y_pred=changed, scores=values, threshold=config["evaluation"]["threshold"], source=source, unit=unit)
        comparisons[name] = {"metrics": binary_metrics(data["labels"], changed), "paired_accuracy_difference": grouped_bootstrap(data["labels"], predictions, data["groups"], metric="accuracy", n_resamples=config["evaluation"]["bootstrap_resamples"], seed=config["training"]["seed"], paired_pred=changed)}
    write_json(target / "robustness.json", {"parameters": config["robustness"], "comparisons": comparisons, "note": "Synthetic perturbations on observed videos; label validity and camera/crowd representativeness require review."})
    plot_robustness_comparison(comparisons, target / "figures")


def evaluate_model(run_id: str, *, dataset_root: str | Path | None = None, external_manifest: str | Path | None = None, device: str | None = None, robustness: bool = False, video_compression: bool = False) -> Path:
    """Evaluate test or independent data. / Avalia teste ou dados independentes."""
    run_dir, config, selected, model = _load_run(run_id, device)
    ffmpeg, ffmpeg_version = _ffmpeg_details(video_compression, robustness)
    manifest, records, evaluation_type = _evaluation_records(run_dir, external_manifest, dataset_root)
    if not records:
        raise ValueError("No videos are available for evaluation.")
    root = resolve_project_path(dataset_root or config["data"]["root"])
    for record in records:
        video_path(root, record, verify_hash=True)
    set_seed(config["training"]["seed"])
    evaluation_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    target = run_dir / "evaluations" / evaluation_id
    target.mkdir(parents=True, exist_ok=False)
    print(f"Evaluation ID: {evaluation_id}", flush=True)
    status = {"id": evaluation_id, "run_id": run_id, "status": "running", "type": evaluation_type, "source": manifest["source"], "unit": "one video; one uniformly sampled sequence", "threshold": config["evaluation"]["threshold"], "robustness": robustness, "video_compression": video_compression, "ffmpeg_version": ffmpeg_version, "checkpoint_sha256": _hash_file(run_dir / "best.pt"), "started_utc": datetime.now(timezone.utc).isoformat()}
    try:
        write_json(target / _RUN_MANIFEST_FILENAME, status)
        write_json(target / "dataset_manifest.json", manifest)
        data = _collect_scores(records, root, config, selected, model, robustness, ffmpeg)
        predictions = (np.asarray(data["scores"]) >= config["evaluation"]["threshold"]).astype(int)
        result = save_evaluation(target / "clean", sample_ids=data["ids"], y_true=data["labels"], y_pred=predictions, scores=data["scores"], threshold=config["evaluation"]["threshold"], source=manifest["source"], unit=status["unit"])
        plot_confusion_matrix(result["metrics"], target / "figures")
        plot_binary_metrics(result["metrics"], target / "figures")
        write_json(target / "confidence_intervals.json", _intervals(data["labels"], predictions, data["groups"], seed=config["training"]["seed"], resamples=config["evaluation"]["bootstrap_resamples"]))
        write_json(target / "timing.json", _timing_summary(data["decode_times"], data["preprocess_times"], data["inference_times"], config["video"]["frames"]))
        if robustness:
            _save_robustness(target, config, manifest["source"], status["unit"], data, predictions)
        status.update(status="completed", finished_utc=datetime.now(timezone.utc).isoformat(), device=str(selected))
        write_json(target / _RUN_MANIFEST_FILENAME, status)
    except Exception as error:
        status.update(status="failed", error_type=type(error).__name__, finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(target / _RUN_MANIFEST_FILENAME, status)
        raise
    return target
