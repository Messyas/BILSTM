"""Measure offline and streaming latency. / Mede latência offline e em fluxo contínuo."""

from collections import deque
from datetime import datetime, timezone
import time
from uuid import uuid4

import numpy as np
import torch

from .execution import write_json
from .hardware import hardware_metadata
from .research_config import resolve_project_path
from .research_data import _open_video_capture, decode_video, load_manifest, video_path
from .research_evaluation import WARMUP_INFERENCES, _load_run, _predict
from .research_training import video_tensor


def _stats(values) -> dict:
    array = np.asarray(values, dtype=float) * 1000
    return {"n": int(array.size), "median_ms": float(np.median(array)), "p95_ms": float(np.percentile(array, 95)), "max_ms": float(array.max()), "mean_ms": float(array.mean())}


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _offline(records, root, config, device, model) -> dict:
    """Whole-clip latency: two-pass decode + preprocess + inference. / Latência do clipe inteiro."""
    decode, preprocess, inference = [], [], []
    for record in records:
        start = time.perf_counter()
        frames = decode_video(video_path(root, record), **config["video"])
        decode.append(time.perf_counter() - start)
        _, prep, infer = _predict(model, frames, device)
        preprocess.append(prep)
        inference.append(infer)
    total = np.add(np.add(decode, preprocess), inference)
    return {"unit": "one whole video (two-pass decode of the full file, 15 sampled frames, batch 1)", "decode": _stats(decode), "preprocess_and_transfer": _stats(preprocess), "inference": _stats(inference), "total": _stats(total), "note": "Offline processing time of a stored clip; the clip must be fully recorded first, so this is not a live-alert delay."}


def _stream(records, root, config, device, model, *, fps: float, window_seconds: float, stride_seconds: float) -> dict:
    """Sliding-window replay of each video as a live stream. / Reproduz cada vídeo como fluxo ao vivo."""
    import cv2
    frames_per_window = int(round(window_seconds * fps))
    stride = int(round(stride_seconds * fps))
    steps, width, height = config["video"]["frames"], config["video"]["width"], config["video"]["height"]
    if frames_per_window < steps or stride < 1:
        raise ValueError("The window must contain at least the model's frames and the stride at least one frame.")
    picks = np.arange(steps) * (frames_per_window // steps)
    decode, ingest, assemble, inference, decision_total = [], [], [], [], []
    for record in records:
        capture = _open_video_capture(video_path(root, record))
        buffer: deque = deque(maxlen=frames_per_window)
        arrived = 0
        try:
            while True:
                start = time.perf_counter()
                ok, image = capture.read()
                if not ok:
                    break
                decode.append(time.perf_counter() - start)
                start = time.perf_counter()
                small = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (width, height), interpolation=cv2.INTER_AREA)
                buffer.append(small)
                ingest.append(time.perf_counter() - start)
                arrived += 1
                if arrived < frames_per_window or (arrived - frames_per_window) % stride:
                    continue
                start = time.perf_counter()
                window = np.stack([buffer[index] for index in picks]).astype(np.float32) / 255.0
                tensor = video_tensor(window).unsqueeze(0).to(device)
                _sync(device)
                assembled = time.perf_counter()
                with torch.inference_mode():
                    torch.softmax(model(tensor).double(), dim=1)[0, 1].item()
                _sync(device)
                finished = time.perf_counter()
                assemble.append(assembled - start)
                inference.append(finished - assembled)
                decision_total.append(finished - start)
        finally:
            capture.release()
    if not decision_total:
        raise ValueError("No video is long enough for one streaming window.")
    p95_decision = float(np.percentile(decision_total, 95))
    p95_frame = float(np.percentile(np.add(decode, ingest), 95))
    return {
        "protocol": {"fps": fps, "window_seconds": window_seconds, "window_frames": frames_per_window, "stride_seconds": stride_seconds, "stride_frames": stride, "model_frames_per_window": steps, "frame_offsets": picks.tolist(), "batch_size": 1},
        "per_frame_decode": _stats(decode),
        "per_frame_ingest_resize": _stats(ingest),
        "per_decision_assemble_and_transfer": _stats(assemble),
        "per_decision_inference": _stats(inference),
        "per_decision_total": _stats(decision_total),
        "budget": {
            "frame_interval_ms": 1000 / fps,
            "decision_interval_ms": stride_seconds * 1000,
            "frame_path_fits_p95": p95_frame < 1 / fps,
            "decision_fits_p95": p95_decision < stride_seconds,
            "compute_load_fraction_p95": (p95_frame * stride + p95_decision) / stride_seconds,
            "alert_delay_after_window_complete_p95_ms": p95_decision * 1000,
            "window_formation_delay_s": window_seconds,
        },
        "note": "Video files replayed as if live at the stated fps; capture/network latency of a real camera is not included. Only timing is measured: sliding-window accuracy is not evaluated, and the model was trained on whole-clip sampling. The BiLSTM is bidirectional within each completed window, so an alert can only follow a full window (window_formation_delay_s).",
    }


def measure_latency(run_id: str, *, videos: int = 100, fps: float = 30.0, window_seconds: float = 2.0, stride_seconds: float = 0.5, dataset_root=None, device: str | None = None) -> str:
    """Run offline and streaming benchmarks on test videos. / Executa os benchmarks nos vídeos de teste."""
    run_dir, config, selected, model = _load_run(run_id, device)
    manifest = load_manifest(run_dir / "dataset_manifest.json")
    test = sorted((record for record in manifest["records"] if manifest["assignments"][record["sample_id"]] == "test"), key=lambda record: record["sample_id"])
    rng = np.random.default_rng(config["training"]["seed"])
    records = [test[index] for index in sorted(rng.choice(len(test), size=min(videos, len(test)), replace=False))]
    root = resolve_project_path(dataset_root or config["data"]["root"])
    for record in records:
        # Read every file once so all backbones are timed from the page cache, not cold network storage.
        video_path(root, record).read_bytes()
    warm = decode_video(video_path(root, records[0]), **config["video"])
    for _ in range(WARMUP_INFERENCES):
        _predict(model, warm, selected)
    target = run_dir / "latency" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    target.mkdir(parents=True, exist_ok=False)
    result = {
        "run_id": run_id,
        "backbone": config["model"]["backbone"],
        "device": str(selected),
        "videos": len(records),
        "sample_ids": [record["sample_id"] for record in records],
        "warmup_inferences": WARMUP_INFERENCES,
        "files_preloaded_into_page_cache": True,
        "hardware": hardware_metadata(selected),
        "offline": _offline(records, root, config, selected, model),
        "streaming": _stream(records, root, config, selected, model, fps=fps, window_seconds=window_seconds, stride_seconds=stride_seconds),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(target / "latency.json", result)
    return str(target)
