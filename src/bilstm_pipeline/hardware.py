"""Record the execution hardware. / Registra o hardware da execução."""

import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess


def _cpu_details() -> dict:
    """Read CPU model and core counts from /proc. / Lê modelo da CPU e núcleos em /proc."""
    model, physical = None, set()
    current: dict[str, str] = {}
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines() + [""]:
            if not line.strip():
                if "core id" in current:
                    physical.add((current.get("physical id", "0"), current["core id"]))
                current = {}
                continue
            key, _, value = line.partition(":")
            current[key.strip()] = value.strip()
            if key.strip() == "model name" and model is None:
                model = value.strip()
    try:
        allocated = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        allocated = None
    return {"model": model or platform.processor() or None, "physical_cores": len(physical) or None, "logical_cpus": os.cpu_count(), "cpus_allocated_to_process": allocated}


def _memory_gib() -> float | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            return round(int(line.split()[1]) / 1024**2, 1)
    return None


def _nvidia_driver() -> str | None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        output = subprocess.run([executable, "--query-gpu=driver_version", "--format=csv,noheader"], check=True, capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return output.strip().splitlines()[0] if output.strip() else None


def hardware_metadata(device=None) -> dict:
    """Describe host, CPU, RAM, GPU and threads. / Descreve host, CPU, RAM, GPU e threads."""
    import torch
    gpu = None
    if device is not None and getattr(device, "type", str(device)) == "cuda" and torch.cuda.is_available():
        index = device.index if getattr(device, "index", None) is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        gpu = {"name": properties.name, "memory_gib": round(properties.total_memory / 1024**3, 1), "compute_capability": f"{properties.major}.{properties.minor}", "multiprocessors": properties.multi_processor_count, "driver": _nvidia_driver(), "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    environment = {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK")}
    return {"hostname": socket.gethostname(), "os": platform.platform(), "cpu": _cpu_details(), "ram_gib": _memory_gib(), "gpu": gpu, "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(), "environment": environment}
