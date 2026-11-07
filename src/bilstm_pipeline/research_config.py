"""Research configuration. / Configuração da pesquisa."""

from copy import deepcopy
from pathlib import Path
import math
import tomllib

from .paths import project_root

_FIELDS = {
    "data": {"root", "manifest"},
    "split": {"train", "validation", "test", "seed"},
    "video": {"frames", "width", "height"},
    "model": {"hidden_size", "dropout", "pretrained"},
    "training": {"epochs", "batch_size", "learning_rate", "seed", "device", "num_workers"},
    "evaluation": {"threshold", "bootstrap_resamples"},
    "robustness": {"brightness_delta", "contrast_factor", "noise_sigma", "occlusion_fraction", "jpeg_quality", "video_crf"},
}


def resolve_project_path(value: str | Path) -> Path:
    """Resolve paths against the project root. / Resolve caminhos a partir da raiz do projeto."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root() / path).resolve()


def load_research_config(path: str | Path, *, seed: int | None = None) -> dict:
    """Read and validate the explicit protocol. / Lê e valida o protocolo explícito."""
    with resolve_project_path(path).open("rb") as stream:
        config = tomllib.load(stream)
    return validate_research_config(config, seed=seed)


def _validate_structure(config):
    """Check required sections. / Verifica seções obrigatórias."""
    if set(config) != set(_FIELDS):
        raise ValueError(f"Configuration must contain sections: {', '.join(_FIELDS)}.")
    for section, fields in _FIELDS.items():
        if not isinstance(config[section], dict) or set(config[section]) != fields:
            raise ValueError(f"Section [{section}] must contain exactly: {', '.join(sorted(fields))}.")


def _validate_core(config):
    """Check core parameters. / Verifica parâmetros centrais."""
    for key in ("root", "manifest"):
        if not isinstance(config["data"][key], str) or not config["data"][key].strip():
            raise ValueError(f"data.{key} must be a nonempty path.")
    for section, keys in (("video", ("frames", "width", "height")), ("model", ("hidden_size",)), ("training", ("epochs", "batch_size")), ("evaluation", ("bootstrap_resamples",))):
        for key in keys:
            if type(config[section][key]) is not int or config[section][key] < (2 if key == "bootstrap_resamples" else 1):
                raise ValueError(f"{section}.{key} must be a positive integer.")
    for section in ("split", "training"):
        if type(config[section]["seed"]) is not int or not 0 <= config[section]["seed"] < 2**32:
            raise ValueError(f"{section}.seed must be an integer in [0, 2^32).")
    if type(config["training"]["num_workers"]) is not int or config["training"]["num_workers"] < 0:
        raise ValueError("training.num_workers must be a nonnegative integer.")
    if config["training"]["device"] not in ("cuda", "cpu", "auto"):
        raise ValueError("training.device must be cuda, cpu or auto.")
    if type(config["model"]["pretrained"]) is not bool:
        raise ValueError("model.pretrained must be boolean.")
    for section, key, low, high, inclusive_high in (
        ("model", "dropout", 0, 1, False), ("training", "learning_rate", 0, math.inf, False),
        ("evaluation", "threshold", 0, 1, True),
    ):
        value = config[section][key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < low or (value > high if inclusive_high else value >= high):
            raise ValueError(f"Invalid {section}.{key}.")
        if key == "learning_rate" and value == 0:
            raise ValueError("training.learning_rate must be positive.")


def _validate_split_and_robustness(config):
    """Check split and perturbations. / Verifica divisão e perturbações."""
    fractions = [config["split"][name] for name in ("train", "validation", "test")]
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in fractions) or not math.isclose(sum(fractions), 1.0, abs_tol=1e-8):
        raise ValueError("Split proportions must be positive and sum to one.")
    for key, low, high in (("brightness_delta", -1, 1), ("contrast_factor", 0, 10), ("noise_sigma", 0, 1), ("occlusion_fraction", 0, 1)):
        value = config["robustness"][key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"Invalid robustness.{key}.")
    if config["robustness"]["occlusion_fraction"] == 0:
        raise ValueError("robustness.occlusion_fraction must be positive.")
    quality = config["robustness"]["jpeg_quality"]
    if type(quality) is not int or not 1 <= quality <= 100:
        raise ValueError("robustness.jpeg_quality must be an integer in [1, 100].")
    crf = config["robustness"]["video_crf"]
    if type(crf) is not int or not 0 <= crf <= 51:
        raise ValueError("robustness.video_crf must be an integer in [0, 51].")


def validate_research_config(config: dict, *, seed: int | None = None) -> dict:
    """Validate research parameters. / Valida parâmetros de pesquisa."""
    config = deepcopy(config)
    _validate_structure(config)
    if seed is not None:
        config["training"]["seed"] = seed
    _validate_core(config)
    _validate_split_and_robustness(config)
    return config
