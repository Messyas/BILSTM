"""Research configuration. / Configuração da pesquisa."""

from copy import deepcopy
from pathlib import Path
import math
import tomllib

from .paths import project_root

_BACKBONES = ("mobilenet_v2", "densenet121", "inception_v3")
_POOLING_MODES = ("avg", "flatten")
_OPTIMIZERS = ("sgd", "adam")

_FIELDS = {
    "data": {"root", "manifest"},
    "split": {"train", "validation", "test", "seed"},
    "video": {"frames", "width", "height"},
    "model": {"backbone", "pretrained", "trainable_layers", "pooling", "hidden_size", "attention", "dense_units", "dropout"},
    "training": {
        "epochs", "batch_size", "optimizer", "learning_rate", "momentum", "weight_decay",
        "lr_factor", "lr_patience", "min_learning_rate", "early_stopping_patience",
        "seed", "device", "num_workers",
    },
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
    """Check core parameters. / Verifica parametros centrais."""
    _validate_data_paths(config)
    _validate_positive_integers(config)
    _validate_seeds(config)
    _validate_training_options(config)
    _validate_numeric_ranges(config)


def _validate_data_paths(config):
    """Check dataset paths. / Verifica caminhos do conjunto de dados."""
    for key in ("root", "manifest"):
        if not isinstance(config["data"][key], str) or not config["data"][key].strip():
            raise ValueError(f"data.{key} must be a nonempty path.")


def _validate_positive_integers(config):
    """Check positive integer parameters. / Verifica parametros inteiros positivos."""
    fields = (
        ("video", ("frames", "width", "height")),
        ("model", ("hidden_size",)),
        ("training", ("epochs", "batch_size", "lr_patience", "early_stopping_patience")),
        ("evaluation", ("bootstrap_resamples",)),
    )
    for section, keys in fields:
        for key in keys:
            minimum = 2 if key == "bootstrap_resamples" else 1
            if type(config[section][key]) is not int or config[section][key] < minimum:
                raise ValueError(f"{section}.{key} must be a positive integer.")
    if type(config["model"]["trainable_layers"]) is not int or config["model"]["trainable_layers"] < 0:
        raise ValueError("model.trainable_layers must be a nonnegative integer.")
    dense_units = config["model"]["dense_units"]
    if not isinstance(dense_units, list) or not dense_units or any(type(unit) is not int or unit < 1 for unit in dense_units):
        raise ValueError("model.dense_units must be a nonempty list of positive integers.")


def _validate_seeds(config):
    """Check random seeds. / Verifica sementes aleatorias."""
    for section in ("split", "training"):
        if type(config[section]["seed"]) is not int or not 0 <= config[section]["seed"] < 2**32:
            raise ValueError(f"{section}.seed must be an integer in [0, 2^32).")


def _validate_training_options(config):
    """Check model and training options. / Verifica opcoes de modelo e treinamento."""
    if type(config["training"]["num_workers"]) is not int or config["training"]["num_workers"] < 0:
        raise ValueError("training.num_workers must be a nonnegative integer.")
    if config["training"]["device"] not in ("cuda", "cpu", "auto"):
        raise ValueError("training.device must be cuda, cpu or auto.")
    if config["training"]["optimizer"] not in _OPTIMIZERS:
        raise ValueError(f"training.optimizer must be one of: {', '.join(_OPTIMIZERS)}.")
    if config["model"]["backbone"] not in _BACKBONES:
        raise ValueError(f"model.backbone must be one of: {', '.join(_BACKBONES)}.")
    if config["model"]["pooling"] not in _POOLING_MODES:
        raise ValueError(f"model.pooling must be one of: {', '.join(_POOLING_MODES)}.")
    for key in ("pretrained", "attention"):
        if type(config["model"][key]) is not bool:
            raise ValueError(f"model.{key} must be boolean.")


def _validate_numeric_ranges(config):
    """Check numeric ranges. / Verifica intervalos numericos."""
    ranges = (
        ("model", "dropout", 0, 1, False),
        ("training", "learning_rate", 0, math.inf, False),
        ("training", "momentum", 0, 1, False),
        ("training", "weight_decay", 0, math.inf, True),
        ("training", "lr_factor", 0, 1, False),
        ("training", "min_learning_rate", 0, math.inf, False),
        ("evaluation", "threshold", 0, 1, True),
    )
    for section, key, low, high, inclusive_high in ranges:
        value = config[section][key]
        invalid = type(value) not in (int, float) or not math.isfinite(value)
        invalid = invalid or value < low or (value > high if inclusive_high else value >= high)
        if invalid:
            raise ValueError(f"Invalid {section}.{key}.")
        if key in ("learning_rate", "lr_factor", "min_learning_rate") and value == 0:
            raise ValueError(f"training.{key} must be positive.")
    if config["training"]["min_learning_rate"] > config["training"]["learning_rate"]:
        raise ValueError("training.min_learning_rate cannot exceed training.learning_rate.")


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


def _apply_defaults(config: dict) -> None:
    """Fill defaults for backward compatibility. / Preenche padroes para compatibilidade retroativa."""
    if isinstance(config.get("model"), dict):
        model = config["model"]
        model.setdefault("backbone", "mobilenet_v2")
        model.setdefault("trainable_layers", 80)
        model.setdefault("pooling", "avg")
        model.setdefault("attention", True)
        model.setdefault("dense_units", [256, 128, 64, 32])
        if isinstance(model.get("dense_units"), tuple):
            model["dense_units"] = list(model["dense_units"])
    if isinstance(config.get("training"), dict):
        training = config["training"]
        training.setdefault("optimizer", "sgd")
        training.setdefault("momentum", 0.9)
        training.setdefault("weight_decay", 0.0)
        training.setdefault("lr_factor", 0.6)
        training.setdefault("lr_patience", 5)
        lr = training.get("learning_rate", 0.01)
        default_min_lr = min(0.00005, lr) if type(lr) in (int, float) and lr > 0 else 0.00005
        training.setdefault("min_learning_rate", default_min_lr)
        training.setdefault("early_stopping_patience", 10)


def validate_research_config(config: dict, *, seed: int | None = None) -> dict:
    """Validate research parameters. / Valida parâmetros de pesquisa."""
    config = deepcopy(config)
    _apply_defaults(config)
    _validate_structure(config)
    if seed is not None:
        config["training"]["seed"] = seed
    _validate_core(config)
    _validate_split_and_robustness(config)
    return config
