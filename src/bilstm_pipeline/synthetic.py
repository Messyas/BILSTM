"""Exercise components with invented data. / Exercita componentes com dados inventados."""

from pathlib import Path
import json
import numpy as np

from .dataset import group_split, save_manifest
from .metrics import binary_metrics, grouped_bootstrap
from .model import predict
from .plots import plot_binary_metrics, plot_confusion_matrix, plot_robustness_comparison
from .results import save_evaluation
from .robustness import brightness, contrast, gaussian_noise, occlusion


class _SyntheticMeanModel:
    """Dummy model for illustration. / Modelo fictício apenas para ilustração."""

    def predict_scores(self, batch):
        """Average pixels per sample. / Calcula a média dos pixels por amostra."""
        return np.mean(batch, axis=tuple(range(1, batch.ndim)))


def run_components(output_dir: Path, *, seed: int) -> None:
    """Create synthetic manifests, metrics and figures. / Cria manifestos, métricas e gráficos sintéticos."""
    rng = np.random.default_rng(seed)
    size = 24
    labels = np.arange(size) % 2
    frames = np.clip(rng.normal(0.45, 0.2, size=(size, 3, 10, 10, 3)) + labels[:, None, None, None, None] * 0.10, 0, 1)
    model = _SyntheticMeanModel()
    threshold = 0.5
    scores, predictions = predict(model, frames, threshold=threshold)
    records = [
        {"sample_id": f"sample-{index:03d}", "video_id": f"video-{index // 2:03d}",
         "group_id": f"group-{index // 2:03d}", "source": "synthetic-demo", "label": int(labels[index])}
        for index in range(size)
    ]
    assignments = group_split(records, seed=seed)
    split_metadata = {"seed": seed, "proportions": [0.7, 0.15, 0.15], "algorithm": "group_split", "stratified": False}
    save_manifest(output_dir / "dataset_manifest.json", records, dataset_name="Synthetic demonstration", source="Randomly generated in the project", license_name="Not applicable: synthetic demonstration", assignments=assignments, split_metadata=split_metadata)
    evaluation = save_evaluation(output_dir / "evaluation", sample_ids=[item["sample_id"] for item in records], y_true=labels, y_pred=predictions, scores=scores, threshold=threshold, source="Synthetic demonstration only", unit="synthetic sample")
    figure_dir = output_dir / "figures"
    plot_confusion_matrix(evaluation["metrics"], figure_dir, class_names=("Class 0", "Class 1"), title="Synthetic confusion matrix")
    plot_binary_metrics(evaluation["metrics"], figure_dir, title="Synthetic classification metrics")
    groups = [item["group_id"] for item in records]
    interval = grouped_bootstrap(labels, predictions, groups, metric="accuracy", n_resamples=200, seed=seed)
    variants = {
        "brightness": np.stack([brightness(video, -0.1) for video in frames]),
        "contrast": np.stack([contrast(video, 0.8) for video in frames]),
        "noise": np.stack([gaussian_noise(video, 0.05, seed=seed + index) for index, video in enumerate(frames)]),
        "occlusion": np.stack([occlusion(video, top=0, left=0, height=3, width=3) for video in frames]),
    }
    perturbed = {}
    for name, variant in variants.items():
        _, variant_predictions = predict(model, variant, threshold=threshold)
        perturbed[name] = {
            "metrics": binary_metrics(labels, variant_predictions),
            "paired_accuracy_difference": grouped_bootstrap(labels, predictions, groups, metric="accuracy", n_resamples=200, seed=seed, paired_pred=variant_predictions),
        }
    output = {
        "notice": "Synthetic demonstration only; not scientific results.",
        "unit": "synthetic sample", "group_unit": "synthetic video", "seed": seed, "threshold": threshold,
        "perturbation_parameters": {
            "brightness": {"delta": -0.1}, "contrast": {"factor": 0.8},
            "noise": {"sigma": 0.05, "seed": "run seed plus sample index"},
            "occlusion": {"top": 0, "left": 0, "height": 3, "width": 3, "fill": 0},
        },
        "accuracy_interval": interval, "perturbations": perturbed,
    }
    (output_dir / "component_demo.json").write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    plot_robustness_comparison(perturbed, figure_dir, title="Synthetic robustness comparison")
