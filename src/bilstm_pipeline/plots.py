"""Plotting and export style. / Estilo e exportação de gráficos."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.typing import RcKeyType

STYLE: dict[RcKeyType, Any] = {
    "font.family": "serif", "font.size": 10, "axes.labelsize": 11,
    "axes.titlesize": 12, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linestyle": "--", "savefig.dpi": 300,
}


def plot_confusion_matrix(metrics: dict, output_dir: str | Path, *, class_names=("Non-violence", "Violence"), title="Confusion matrix (counts)") -> None:
    """Plot true rows and predicted columns. / Desenha linhas verdadeiras e colunas previstas."""
    matrix = np.asarray(metrics["confusion_matrix"])
    if matrix.shape != (2, 2) or len(class_names) != 2 or not np.issubdtype(matrix.dtype, np.number) or not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise ValueError("Expected a finite nonnegative 2x2 confusion matrix and two class names.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context(STYLE):
        figure, axes = plt.subplots(figsize=(8 / 2.54, 8 / 2.54), layout="constrained")
        try:
            axes.grid(False)
            image = axes.imshow(matrix, cmap="cividis")
            axes.set_xticks([0, 1], labels=class_names, rotation=20, ha="right")
            axes.set_yticks([0, 1], labels=class_names)
            axes.set(xlabel="Predicted class", ylabel="True class", title=title)
            for row in range(2):
                for col in range(2):
                    normalized = float(np.asarray(image.norm(matrix[row, col]), dtype=float).item())
                    color = "white" if normalized < 0.5 else "black"
                    axes.text(col, row, f"{matrix[row, col]:g}", ha="center", va="center", color=color)
            figure.savefig(output_dir / "confusion_matrix.pdf")
            figure.savefig(output_dir / "confusion_matrix.png", dpi=300)
        finally:
            plt.close(figure)


def plot_binary_metrics(metrics: dict, output_dir: str | Path, *, title="Binary classification metrics") -> None:
    """Plot available positive-class metrics. / Desenha métricas disponíveis da classe positiva."""
    positive = metrics["classes"]["1"]
    values = {"Precision": positive["precision"], "Recall": positive["recall"], "F1": positive["f1"], "Specificity": metrics["specificity"]}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context(STYLE):
        figure, axes = plt.subplots(figsize=(17 / 2.54, 9 / 2.54), layout="constrained")
        try:
            available = [(name, value) for name, value in values.items() if value is not None]
            if not available:
                raise ValueError("No metrics are available to plot.")
            names, numbers = zip(*available, strict=True)
            axes.bar(names, numbers, color="#0173b2")
            axes.set_ylim(0, 1.05)
            axes.set(ylabel="Score (0–1)", title=title)
            axes.spines["top"].set_visible(False)
            axes.spines["right"].set_visible(False)
            figure.savefig(output_dir / "binary_metrics.pdf")
            figure.savefig(output_dir / "binary_metrics.png", dpi=300)
        finally:
            plt.close(figure)


def plot_robustness_comparison(perturbations: dict, output_dir: str | Path, *, title="Robustness comparison") -> None:
    """Plot paired accuracy differences. / Desenha diferenças pareadas de acurácia."""
    if not perturbations:
        raise ValueError("At least one perturbation is required.")
    names = list(perturbations)
    intervals = [perturbations[name]["paired_accuracy_difference"] for name in names]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context(STYLE):
        figure, axes = plt.subplots(figsize=(17 / 2.54, 10 / 2.54), layout="constrained")
        try:
            for index, interval in enumerate(intervals):
                axes.hlines(index, interval["lower"], interval["upper"], color="#0173b2", linewidth=2)
                if interval["estimate"] is not None:
                    axes.plot(interval["estimate"], index, "o", color="#0173b2")
            axes.axvline(0, color="black", linestyle="--", linewidth=1)
            axes.set_yticks(range(len(names)), labels=names)
            axes.set(xlabel="Accuracy difference (perturbed − clean)", title=title)
            axes.spines["top"].set_visible(False)
            axes.spines["right"].set_visible(False)
            figure.savefig(output_dir / "robustness.pdf")
            figure.savefig(output_dir / "robustness.png", dpi=300)
        finally:
            plt.close(figure)


def plot_training_history(history: list[dict], output_dir: str | Path) -> None:
    """Plot train and validation loss and accuracy. / Desenha perda e acurácia de treino e validação."""
    if not history:
        raise ValueError("Training history cannot be empty.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    has_accuracy = all("train_accuracy" in item and "validation_accuracy" in item for item in history)
    with mpl.rc_context(STYLE):
        figure, axes = plt.subplots(1, 2 if has_accuracy else 1, figsize=(17 / 2.54, 9 / 2.54), layout="constrained")
        try:
            loss_ax = axes[0] if has_accuracy else axes
            epochs = [item["epoch"] for item in history]
            loss_ax.plot(epochs, [item["train_loss"] for item in history], marker="o", label="Training")
            loss_ax.plot(epochs, [item["validation_loss"] for item in history], marker="s", label="Validation")
            loss_ax.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Training loss")
            loss_ax.legend()
            if has_accuracy:
                acc_ax = axes[1]
                acc_ax.plot(epochs, [item["train_accuracy"] for item in history], marker="o", label="Training")
                acc_ax.plot(epochs, [item["validation_accuracy"] for item in history], marker="s", label="Validation")
                acc_ax.set_ylim(0, 1.05)
                acc_ax.set(xlabel="Epoch", ylabel="Accuracy (0–1)", title="Training accuracy")
                acc_ax.legend()
            figure.savefig(output_dir / "training_history.pdf")
            figure.savefig(output_dir / "training_history.png", dpi=300)
        finally:
            plt.close(figure)
