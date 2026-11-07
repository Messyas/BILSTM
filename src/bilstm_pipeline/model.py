"""Framework-neutral model contract. / Contrato de modelo independente de framework."""

from typing import Protocol
import numpy as np


class ModelAdapter(Protocol):
    """Implement positive-class probabilities. / Implemente probabilidades da classe positiva."""

    def predict_scores(self, batch: np.ndarray) -> np.ndarray:
        """Return one score per input sample. / Retorna uma probabilidade por amostra."""
        ...


def predict(adapter: ModelAdapter, batch, *, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Validate adapter scores and threshold. / Valida probabilidades e limiar do adaptador."""
    array = np.asarray(batch)
    if array.ndim == 0 or len(array) == 0:
        raise ValueError("batch must contain at least one sample.")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and within [0, 1].")
    scores = np.asarray(adapter.predict_scores(array), dtype=float)
    if scores.shape != (len(array),) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("The adapter must return one finite probability in [0, 1] per sample.")
    return scores, (scores >= threshold).astype(int)
