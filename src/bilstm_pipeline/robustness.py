"""Frame perturbations. / Perturbações em quadros."""

import numpy as np


def _prepare(frames):
    array = np.asarray(frames)
    if array.ndim != 4 or array.shape[-1] not in (1, 3) or any(size == 0 for size in array.shape):
        raise ValueError("Frames must have shape (time, height, width, channels), with 1 or 3 channels.")
    if array.dtype == np.uint8:
        return array.astype(float) / 255, array.dtype
    if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
        raise ValueError("Frames must be uint8 or finite floating-point values in [0, 1].")
    return array.astype(float), array.dtype


def _restore(values, dtype):
    values = np.clip(values, 0, 1)
    return np.rint(values * 255).astype(dtype) if dtype == np.uint8 else values.astype(dtype)


def brightness(frames, delta: float):
    """Apply fixed brightness to each frame. / Aplica brilho fixo a cada quadro."""
    values, dtype = _prepare(frames)
    if not np.isfinite(delta) or not -1 <= delta <= 1:
        raise ValueError("delta must be finite and within [-1, 1].")
    return _restore(values + delta, dtype)


def contrast(frames, factor: float):
    """Apply fixed contrast around 0.5. / Aplica contraste fixo em torno de 0,5."""
    values, dtype = _prepare(frames)
    if not np.isfinite(factor) or factor < 0:
        raise ValueError("factor must be finite and nonnegative.")
    return _restore((values - 0.5) * factor + 0.5, dtype)


def gaussian_noise(frames, sigma: float, *, seed: int):
    """Add reproducible independent pixel noise. / Adiciona ruído independente reproduzível por pixel."""
    values, dtype = _prepare(frames)
    if not np.isfinite(sigma) or sigma < 0 or type(seed) is not int or seed < 0:
        raise ValueError("sigma must be nonnegative and seed must be a nonnegative integer.")
    return _restore(values + np.random.default_rng(seed).normal(0, sigma, values.shape), dtype)


def occlusion(frames, *, top: int, left: int, height: int, width: int, fill: float = 0):
    """Mask one fixed region across time. / Mascara uma região fixa ao longo do tempo."""
    values, dtype = _prepare(frames)
    if any(type(value) is not int for value in (top, left, height, width)) or top < 0 or left < 0 or height < 1 or width < 1:
        raise ValueError("Occlusion coordinates must be nonnegative integers and size must be positive.")
    if top + height > values.shape[1] or left + width > values.shape[2] or not np.isfinite(fill) or not 0 <= fill <= 1:
        raise ValueError("Occlusion must fit the frame and fill must be within [0, 1].")
    values[:, top:top + height, left:left + width, :] = fill
    return _restore(values, dtype)


def jpeg_compression(frames, *, quality: int):
    """Compress individual frames as JPEG. / Comprime quadros individuais em JPEG."""
    import cv2
    values, dtype = _prepare(frames)
    if type(quality) is not int or not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be an integer between 1 and 100.")
    decoded = []
    for frame in values:
        image = np.rint(frame * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.shape[-1] == 3 else image[:, :, 0]
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise ValueError("Could not encode a frame as JPEG.")
        restored = cv2.imdecode(encoded, cv2.IMREAD_COLOR if frame.shape[-1] == 3 else cv2.IMREAD_GRAYSCALE)
        if restored is None:
            raise ValueError("Could not decode a JPEG frame.")
        restored = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB) if frame.shape[-1] == 3 else restored[:, :, None]
        decoded.append(restored.astype(float) / 255)
    return _restore(np.stack(decoded), dtype)
