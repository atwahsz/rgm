from __future__ import annotations

import numpy as np


def sinc_wavelet(t: np.ndarray, f0: float) -> np.ndarray:
    """Sinc wavelet: sinc(pi*f0*t)."""
    return np.sinc((np.pi * float(f0) * t) / np.pi).astype(np.float32)


def gaussian_wavelet(t: np.ndarray, f0: float) -> np.ndarray:
    """Gaussian wavelet: exp(-(pi*f0*t)^2)."""
    tt = np.pi * float(f0) * t
    return np.exp(-(tt**2)).astype(np.float32)


def gaussian_deriv_wavelet(t: np.ndarray, f0: float) -> np.ndarray:
    """First derivative of a Gaussian wavelet."""
    tt = np.pi * float(f0) * t
    return (-t * np.exp(-(tt**2))).astype(np.float32)


def ricker_wavelet(t: np.ndarray, f0: float) -> np.ndarray:
    """Ricker wavelet (second derivative of a Gaussian)."""
    tt2 = (np.pi * float(f0) * t) ** 2
    return ((1.0 - 2.0 * tt2) * np.exp(-tt2)).astype(np.float32)


def ricker_deriv_wavelet(t: np.ndarray, f0: float) -> np.ndarray:
    """First derivative of a Ricker wavelet."""
    tt2 = (np.pi * float(f0) * t) ** 2
    return ((2.0 * tt2 - 3.0) * np.exp(-tt2) * t).astype(np.float32)

