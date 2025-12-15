from __future__ import annotations

from typing import Sequence

import numpy as np

from .numpy_utils import ArrayF32, next_pow2


def gauss_filt(x: ArrayF32, sigma: float | Sequence[float]) -> ArrayF32:
    """
    Gaussian smoothing using FFT along each axis.

    This is a SciPy-free replacement for the Fortran `gauss_filt` used by RGM.

    Args:
        x: Real-valued array (1D/2D/3D).
        sigma: Standard deviation in samples for each axis. Use 0 to skip an axis.
    """
    y = np.asarray(x, dtype=np.float32)
    if isinstance(sigma, (int, float)):
        sigmas = (float(sigma),) * y.ndim
    else:
        sigmas = tuple(float(s) for s in sigma)
        if len(sigmas) != y.ndim:
            raise ValueError(f"sigma ndim mismatch: sigma={sigmas} for x.ndim={y.ndim}")

    out = y
    for axis, s in enumerate(sigmas):
        if s <= 0.0:
            continue
        n = out.shape[axis]
        freqs = np.fft.rfftfreq(n, d=1.0).astype(np.float32)  # cycles/sample
        # Fourier response for a Gaussian kernel with std=s:
        # exp(-2*pi^2*s^2*f^2)
        resp = np.exp(-2.0 * (np.pi**2) * (s**2) * (freqs**2)).astype(np.float32)
        X = np.fft.rfft(out, axis=axis)
        shape = [1] * out.ndim
        shape[axis] = resp.size
        X *= resp.reshape(shape)
        out = np.fft.irfft(X, n=n, axis=axis).astype(np.float32)
    return out


def convolve1d_same_fft(x: ArrayF32, kernel: ArrayF32, axis: int) -> ArrayF32:
    """
    Linear 1D convolution (same-size output) along one axis using FFT.

    Notes:
    - This is designed for long, dense kernels (e.g., wavelet/PSF axes).
    - Output matches the common "same" convention: take the centered part of
      the full convolution of length n + m - 1.
    """
    x = np.asarray(x, dtype=np.float32)
    k = np.asarray(kernel, dtype=np.float32).ravel()
    n = x.shape[axis]
    m = int(k.size)
    if m == 1:
        return x * float(k[0])

    n_full = n + m - 1
    n_fft = next_pow2(n_full)

    # FFT of kernel (zero-padded)
    K = np.fft.rfft(k, n=n_fft).astype(np.complex64)

    # FFT of x along axis (zero-padded)
    X = np.fft.rfft(x, n=n_fft, axis=axis).astype(np.complex64)
    shape = [1] * x.ndim
    shape[axis] = K.size
    X *= K.reshape(shape)
    y_full = np.fft.irfft(X, n=n_fft, axis=axis).astype(np.float32)

    start = (m - 1) // 2
    end = start + n
    slc = [slice(None)] * x.ndim
    slc[axis] = slice(start, end)
    return y_full[tuple(slc)].astype(np.float32)


def apply_separable_psf_same(
    x: ArrayF32,
    k1: ArrayF32,
    k2: ArrayF32,
    k3: ArrayF32,
) -> ArrayF32:
    """
    Apply a separable 3D PSF: k1(z) * k2(y) * k3(x).

    This replaces the Fortran `conv(image, psf, 'same')` where `psf` is built
    as an outer-product of three 1D kernels.
    """
    out = convolve1d_same_fft(x, k1, axis=0)
    out = convolve1d_same_fft(out, k2, axis=1)
    out = convolve1d_same_fft(out, k3, axis=2)
    return out

