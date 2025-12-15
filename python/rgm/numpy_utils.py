from __future__ import annotations

from typing import Sequence

import numpy as np


ArrayF32 = np.ndarray


def _as_rng(seed: int | None) -> np.random.Generator:
    """
    Create a deterministic RNG.

    In the original Fortran, `seed = -1` means "random". Here:
    - `seed is None` or `seed < 0` uses entropy from NumPy.
    - otherwise, it is deterministic.
    """
    if seed is None or seed < 0:
        return np.random.default_rng()
    return np.random.default_rng(int(seed))


def rand_uniform(shape: Sequence[int], low: float, high: float, seed: int | None = None) -> ArrayF32:
    """Uniform random array in [low, high)."""
    rng = _as_rng(seed)
    return rng.uniform(low, high, size=tuple(shape)).astype(np.float32)


def rand_normal(shape: Sequence[int], seed: int | None = None) -> ArrayF32:
    """Standard normal random array."""
    rng = _as_rng(seed)
    return rng.standard_normal(size=tuple(shape)).astype(np.float32)


def irand_uniform(n: int, low: int, high: int, seed: int | None = None) -> np.ndarray:
    """Uniform random integers in [low, high], inclusive."""
    rng = _as_rng(seed)
    return rng.integers(low, high + 1, size=int(n), dtype=np.int64)


def clip(x: ArrayF32, lo: float, hi: float) -> ArrayF32:
    return np.clip(x, lo, hi)


def norm2(x: ArrayF32) -> float:
    """L2 norm, like Fortran `norm2`."""
    return float(np.sqrt(np.sum(np.asarray(x, dtype=np.float64) ** 2)))


def rov(x: ArrayF32) -> float:
    """Range of values (max - min)."""
    x = np.asarray(x)
    return float(np.max(x) - np.min(x))


def rescale(x: ArrayF32, out_range: Sequence[float]) -> ArrayF32:
    """
    Rescale an array to [out_range[0], out_range[1]].

    This matches the Fortran `rescale` usage in this repository.
    """
    a, b = float(out_range[0]), float(out_range[1])
    x = np.asarray(x, dtype=np.float32)
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmax == xmin:
        return np.full_like(x, (a + b) * 0.5, dtype=np.float32)
    y = (x - xmin) / (xmax - xmin)
    return (a + (b - a) * y).astype(np.float32)


def linspace(a: float, b: float, n: int) -> ArrayF32:
    return np.linspace(float(a), float(b), int(n), dtype=np.float32)


def sort1(x: np.ndarray) -> np.ndarray:
    return np.sort(x)


def next_pow2(n: int) -> int:
    """Small helper for FFT padding."""
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def rotate_point(point: Sequence[float], theta: float, center: Sequence[float]) -> tuple[float, float]:
    """
    Rotate a 2D point around a center.

    Args:
        point: (x, y)
        theta: rotation angle in radians.
        center: (cx, cy)
    """
    x, y = float(point[0]), float(point[1])
    cx, cy = float(center[0]), float(center[1])
    ct, st = float(np.cos(theta)), float(np.sin(theta))
    x0, y0 = x - cx, y - cy
    xr = x0 * ct - y0 * st + cx
    yr = x0 * st + y0 * ct + cy
    return xr, yr


def interp1_linear_4pt(
    t: ArrayF32,
    y0: ArrayF32,
    y1: ArrayF32,
    y2: ArrayF32,
    y3: ArrayF32,
    x1: float,
    x2: float,
) -> ArrayF32:
    """
    Vectorized linear interpolation for a fixed 4-point x-grid.

    x-grid is: [0, x1, x2, 1]. `t` is 1D in [0, 1].
    y0..y3 are arrays with the same shape (e.g. (n2, n3)).

    Returns:
        Array of shape (len(t), *y0.shape).
    """
    t = np.asarray(t, dtype=np.float32)
    x1f = float(x1)
    x2f = float(x2)

    out = np.empty((t.size,) + y0.shape, dtype=np.float32)

    m0 = t <= x1f
    m1 = (t > x1f) & (t <= x2f)
    m2 = t > x2f

    if np.any(m0):
        tt = t[m0] / max(x1f, 1.0e-8)
        out[m0] = (1.0 - tt)[:, None, None] * y0 + tt[:, None, None] * y1
    if np.any(m1):
        tt = (t[m1] - x1f) / max(x2f - x1f, 1.0e-8)
        out[m1] = (1.0 - tt)[:, None, None] * y1 + tt[:, None, None] * y2
    if np.any(m2):
        tt = (t[m2] - x2f) / max(1.0 - x2f, 1.0e-8)
        out[m2] = (1.0 - tt)[:, None, None] * y2 + tt[:, None, None] * y3

    return out

