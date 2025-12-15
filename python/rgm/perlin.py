from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .numpy_utils import ArrayF32, _as_rng


def _fade(t: ArrayF32) -> ArrayF32:
    """Cubic fade: 6t^5 - 15t^4 + 10t^3."""
    t = np.asarray(t, dtype=np.float32)
    return (6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3).astype(np.float32)


def _lerp(a: ArrayF32, b: ArrayF32, t: ArrayF32) -> ArrayF32:
    return (a + t * (b - a)).astype(np.float32)


def perlin_1d(period: float, n: int, seed: int) -> ArrayF32:
    """
    1D Perlin noise, ported from `module_geological_model_utility.f90`.
    """
    n = int(n)
    period = float(period)
    rng = _as_rng(seed)

    x = np.arange(n, dtype=np.float32) / n * period
    g = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=int(np.ceil(period)) + 2, replace=True)

    x0 = np.floor(x).astype(np.int32)
    dx = x - x0.astype(np.float32)

    g0 = g[x0] * dx
    g1 = g[x0 + 1] * (dx - 1.0)

    sx = _fade(dx)
    return _lerp(g0, g1, sx)


def perlin_2d(periodx: float, periody: float, nx: int, ny: int, seed: int) -> ArrayF32:
    """
    2D Perlin noise, ported from `module_geological_model_utility.f90`.

    Returns:
        Array with shape (nx, ny).
    """
    nx = int(nx)
    ny = int(ny)
    periodx = float(periodx)
    periody = float(periody)
    rng = _as_rng(seed)

    x = (np.arange(nx, dtype=np.float32) / nx) * periodx
    y = (np.arange(ny, dtype=np.float32) / ny) * periody

    mx = int(np.ceil(periodx)) + 2
    my = int(np.ceil(periody)) + 2
    ang = rng.uniform(0.0, 2.0 * np.pi, size=(mx, my)).astype(np.float32)
    gd_x = np.cos(ang).astype(np.float32)
    gd_y = np.sin(ang).astype(np.float32)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)

    dx = (x - x0.astype(np.float32)).astype(np.float32)[:, None]  # (nx, 1)
    dy = (y - y0.astype(np.float32)).astype(np.float32)[None, :]  # (1, ny)

    x1 = x0 + 1
    y1 = y0 + 1

    # Broadcast indices to (nx, ny)
    X0 = x0[:, None]
    X1 = x1[:, None]
    Y0 = y0[None, :]
    Y1 = y1[None, :]

    g00 = gd_x[X0, Y0] * dx + gd_y[X0, Y0] * dy
    g01 = gd_x[X0, Y1] * dx + gd_y[X0, Y1] * (dy - 1.0)
    g10 = gd_x[X1, Y0] * (dx - 1.0) + gd_y[X1, Y0] * dy
    g11 = gd_x[X1, Y1] * (dx - 1.0) + gd_y[X1, Y1] * (dy - 1.0)

    sx = _fade(dx)
    sy = _fade(dy)
    a = _lerp(g00, g10, sx)
    b = _lerp(g01, g11, sx)
    return _lerp(a, b, sy)


@dataclass(frozen=True, slots=True)
class FractalNoise2D:
    n1: int
    n2: int
    periods1: int = 5
    periods2: int = 5
    seed: int = -1
    octaves: int = 5
    persistence: float = 0.5
    lacunarity: float = 2.0

    def generate(self) -> ArrayF32:
        """
        Multi-octave (fractal) Perlin noise with mean 0 and std 1.
        """
        n1 = int(self.n1)
        n2 = int(self.n2)

        frequency = 1.0
        amplitude = 1.0
        out = np.zeros((n1, n2), dtype=np.float32)

        base_seed = int(self.seed) * 20 if self.seed >= 0 else 0
        for octave in range(1, int(self.octaves) + 1):
            noise = perlin_2d(self.periods1 * frequency, self.periods2 * frequency, n1, n2, base_seed + octave)
            out += amplitude * noise
            amplitude *= float(self.persistence)
            frequency *= float(self.lacunarity)

        out = (out - float(out.mean())) / (float(out.std()) + 1.0e-8)
        return out.astype(np.float32)


def random_circular(
    n: int,
    r: float,
    dr: float,
    alpha: float,
    smooth: float,
    seed: int,
) -> ArrayF32:
    """
    Random smooth radius on a circle, matching the intent of Fortran `random_circular`.
    """
    n = int(n)
    rng = _as_rng(seed)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=n).astype(np.float32)
    z = np.exp(1j * phase).astype(np.complex64)

    omega = (2.0 * np.pi * np.fft.fftfreq(n, d=1.0)).astype(np.float32)
    filt = np.exp(-(omega**2) * float(smooth)).astype(np.float32)
    filt = (filt / np.sqrt((1.0 / (float(alpha) * n)) ** 2 + omega**2)).astype(np.float32)

    zz = np.fft.ifft(np.fft.fft(z) * filt).astype(np.complex64)
    m = zz.real.astype(np.float32)
    m = (m - float(m.mean())) / (float(np.max(np.abs(m))) + 1.0e-8)
    return (m * float(dr) + float(r)).astype(np.float32)

