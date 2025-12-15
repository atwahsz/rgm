from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from .filters import apply_separable_psf_same, gauss_filt
from .numpy_utils import (
    ArrayF32,
    interp1_linear_4pt,
    linspace,
    norm2,
    rand_normal,
    rand_uniform,
    rescale,
    rotate_point,
)
from .perlin import FractalNoise1D, FractalNoise2D, random_circular
from .wavelets import (
    gaussian_deriv_wavelet,
    gaussian_wavelet,
    ricker_deriv_wavelet,
    ricker_wavelet,
    sinc_wavelet,
)


WaveName = Literal["ricker", "ricker_deriv", "gaussian", "gaussian_deriv", "sinc", "delta", ""]
ReflShape = Literal["random", "perlin", "gaussian", "cauchy"]


@dataclass(frozen=True, slots=True)
class RGM3CurvedConfig:
    """
    Configuration for a 3D random geological model with curved faults.

    This mirrors the core fields of the Fortran `type(rgm3_curved)` used in
    `src/module_geological_model_3d_curved.f90`, with a few simplifications.
    """

    # Grid
    n1: int = 200  # vertical (Z)
    n2: int = 128  # horizontal (Y)
    n3: int = 128  # horizontal (X)

    # Randomness
    seed: int = 123

    # Layering
    nl: int = 30
    refl_shape: ReflShape = "perlin"
    refl_shape_top: Literal["same", "random", "perlin", "gaussian", "cauchy"] = "perlin"
    refl_smooth: float = 3.0
    refl_smooth_top: float = 3.0
    refl_slope_yx: tuple[float, float] = (20.0, -10.0)  # (slope along n2, slope along n3)
    refl_slope_top_yx: tuple[float, float] = (-2.0, 3.0)
    refl_height: tuple[float, float] = (0.0, 50.0)
    refl_height_top: tuple[float, float] = (0.0, 4.0)
    ng: int = 3
    refl_sigma2: tuple[float, float] = (20.0, 40.0)
    refl_sigma3: tuple[float, float] = (20.0, 40.0)
    refl_mu2: tuple[float, float] = (0.0, 0.0)
    refl_mu3: tuple[float, float] = (0.0, 0.0)
    lwv: float = 0.25
    lwh: float = 0.1
    secondary_refl_smooth: float = 10.0
    rotate_fold: bool = False

    # Velocity
    vmin: float = 2000.0
    vmax: float = 4000.0
    delta_v: float = 500.0
    rho_a: float = 310.0
    rho_b: float = 0.25
    rho_c: float = 0.0

    # Faults
    yn_fault: bool = True
    nf: int = 7
    yn_regular_fault: bool = False
    yn_group_faults: bool = False
    fwidth: float = 3.0
    dip_deg: tuple[float, float] = (50.0, 130.0)
    strike_deg: tuple[float, float] = (0.0, 180.0)
    rake_deg: tuple[float, float] = (0.0, 180.0)
    disp: tuple[float, float] = (5.0, 12.0)
    delta_dip_deg: tuple[float, float] = (0.0, 10.0)

    # Salt
    yn_salt: bool = False
    nsalt: int = 3
    salt_radius: tuple[float, float] = (15.0, 40.0)
    salt_top_z: tuple[float, float] = (0.4, 0.6)  # fraction of n1 (top of salt)
    salt_top_height: float = 20.0
    salt_path_variation: float = 20.0
    salt_nnode: int = 10
    salt_vp: float = 5000.0
    salt_rho: float = 2150.0
    salt_radius_variation: float = 0.9

    # Unconformity
    unconf: int = 0
    unconf_z: tuple[float, float] = (0.1, 0.3)
    unconf_height: tuple[float, float] = (10.0, 20.0)
    unconf_smooth: float = 20.0

    # Image
    yn_image: bool = True
    wave: WaveName = "ricker"
    dt: float = 1.0e-3
    f0: float = 150.0
    psf_sigma: tuple[float, float, float] = (10.0, 1.0, 1.0)

    # Additive noise in the image domain (matches Fortran generate_image)
    noise_level: float = 0.0
    noise_type: Literal["normal", "gaussian", "uniform", "exp", "wavenumber"] = "normal"
    noise_smooth: tuple[float, float, float] = (1.0, 1.0, 1.0)
    yn_conv_noise: bool = False


@dataclass(frozen=True, slots=True)
class RGM3CurvedResult:
    vp: ArrayF32
    rho: ArrayF32
    image: ArrayF32 | None
    fault: ArrayF32 | None
    fault_dip: ArrayF32 | None
    fault_strike: ArrayF32 | None
    salt: ArrayF32 | None
    rgt: ArrayF32 | None
    facies: ArrayF32 | None


class RGM3Curved:
    """
    Generate 3D random geological models (Python port of `rgm3_curved`).
    """

    def __init__(self, cfg: RGM3CurvedConfig) -> None:
        self.cfg = cfg

    # -----------------------------
    # Public API
    # -----------------------------

    def generate(self) -> RGM3CurvedResult:
        """
        Generate the model.

        Returns:
            A result containing Vp, density, optional seismic image, faults, and salt mask.
        """
        if self.cfg.unconf and self.cfg.unconf > 0:
            return self._generate_unconformal()
        return self._generate_geological()

    # -----------------------------
    # Core steps (ported + simplified)
    # -----------------------------

    def _make_wavelet_and_psf_kernels(self, n1: int, n2: int, n3: int) -> tuple[ArrayF32, ArrayF32, ArrayF32]:
        cfg = self.cfg
        t = ((np.arange(n1, dtype=np.float32) - (n1 - 1.0) / 2.0) * float(cfg.dt)).astype(np.float32)

        if cfg.wave == "":
            k1 = np.zeros(n1, dtype=np.float32)
            k1[n1 // 2] = 1.0
        elif cfg.wave == "delta":
            k1 = np.zeros(n1, dtype=np.float32)
            k1[n1 // 2] = 1.0
        elif cfg.wave == "ricker":
            k1 = ricker_wavelet(t, cfg.f0)
        elif cfg.wave == "ricker_deriv":
            k1 = ricker_deriv_wavelet(t, cfg.f0)
        elif cfg.wave == "gaussian":
            k1 = gaussian_wavelet(t, cfg.f0)
        elif cfg.wave == "gaussian_deriv":
            k1 = gaussian_deriv_wavelet(t, cfg.f0)
        elif cfg.wave == "sinc":
            k1 = sinc_wavelet(t, cfg.f0)
        else:
            k1 = ricker_wavelet(t, cfg.f0)

        # PSF windows along each axis (Gaussian; sigma=0 -> delta)
        def _gauss_1d(n: int, sigma: float) -> ArrayF32:
            x = np.arange(n, dtype=np.float32) - (n - 1.0) / 2.0
            if sigma == 0.0:
                out = np.zeros(n, dtype=np.float32)
                out[n // 2] = 1.0
                return out
            return np.exp(-0.5 * (x**2) / (sigma**2)).astype(np.float32)

        ps1 = _gauss_1d(n1, float(cfg.psf_sigma[0]))
        ps2 = _gauss_1d(n2, float(cfg.psf_sigma[1]))
        ps3 = _gauss_1d(n3, float(cfg.psf_sigma[2]))

        # Normalize to match Fortran: psf = wavelet * ps1 * ps2 * ps3, then psf /= norm2(psf)
        k1 = (k1 / (norm2(k1) + 1.0e-12)).astype(np.float32)
        k1 = (k1 * ps1).astype(np.float32)

        n_k1 = norm2(k1) + 1.0e-12
        n_ps2 = norm2(ps2) + 1.0e-12
        n_ps3 = norm2(ps3) + 1.0e-12
        k1 = (k1 / (n_k1 * n_ps2 * n_ps3)).astype(np.float32)
        return k1, (ps2 / n_ps2).astype(np.float32), (ps3 / n_ps3).astype(np.float32)

    def _generate_reflector_surface(
        self,
        n2: int,
        n3: int,
        shape: Literal["random", "perlin", "gaussian", "cauchy"],
        smooth: float,
        height: tuple[float, float],
        slope_yx: tuple[float, float],
        seed_mul: int,
        *,
        ne2: int,
        ne3: int,
        crop_n2: int,
        crop_n3: int,
    ) -> ArrayF32:
        cfg = self.cfg
        sd = int(cfg.seed) * seed_mul

        if shape == "random":
            r = rand_normal((n2, n3), seed=sd)
            r = gauss_filt(r, (smooth, smooth))
        elif shape == "perlin":
            pn = FractalNoise2D(n1=n2, n2=n3, seed=sd, octaves=4)
            r = pn.generate()
            r = gauss_filt(r, (smooth, smooth))
        elif shape in ("gaussian", "cauchy"):
            yy = np.linspace(0.0, n2 - 1.0, n2, dtype=np.float32)[:, None]
            xx = np.linspace(0.0, n3 - 1.0, n3, dtype=np.float32)[None, :]
            r = np.zeros((n2, n3), dtype=np.float32)
            # Match Fortran defaults: mu ranges default to [1, this%n2-1]/[1, this%n3-1] (cropped sizes),
            # sigma ranges default to [0.05, 0.15]*n2/n3 when user provides zeros.
            mu2_rng = cfg.refl_mu2 if max(cfg.refl_mu2) != 0.0 else (1.0, float(crop_n2 - 1))
            mu3_rng = cfg.refl_mu3 if max(cfg.refl_mu3) != 0.0 else (1.0, float(crop_n3 - 1))
            sg2_rng = cfg.refl_sigma2 if max(cfg.refl_sigma2) != 0.0 else (0.05 * n2, 0.15 * n2)
            sg3_rng = cfg.refl_sigma3 if max(cfg.refl_sigma3) != 0.0 else (0.05 * n3, 0.15 * n3)

            mu2 = rand_uniform((cfg.ng,), float(mu2_rng[0]), float(mu2_rng[1]), seed=sd).astype(np.float32) + float(ne2)
            mu3 = rand_uniform((cfg.ng,), float(mu3_rng[0]), float(mu3_rng[1]), seed=sd + 1).astype(np.float32) + float(ne3)
            sg2 = rand_uniform((cfg.ng,), float(sg2_rng[0]), float(sg2_rng[1]), seed=sd + 2).astype(np.float32)
            sg3 = rand_uniform((cfg.ng,), float(sg3_rng[0]), float(sg3_rng[1]), seed=sd + 3).astype(np.float32)
            hh = rand_uniform((cfg.ng,), float(height[0]), float(height[1]), seed=sd + 4).astype(np.float32)
            thetas = (
                rand_uniform((cfg.ng,), 0.0, 180.0, seed=sd + 5).astype(np.float32) * (np.pi / 180.0)
                if cfg.rotate_fold
                else np.zeros((cfg.ng,), dtype=np.float32)
            )
            for i in range(cfg.ng):
                ct = float(np.cos(thetas[i]))
                st = float(np.sin(thetas[i]))
                y0 = yy - float(mu2[i])
                x0 = xx - float(mu3[i])
                yr = ct * y0 + st * x0
                xr = -st * y0 + ct * x0
                if shape == "gaussian":
                    blob = np.exp(-0.5 * ((yr / float(sg2[i])) ** 2 + (xr / float(sg3[i])) ** 2)).astype(np.float32)
                else:
                    blob = (1.0 / (1.0 + (yr / float(sg2[i])) ** 2 + (xr / float(sg3[i])) ** 2)).astype(np.float32)
                r += rescale(blob, (0.0, float(hh[i])))
        else:
            raise ValueError(f"unsupported reflector shape: {shape}")

        # Rescale reflectors to their height, matching Fortran's rov(r)/rov(crop(r))
        crop = r[ne2 : ne2 + crop_n2, ne3 : ne3 + crop_n3]
        scale = (np.max(r) - np.min(r)) / (np.max(crop) - np.min(crop) + 1.0e-8)
        r = rescale(r, (float(height[0]) * float(scale), float(height[1]) * float(scale)))

        # Add slopes (slopes are scaled by the *cropped/original* dimensions)
        sy, sx = float(slope_yx[0]), float(slope_yx[1])
        y = (np.arange(n2, dtype=np.float32) / max(crop_n2, 1))[:, None]
        x = (np.arange(n3, dtype=np.float32) / max(crop_n3, 1))[None, :]
        r = (r + y * sy + x * sx).astype(np.float32)
        r = (r - float(r.mean())).astype(np.float32)
        return r

    def _smooth_vp_from_reflectors(
        self,
        lz: ArrayF32,
        vv: ArrayF32,
        *,
        n1: int,
    ) -> ArrayF32:
        """
        Match Fortran vertical interpolation:
        create dense samples between reflectors then linearly interpolate to the voxel grid.
        """
        nl, n2, n3 = lz.shape
        nd = int(np.rint(n1 * 2.0 / nl))
        rc = np.empty(((nl - 1) * nd,), dtype=np.float32)
        rfc = np.empty(((nl - 1) * nd,), dtype=np.float32)
        zq = np.linspace(n1 - 1.0, 0.0, n1, dtype=np.float32)
        w = np.empty((n1, n2, n3), dtype=np.float32)

        for j in range(n2):
            for k in range(n3):
                bounds = lz[:, j, k]
                for l in range(nl - 1):
                    tp = float(bounds[l + 1] - bounds[l])
                    lo = float(bounds[l] + tp / (nd + 1))
                    hi = float(bounds[l + 1] - tp / (nd + 1))
                    rc[l * nd : (l + 1) * nd] = np.linspace(lo, hi, nd, dtype=np.float32)
                    rfc[l * nd : (l + 1) * nd] = float(vv[l])
                # Ensure rc is increasing for np.interp
                order = np.argsort(rc)
                w[:, j, k] = np.interp(zq, rc[order], rfc[order]).astype(np.float32)
        return w

    def _fault_params(self, nf: int, n1: int, n2: int, n3: int) -> tuple[ArrayF32, ArrayF32, ArrayF32, ArrayF32, ArrayF32]:
        """
        Sample fault parameters (dip/strike/rake/disp) and build curved dip profile.
        """
        cfg = self.cfg
        if not cfg.yn_fault or nf <= 0:
            dips = np.zeros((n1, 1), dtype=np.float32)
            dip = np.zeros((1,), dtype=np.float32)
            strike = np.zeros((1,), dtype=np.float32)
            rake = np.zeros((1,), dtype=np.float32)
            disp = np.zeros((1,), dtype=np.float32)
            return dips, dip, strike, rake, disp

        dip = rand_uniform((nf,), float(cfg.dip_deg[0]), float(cfg.dip_deg[1]), seed=cfg.seed).astype(np.float32) * (
            np.pi / 180.0
        )
        strike = rand_uniform((nf,), float(cfg.strike_deg[0]), float(cfg.strike_deg[1]), seed=cfg.seed * 2).astype(
            np.float32
        ) * (np.pi / 180.0)
        rake = rand_uniform((nf,), float(cfg.rake_deg[0]), float(cfg.rake_deg[1]), seed=cfg.seed * 3).astype(
            np.float32
        ) * (np.pi / 180.0)
        disp = rand_uniform((nf,), float(cfg.disp[0]), float(cfg.disp[1]), seed=cfg.seed * 4).astype(np.float32)
        disp[::2] *= -1.0  # alternate sign like Fortran

        # Curved dip profiles (vary dip with depth)
        delta = rand_uniform((nf,), float(cfg.delta_dip_deg[0]), float(cfg.delta_dip_deg[1]), seed=cfg.seed * 5).astype(
            np.float32
        ) * (np.pi / 180.0)
        z = np.arange(n1, dtype=np.float32)
        dips = np.empty((n1, nf), dtype=np.float32)
        for i in range(nf):
            if dip[i] <= (np.pi / 2.0):
                dips[:, i] = np.clip(dip[i] + (z / max(n1 - 1, 1)) * (-delta[i]), 0.0, np.pi / 2.0)
            else:
                dips[:, i] = np.clip(dip[i] + (z / max(n1 - 1, 1)) * (+delta[i]), np.pi / 2.0, np.pi)
        return dips, dip, strike, rake, disp

    def _apply_faults(
        self,
        w: ArrayF32,
        dips: ArrayF32,
        dip: ArrayF32,
        strike: ArrayF32,
        rake: ArrayF32,
        disp: ArrayF32,
        seed: int,
    ) -> tuple[ArrayF32, ArrayF32, ArrayF32, ArrayF32]:
        """
        Apply faults by translating one side of each curved fault plane.

        Returns:
            fault_id, fault_dip, fault_strike, shifted_w
        """
        cfg = self.cfg
        n1, n2, n3 = w.shape
        if not cfg.yn_fault or cfg.nf <= 0:
            return (
                np.zeros((n1, n2, n3), dtype=np.float32),
                np.zeros((n1, n2, n3), dtype=np.float32),
                np.zeros((n1, n2, n3), dtype=np.float32),
                w,
            )

        nf = int(cfg.nf)
        fwidth = float(cfg.fwidth)

        # Fault centers (simplified: random without spacing constraints)
        f2 = rand_uniform((nf,), 0.1 * n2, 0.9 * n2, seed=seed * 7).astype(np.float32)
        f3 = rand_uniform((nf,), 0.1 * n3, 0.9 * n3, seed=seed * 8).astype(np.float32)

        # Correct centers so patterns occur in depth center (like Fortran)
        for i in range(nf):
            f2[i] = f2[i] - np.tan(dip[i] - np.pi / 2.0) * n1 * 0.5 * np.cos(strike[i])
            f3[i] = f3[i] - np.tan(dip[i] - np.pi / 2.0) * n1 * 0.5 * np.sin(strike[i])
            f2[i], f3[i] = rotate_point((float(f2[i]), float(f3[i])), (np.pi / 2.0) - float(strike[i]), (0.5 * (n2 - 1), 0.5 * (n3 - 1)))

        fault_id = np.zeros((n1, n2, n3), dtype=np.float32)
        fault_dip = np.zeros((n1, n2, n3), dtype=np.float32)
        fault_strike = np.zeros((n1, n2, n3), dtype=np.float32)

        # Precompute grids for each slice (j,k)
        J = np.arange(n2, dtype=np.float32)[:, None]  # (n2,1)
        K = np.arange(n3, dtype=np.float32)[None, :]  # (1,n3)

        w_out = w.copy()

        for fi in range(nf):
            ww = w_out.copy()
            fblock = np.zeros((n1, n2, n3), dtype=bool)

            a_strike = float(strike[fi])
            sinS = float(np.sin(a_strike))
            cosS = float(np.cos(a_strike))

            for i in range(n1):
                theta = float(dips[i, fi])
                # Fault location shift with depth
                xs = -1.0 / np.tan(theta) * sinS * float(i)
                ys = +1.0 / np.tan(theta) * cosS * float(i)

                if abs(a_strike - np.pi / 2.0) >= np.pi / 4.0:
                    a = float(np.tan(a_strike))
                    b = (float(f2[fi]) + ys) - a * (float(f3[fi]) + xs)
                    # distance to line y = a x + b, in (k -> x, j -> y) coordinates
                    dist = np.abs(J - (a * K + b)) / np.sqrt(1.0 + a * a)
                    mask = dist < (0.5 * fwidth / max(np.sin(theta), 1.0e-6))
                    fault_id[i, :, :][mask] = float(fi + 1)
                    fault_dip[i, :, :][mask] = theta * (180.0 / np.pi)
                    fault_strike[i, :, :][mask] = a_strike * (180.0 / np.pi)
                    if theta < (np.pi / 2.0):
                        fblock[i, :, :][J - (a * K + b) < 0] = True
                    else:
                        fblock[i, :, :][J - (a * K + b) > 0] = True
                else:
                    a = float(np.tan((np.pi / 2.0) - a_strike))
                    b = (float(f3[fi]) + xs) - a * (float(f2[fi]) + ys)
                    dist = np.abs(K - (a * J + b)) / np.sqrt(1.0 + a * a)
                    mask = dist < (0.5 * fwidth / max(np.sin(theta), 1.0e-6))
                    fault_id[i, :, :][mask] = float(fi + 1)
                    fault_dip[i, :, :][mask] = theta * (180.0 / np.pi)
                    fault_strike[i, :, :][mask] = a_strike * (180.0 / np.pi)
                    if theta < (np.pi / 2.0):
                        fblock[i, :, :][K - (a * J + b) <= 0] = True
                    else:
                        fblock[i, :, :][K - (a * J + b) >= 0] = True

            # Translate the fault block
            di = int(np.rint((-np.sin(float(rake[fi])) * np.sin(float(dip[fi]))) * float(disp[fi])))
            dj = int(
                np.rint(
                    (np.cos(float(rake[fi])) * np.sin(float(strike[fi])) - np.sin(float(rake[fi])) * np.cos(float(dip[fi])) * np.cos(float(strike[fi])))
                    * float(disp[fi])
                )
            )
            dk = int(
                np.rint(
                    (np.cos(float(rake[fi])) * np.cos(float(strike[fi])) + np.sin(float(rake[fi])) * np.cos(float(dip[fi])) * np.sin(float(strike[fi])))
                    * float(disp[fi])
                )
            )

            ii, jj, kk = np.nonzero(fblock)
            ni = ii + di
            nj = jj + dj
            nk = kk + dk
            ok = (ni >= 0) & (ni < n1) & (nj >= 0) & (nj < n2) & (nk >= 0) & (nk < n3)
            ii = ii[ok]
            jj = jj[ok]
            kk = kk[ok]
            ni = ni[ok]
            nj = nj[ok]
            nk = nk[ok]

            w_out[ni, nj, nk] = ww[ii, jj, kk]

        return fault_id, fault_dip, fault_strike, w_out

    def _generate_geological(self) -> RGM3CurvedResult:
        cfg = self.cfg

        n1o = int(cfg.n1)
        n2o = int(cfg.n2)
        n3o = int(cfg.n3)

        # --- Fault params first (needed for padding like Fortran)
        dips0, dip0, strike0, rake0, disp0 = self._fault_params(int(cfg.nf), n1o, n2o, n3o)
        if not cfg.yn_fault or cfg.nf <= 0:
            dip0 = np.zeros((1,), dtype=np.float32)
            strike0 = np.zeros((1,), dtype=np.float32)
            rake0 = np.zeros((1,), dtype=np.float32)
            disp0 = np.zeros((1,), dtype=np.float32)

        # --- Compute padding (ne1/ne2/ne3) to match Fortran
        sum1 = disp0 * (-np.sin(rake0) * np.sin(dip0))
        m1 = max(float(np.sum(sum1[sum1 > 0])), float(-np.sum(sum1[sum1 < 0]))) if sum1.size else 0.0
        m2 = max(abs(cfg.refl_slope_yx[0]), abs(cfg.refl_slope_yx[1]), abs(cfg.refl_slope_top_yx[0]), abs(cfg.refl_slope_top_yx[1]))
        m3 = max(abs(cfg.refl_height[0]), abs(cfg.refl_height[1]), abs(cfg.refl_height_top[0]), abs(cfg.refl_height_top[1]))
        ne1 = int(np.ceil(max(m1, m2) + m3))

        sum2 = disp0 * (np.cos(rake0) * np.sin(strike0) - np.sin(rake0) * np.cos(dip0) * np.cos(strike0))
        ne2 = int(np.ceil(max(float(np.sum(sum2[sum2 > 0])), float(-np.sum(sum2[sum2 < 0]))))) if sum2.size else 0

        sum3 = disp0 * (np.cos(rake0) * np.cos(strike0) + np.sin(rake0) * np.cos(dip0) * np.sin(strike0))
        ne3 = int(np.ceil(max(float(np.sum(sum3[sum3 > 0])), float(-np.sum(sum3[sum3 < 0]))))) if sum3.size else 0

        n1 = n1o + 2 * ne1
        n2 = n2o + 2 * ne2
        n3 = n3o + 2 * ne3

        # Reflector surfaces (bottom + top)
        r = self._generate_reflector_surface(
            n2=n2,
            n3=n3,
            shape=cfg.refl_shape,
            smooth=cfg.refl_smooth,
            height=cfg.refl_height,
            slope_yx=cfg.refl_slope_yx,
            seed_mul=5,
            ne2=ne2,
            ne3=ne3,
            crop_n2=n2o,
            crop_n3=n3o,
        )
        if cfg.refl_shape_top == "same":
            rt = r.copy()
        else:
            rt = self._generate_reflector_surface(
                n2=n2,
                n3=n3,
                shape=cfg.refl_shape_top if cfg.refl_shape_top != "same" else cfg.refl_shape,
                smooth=cfg.refl_smooth_top,
                height=cfg.refl_height_top,
                slope_yx=cfg.refl_slope_top_yx,
                seed_mul=6,
                ne2=ne2,
                ne3=ne3,
                crop_n2=n2o,
                crop_n3=n3o,
            )

        # Like Fortran: bottom around 0, top around n1
        r = (r - float(r.mean())).astype(np.float32)
        rt = (rt - float(rt.mean()) + float(n1)).astype(np.float32)

        # Match Fortran: adjust nl for padded n1
        nl = int(np.rint(float(cfg.nl) + float(cfg.nl) * 2.0 * float(ne1) / max(float(n1o), 1.0)))
        tgrid = np.linspace(0.0, 1.0, nl, dtype=np.float32)
        x1 = (float(ne1) + 1.0) / max(float(n1), 1.0)
        x2 = (float(n1) - float(ne1) - 1.0) / max(float(n1), 1.0)
        lz0 = interp1_linear_4pt(tgrid, r, r + float(ne1), rt - float(ne1), rt, x1=x1, x2=x2)  # (nl,n2,n3)

        # Layer thickness variation (vertical)
        dlz = np.diff(lz0, axis=0, prepend=lz0[:1]).astype(np.float32)
        dlz = np.maximum(dlz, 1.0e-9).astype(np.float32)
        plw = rand_uniform((nl,), 1.0 - float(cfg.lwv), 1.0 + float(cfg.lwv), seed=cfg.seed * 7).astype(np.float32)
        dlz = (dlz * plw[:, None, None]).astype(np.float32)
        csum = np.cumsum(dlz, axis=0).astype(np.float32)
        scale = (rt - r) / (csum[-1] + 1.0e-8)
        lz = (r[None, :, :] + csum * scale[None, :, :]).astype(np.float32)

        # Horizontal layer thickness variation (match Fortran structure)
        if cfg.lwh > 0.0:
            pxy = np.zeros((nl, n2, n3), dtype=np.float32)
            for li in range(nl - 1):
                thick = float((lz[li + 1, 0, 0] - lz[li, 0, 0]) * cfg.lwh * 2.0)
                rr = gauss_filt(
                    rand_uniform((n2, n3), 0.0, 1.0, seed=cfg.seed * 7 - li),
                    (cfg.secondary_refl_smooth, cfg.secondary_refl_smooth),
                )
                taper = 0.5 + (float(nl - li) / max(float(nl - 1), 1.0)) * 0.5
                rr = rescale(rr, (-thick * taper, thick * taper))
                pxy[li, :, :] = rr

            dlz2 = np.diff(lz + pxy, axis=0, prepend=(lz + pxy)[:1]).astype(np.float32)
            dlz2 = np.maximum(dlz2, 1.0e-9).astype(np.float32)
            csum2 = np.cumsum(dlz2, axis=0).astype(np.float32)
            scale2 = (rt - r) / (csum2[-1] + 1.0e-8)
            lz = (r[None, :, :] + csum2 * scale2[None, :, :]).astype(np.float32)

        # Layer velocities (match Fortran scaling by padding)
        vv = linspace(
            float(cfg.vmax) * (1.0 + float(ne1) / max(float(n1o), 1.0)),
            float(cfg.vmin) * (1.0 - float(ne1) / max(float(n1o), 1.0)),
            nl - 1,
        ) + rand_uniform((nl - 1,), 0.0, float(cfg.delta_v), seed=cfg.seed * 6)
        vv = vv.astype(np.float32)

        # Build layered velocity model with smooth vertical interpolation (Fortran style)
        w = self._smooth_vp_from_reflectors(lz, vv, n1=n1)

        # Apply faults
        fault = fault_dip = fault_strike = None
        dips, dip, strike, rake, disp = self._fault_params(int(cfg.nf), n1, n2, n3)
        if cfg.yn_fault and cfg.nf > 0:
            fid, fdip, fstr, w = self._apply_faults(w, dips, dip, strike, rake, disp, seed=int(cfg.seed) * 17)
            fault = fid.astype(np.float32)
            fault_dip = fdip.astype(np.float32)
            fault_strike = fstr.astype(np.float32)

        # Crop to original model (Fortran uses vp with +1 depth for reflectivity; we keep both)
        vp_full = w[ne1 : ne1 + n1o + 1, ne2 : ne2 + n2o, ne3 : ne3 + n3o].astype(np.float32)
        vp_full = rescale(vp_full, (cfg.vmin, cfg.vmax)).astype(np.float32)
        vp = vp_full[:n1o, :, :].astype(np.float32)
        rho = (float(cfg.rho_a) * (vp.astype(np.float32) ** float(cfg.rho_b)) + float(cfg.rho_c)).astype(np.float32)

        # Add salt (simplified but close to Fortran behavior)
        salt = None
        if cfg.yn_salt:
            salt = np.zeros((n1, n2, n3), dtype=np.float32)
            # Salt top surface variation
            topz = FractalNoise2D(n1=n2, n2=n3, seed=int(cfg.seed) * 19, octaves=4).generate()
            topz = rescale(topz, (0.0, float(cfg.salt_top_height)))

            # Salt body centers and sizes
            rr = rand_uniform((cfg.nsalt,), float(cfg.salt_radius[0]), float(cfg.salt_radius[1]), seed=int(cfg.seed) * 21)
            cx = rand_uniform((cfg.nsalt,), 0.2 * n3, 0.8 * n3, seed=int(cfg.seed) * 22)
            cy = rand_uniform((cfg.nsalt,), 0.2 * n2, 0.8 * n2, seed=int(cfg.seed) * 23)
            for isalt in range(int(cfg.nsalt)):
                nd = int(np.rint((1.0 - rand_uniform((1,), float(cfg.salt_top_z[0]), float(cfg.salt_top_z[1]), seed=int(cfg.seed) * 24 + isalt)[0]) * n1))
                # Random, smooth radius per depth (control curves)
                slice_r = np.zeros((360, n1), dtype=np.float32)
                base_r = float(rr[isalt])
                for node in range(int(cfg.salt_nnode)):
                    # Create a few random circular shapes and blend
                    slice_r[:, :] += random_circular(
                        360,
                        r=base_r * (0.8 + 0.2 * node / max(cfg.salt_nnode - 1, 1)),
                        dr=0.25 * base_r,
                        alpha=0.3,
                        smooth=10.0,
                        seed=int(cfg.seed) * 31 + isalt * 100 + node,
                    )[:, None]
                slice_r /= float(cfg.salt_nnode)
                slice_r = gauss_filt(slice_r, (1.0, 3.0))

                # Path wiggle (simple)
                path_x = gauss_filt(rand_normal((n1,), seed=int(cfg.seed) * 41 + isalt), float(cfg.salt_path_variation) / 6.0)
                path_y = gauss_filt(rand_normal((n1,), seed=int(cfg.seed) * 42 + isalt), float(cfg.salt_path_variation) / 6.0)
                path_x = path_x / (np.max(np.abs(path_x)) + 1.0e-8) * float(cfg.salt_path_variation)
                path_y = path_y / (np.max(np.abs(path_y)) + 1.0e-8) * float(cfg.salt_path_variation)

                for i in range(max(n1 - nd - int(np.ceil(float(topz.max()))), 0), n1):
                    xcenter = float(cx[isalt]) + float(path_x[i])
                    ycenter = float(cy[isalt]) + float(path_y[i])
                    # Build angle grid for one depth slice
                    dx = (np.arange(n3, dtype=np.float32)[None, :] - xcenter).astype(np.float32)
                    dy = (np.arange(n2, dtype=np.float32)[:, None] - ycenter).astype(np.float32)
                    dist = np.sqrt(dx * dx + dy * dy).astype(np.float32)
                    ang = (np.arctan2(dx, dy + 1.0e-8) * (180.0 / np.pi) + 180.0).astype(np.int32) % 360
                    rad = slice_r[ang, i]
                    inside = (dist <= rad) & (i >= (n1 - nd - topz))
                    if np.any(inside):
                        vp[i, inside] = float(cfg.salt_vp)
                        rho[i, inside] = float(cfg.salt_rho)
                        salt[i, inside] = 1.0

            # If salt exists, wipe fault attributes inside salt like Fortran (for unconf=0)
            if fault is not None:
                mask = salt == 1.0
                fault = fault.copy()
                fault_dip = fault_dip.copy() if fault_dip is not None else None
                fault_strike = fault_strike.copy() if fault_strike is not None else None
                fault[mask] = 0.0
                if fault_dip is not None:
                    fault_dip[mask] = 0.0
                if fault_strike is not None:
                    fault_strike[mask] = 0.0

        # Generate seismic image (reflectivity + PSF), then optional noise like Fortran
        image = None
        if cfg.yn_image and cfg.wave != "":
            imp = (vp_full[:n1o, :, :] * rho).astype(np.float32)
            # Vertical reflectivity (n1-1, n2, n3) -> pad to n1
            rfc = (imp[1:, :, :] - imp[:-1, :, :]) / (imp[1:, :, :] + imp[:-1, :, :] + 1.0e-8)
            rfc = np.pad(rfc, ((0, 1), (0, 0), (0, 0)), mode="edge").astype(np.float32)
            k1, k2, k3 = self._make_wavelet_and_psf_kernels(n1, n2, n3)
            # Crop PSF application to the original size
            rfc_pad = np.zeros((n1, n2, n3), dtype=np.float32)
            rfc_pad[ne1 : ne1 + n1o, ne2 : ne2 + n2o, ne3 : ne3 + n3o] = rfc

            if cfg.noise_level != 0.0 and cfg.yn_conv_noise:
                image = self._add_image_noise(rfc_pad, seed=cfg.seed * 23)
            else:
                image = rfc_pad

            image = apply_separable_psf_same(image, k1, k2, k3).astype(np.float32)

            if cfg.noise_level != 0.0 and (not cfg.yn_conv_noise):
                image = self._add_image_noise(image, seed=cfg.seed * 23)

            # Final crop
            image = image[ne1 : ne1 + n1o, ne2 : ne2 + n2o, ne3 : ne3 + n3o].astype(np.float32)

        # Crop fault volumes if present
        if fault is not None:
            fault = fault[ne1 : ne1 + n1o, ne2 : ne2 + n2o, ne3 : ne3 + n3o].astype(np.float32)
            fault_dip = fault_dip[ne1 : ne1 + n1o, ne2 : ne2 + n2o, ne3 : ne3 + n3o].astype(np.float32)
            fault_strike = fault_strike[ne1 : ne1 + n1o, ne2 : ne2 + n2o, ne3 : ne3 + n3o].astype(np.float32)

        rgt = None
        facies = None

        return RGM3CurvedResult(
            vp=vp,
            rho=rho,
            image=image,
            fault=fault,
            fault_dip=fault_dip,
            fault_strike=fault_strike,
            salt=salt,
            rgt=rgt,
            facies=facies,
        )

    def _add_image_noise(self, image: ArrayF32, *, seed: int) -> ArrayF32:
        """
        Match Fortran noise injection in `generate_image` (normal/uniform/exp or wavenumber).
        """
        cfg = self.cfg
        n1, n2, n3 = image.shape

        if cfg.noise_type == "wavenumber":
            # Mirror Fortran `noise_wavenumber_3d`
            r = rand_uniform((n1, n2, n3), 0.0, 1.0, seed=seed)
            r = gauss_filt(r, cfg.noise_smooth)
            r = rescale(r, (0.0, 1.0))
            r = (r >= float(cfg.noise_level)).astype(np.float32)
            wt = np.fft.ifftn(np.fft.fftn(image) * r).real.astype(np.float32) - image
            return image + wt

        if cfg.noise_type in ("normal", "gaussian"):
            w = rand_normal((n1, n2, n3), seed=seed)
        elif cfg.noise_type == "uniform":
            w = rand_uniform((n1, n2, n3), -1.0, 1.0, seed=seed)
        elif cfg.noise_type == "exp":
            # symmetric exponential-ish noise
            u = rand_uniform((n1, n2, n3), 0.0, 1.0, seed=seed)
            w = (np.sign(u - 0.5) * (-np.log(np.maximum(1.0e-8, 1.0 - 2.0 * np.abs(u - 0.5))))).astype(np.float32)
        else:
            w = rand_normal((n1, n2, n3), seed=seed)

        w = gauss_filt(w, cfg.noise_smooth)
        w = (w - float(w.mean())).astype(np.float32)
        w = w / (float(np.max(np.abs(w))) + 1.0e-8)
        amp = float(cfg.noise_level) * float(np.max(np.abs(image)))
        return (image + w * amp).astype(np.float32)

    def _generate_unconformal(self) -> RGM3CurvedResult:
        """
        Simplified unconformity: generate two sedimentary units and merge with a Perlin surface.
        """
        cfg = self.cfg
        # Unit below (more deformed + optional salt)
        below = RGM3Curved(cfg)._generate_geological()

        # Unit above: fewer/no faults, smoother layering
        above_cfg = replace(
            cfg,
            yn_fault=False,
            nf=0,
            yn_salt=False,
            seed=cfg.seed * 7,
            lwv=abs(cfg.lwv / 2.0),
            lwh=abs(cfg.lwh / 2.0),
        )
        above = RGM3Curved(above_cfg)._generate_geological()

        n1, n2, n3 = int(cfg.n1), int(cfg.n2), int(cfg.n3)
        # Unconformity surface (depth index; larger -> higher in the volume)
        ufz = float(rand_uniform((1,), float(cfg.unconf_z[0]), float(cfg.unconf_z[1]), seed=cfg.seed * 31)[0]) * n1
        surf = FractalNoise2D(n1=n2, n2=n3, seed=cfg.seed * 41, octaves=5).generate()
        if cfg.unconf_smooth > 0:
            surf = gauss_filt(surf, (cfg.unconf_smooth, cfg.unconf_smooth))
        surf = rescale(surf, (0.0, float(rand_uniform((1,), float(cfg.unconf_height[0]), float(cfg.unconf_height[1]), seed=cfg.seed * 51)[0]))) + ufz

        vp = below.vp.copy()
        rho = below.rho.copy()
        fault = below.fault.copy() if below.fault is not None else None
        fault_dip = below.fault_dip.copy() if below.fault_dip is not None else None
        fault_strike = below.fault_strike.copy() if below.fault_strike is not None else None
        salt = below.salt.copy() if below.salt is not None else None

        # Merge: above overwrites where i < surf(j,k) (Fortran uses i starting at 1)
        for j in range(n2):
            for k in range(n3):
                cut = int(np.clip(np.floor(surf[j, k]), 0, n1))
                if cut <= 0:
                    continue
                vp[:cut, j, k] = above.vp[:cut, j, k]
                rho[:cut, j, k] = above.rho[:cut, j, k]
                if fault is not None and above.fault is not None:
                    fault[:cut, j, k] = above.fault[:cut, j, k]
                    fault_dip[:cut, j, k] = above.fault_dip[:cut, j, k]
                    fault_strike[:cut, j, k] = above.fault_strike[:cut, j, k]
                if salt is not None:
                    salt[:cut, j, k] = 0.0

        # Recompute image after merge
        image = None
        if cfg.yn_image and cfg.wave != "":
            imp = (vp * rho).astype(np.float32)
            rfc = (imp[1:, :, :] - imp[:-1, :, :]) / (imp[1:, :, :] + imp[:-1, :, :] + 1.0e-8)
            rfc = np.pad(rfc, ((0, 1), (0, 0), (0, 0)), mode="edge").astype(np.float32)
            k1, k2, k3 = self._make_wavelet_and_psf_kernels(n1, n2, n3)
            image = apply_separable_psf_same(rfc, k1, k2, k3).astype(np.float32)

        return RGM3CurvedResult(vp=vp, rho=rho, image=image, fault=fault, fault_dip=fault_dip, fault_strike=fault_strike, salt=salt)

