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
from .perlin import FractalNoise2D, random_circular
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


@dataclass(frozen=True, slots=True)
class RGM3CurvedResult:
    vp: ArrayF32
    rho: ArrayF32
    image: ArrayF32 | None
    fault: ArrayF32 | None
    fault_dip: ArrayF32 | None
    fault_strike: ArrayF32 | None
    salt: ArrayF32 | None


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

        k1 = (k1 / (norm2(k1) + 1.0e-12)).astype(np.float32)
        ps1 = (ps1 / (norm2(ps1) + 1.0e-12)).astype(np.float32)
        ps2 = (ps2 / (norm2(ps2) + 1.0e-12)).astype(np.float32)
        ps3 = (ps3 / (norm2(ps3) + 1.0e-12)).astype(np.float32)

        k1 = (k1 * ps1).astype(np.float32)
        k1 = (k1 / (norm2(k1) + 1.0e-12)).astype(np.float32)
        return k1, ps2, ps3

    def _generate_reflector_surface(
        self,
        n2: int,
        n3: int,
        shape: Literal["random", "perlin", "gaussian", "cauchy"],
        smooth: float,
        height: tuple[float, float],
        slope_yx: tuple[float, float],
        seed_mul: int,
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
            mu2 = rand_uniform((cfg.ng,), 0.0, float(n2 - 1), seed=sd).astype(np.float32)
            mu3 = rand_uniform((cfg.ng,), 0.0, float(n3 - 1), seed=sd + 1).astype(np.float32)
            sg2 = rand_uniform((cfg.ng,), float(cfg.refl_sigma2[0]), float(cfg.refl_sigma2[1]), seed=sd + 2).astype(
                np.float32
            )
            sg3 = rand_uniform((cfg.ng,), float(cfg.refl_sigma3[0]), float(cfg.refl_sigma3[1]), seed=sd + 3).astype(
                np.float32
            )
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

        # Scale to height and add slopes
        r = rescale(r, height)
        sy, sx = float(slope_yx[0]), float(slope_yx[1])
        y = (np.arange(n2, dtype=np.float32) / max(n2, 1))[:, None]
        x = (np.arange(n3, dtype=np.float32) / max(n3, 1))[None, :]
        r = (r + y * sy + x * sx).astype(np.float32)
        r = (r - float(r.mean())).astype(np.float32)
        return r

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

        n1 = int(cfg.n1)
        n2 = int(cfg.n2)
        n3 = int(cfg.n3)

        # Reflector surfaces (bottom + top)
        r = self._generate_reflector_surface(
            n2=n2,
            n3=n3,
            shape=cfg.refl_shape,
            smooth=cfg.refl_smooth,
            height=cfg.refl_height,
            slope_yx=cfg.refl_slope_yx,
            seed_mul=5,
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
            )

        # Like Fortran: bottom around 0, top around n1
        r = (r - float(r.mean())).astype(np.float32)
        rt = (rt - float(rt.mean()) + float(n1)).astype(np.float32)

        nl = int(cfg.nl)
        tgrid = np.linspace(0.0, 1.0, nl, dtype=np.float32)
        x1 = 1.0 / max(n1, 1)  # small ramp near bottom
        x2 = 1.0 - 1.0 / max(n1, 1)
        lz = interp1_linear_4pt(tgrid, r, r, rt, rt, x1=x1, x2=x2)  # (nl,n2,n3)

        # Layer thickness variation (vertical)
        dlz = np.diff(lz, axis=0, prepend=lz[:1]).astype(np.float32)
        plw = rand_uniform((nl,), 1.0 - float(cfg.lwv), 1.0 + float(cfg.lwv), seed=cfg.seed * 7).astype(np.float32)
        dlz = (dlz * plw[:, None, None]).astype(np.float32)
        csum = np.cumsum(dlz, axis=0).astype(np.float32)
        scale = (rt - r) / (csum[-1] + 1.0e-8)
        lz = (r[None, :, :] + csum * scale[None, :, :]).astype(np.float32)

        # Horizontal layer thickness variation (optional, simplified)
        if cfg.lwh > 0.0:
            pxy = np.zeros_like(lz)
            for li in range(nl - 1):
                thick = float((lz[li + 1, 0, 0] - lz[li, 0, 0]) * cfg.lwh * 2.0)
                rr = gauss_filt(rand_uniform((n2, n3), 0.0, 1.0, seed=cfg.seed * 9 - li), (cfg.secondary_refl_smooth, cfg.secondary_refl_smooth))
                rr = rescale(rr, (-thick, thick))
                pxy[li, :, :] = rr
            dlz = np.diff(lz + pxy, axis=0, prepend=(lz + pxy)[:1]).astype(np.float32)
            dlz = np.maximum(dlz, 1.0e-6).astype(np.float32)
            csum = np.cumsum(dlz, axis=0).astype(np.float32)
            scale = (rt - r) / (csum[-1] + 1.0e-8)
            lz = (r[None, :, :] + csum * scale[None, :, :]).astype(np.float32)

        # Build layered velocity model by discretizing layers into voxel indices
        vv = linspace(float(cfg.vmax), float(cfg.vmin), nl - 1) + rand_uniform((nl - 1,), 0.0, float(cfg.delta_v), seed=cfg.seed * 11)
        vv = vv.astype(np.float32)

        w = np.zeros((n1 + 1, n2, n3), dtype=np.float32)
        # For each column, fill layers (fast enough for n2=n3=128)
        z_from_bottom = (n1 - 1 - np.arange(n1 + 1, dtype=np.int32)).astype(np.float32)  # length n1+1
        for j in range(n2):
            for k in range(n3):
                bounds = lz[:, j, k]  # nl
                # Ensure monotonic
                bounds = np.maximum.accumulate(bounds).astype(np.float32)
                # Assign layer index by searching bounds
                idx = np.searchsorted(bounds, z_from_bottom, side="right") - 1
                idx = np.clip(idx, 0, nl - 2)
                w[:, j, k] = vv[idx]

        # Apply faults
        fault = fault_dip = fault_strike = None
        dips, dip, strike, rake, disp = self._fault_params(int(cfg.nf), n1 + 1, n2, n3)
        if cfg.yn_fault and cfg.nf > 0:
            fid, fdip, fstr, w = self._apply_faults(w, dips, dip, strike, rake, disp, seed=int(cfg.seed) * 17)
            fault = fid[:n1, :, :].astype(np.float32)
            fault_dip = fdip[:n1, :, :].astype(np.float32)
            fault_strike = fstr[:n1, :, :].astype(np.float32)

        # Rescale to [vmin, vmax] and compute rho
        vp = rescale(w[:n1, :, :], (cfg.vmin, cfg.vmax)).astype(np.float32)
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

        # Generate seismic image (reflectivity + PSF)
        image = None
        if cfg.yn_image and cfg.wave != "":
            imp = (vp * rho).astype(np.float32)
            # Vertical reflectivity (n1-1, n2, n3) -> pad to n1
            rfc = (imp[1:, :, :] - imp[:-1, :, :]) / (imp[1:, :, :] + imp[:-1, :, :] + 1.0e-8)
            rfc = np.pad(rfc, ((0, 1), (0, 0), (0, 0)), mode="edge").astype(np.float32)
            k1, k2, k3 = self._make_wavelet_and_psf_kernels(n1, n2, n3)
            image = apply_separable_psf_same(rfc, k1, k2, k3).astype(np.float32)

        return RGM3CurvedResult(vp=vp, rho=rho, image=image, fault=fault, fault_dip=fault_dip, fault_strike=fault_strike, salt=salt)

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

