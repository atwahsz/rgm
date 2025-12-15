from __future__ import annotations

"""
Generate three random 3D models and doc-style figures.

This script produces Python-generated figures similar in layout to those in `doc/`.
Outputs are written into `doc/` with `py_` prefixes to avoid overwriting the
repository's original, approved figures.
"""

import os
import sys
from dataclasses import replace

import numpy as np

# Allow "python/scripts/..." to import from "python/rgm"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PY_ROOT = os.path.join(REPO_ROOT, "python")
sys.path.insert(0, PY_ROOT)

from rgm import RGM3Curved, RGM3CurvedConfig  # noqa: E402
from rgm.plotting import plot_doc_style_6panel  # noqa: E402


def main() -> None:
    out_dir = os.path.join(REPO_ROOT, "doc")
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------
    # (1) Faulted Vp + Image
    # -------------------------
    cfg_fault = RGM3CurvedConfig(
        n1=200,
        n2=128,
        n3=128,
        seed=123,
        yn_fault=True,
        nf=7,
        yn_salt=False,
        unconf=0,
        wave="ricker",
        f0=150.0,
        psf_sigma=(10.0, 1.0, 1.0),
        refl_shape="perlin",
        refl_shape_top="perlin",
        refl_smooth=3.0,
        refl_smooth_top=3.0,
    )
    res_fault = RGM3Curved(cfg_fault).generate()
    plot_doc_style_6panel(
        res_fault.vp,
        os.path.join(out_dir, "py_vp_faulted.jpg"),
        kind="vp",
        cmap="jet",
        vmin=cfg_fault.vmin,
        vmax=cfg_fault.vmax,
    )
    if res_fault.image is not None:
        plot_doc_style_6panel(
            res_fault.image,
            os.path.join(out_dir, "py_image_faulted.jpg"),
            kind="image",
            cmap="gray",
            vmin=float(np.percentile(res_fault.image, 2)),
            vmax=float(np.percentile(res_fault.image, 98)),
        )
    if res_fault.fault_dip is not None:
        plot_doc_style_6panel(
            res_fault.fault_dip,
            os.path.join(out_dir, "py_fault_dip.jpg"),
            kind="dip",
            cmap="jet",
            vmin=30.0,
            vmax=150.0,
        )

    # -------------------------
    # (2) Salt Vp
    # -------------------------
    cfg_salt = replace(
        cfg_fault,
        seed=13579,
        yn_fault=False,
        nf=0,
        yn_salt=True,
        nsalt=3,
        salt_radius=(15.0, 40.0),
        salt_top_z=(0.4, 0.6),
        salt_top_height=20.0,
        salt_path_variation=20.0,
        salt_nnode=10,
        vmin=2000.0,
        vmax=5000.0,
    )
    res_salt = RGM3Curved(cfg_salt).generate()
    plot_doc_style_6panel(
        res_salt.vp,
        os.path.join(out_dir, "py_vp_salt.jpg"),
        kind="vp",
        cmap="jet",
        vmin=cfg_salt.vmin,
        vmax=cfg_salt.vmax,
        salt=res_salt.salt,
    )

    # -------------------------
    # (3) Unconformity Image
    # -------------------------
    cfg_unconf = replace(
        cfg_fault,
        seed=4566,
        unconf=1,
        yn_salt=True,
        nsalt=2,
        salt_radius=(25.0, 55.0),
        salt_top_z=(0.4, 0.6),
        salt_top_height=30.0,
        salt_path_variation=20.0,
        unconf_z=(0.1, 0.3),
        unconf_height=(10.0, 20.0),
        unconf_smooth=20.0,
    )
    res_unconf = RGM3Curved(cfg_unconf).generate()
    if res_unconf.image is not None:
        plot_doc_style_6panel(
            res_unconf.image,
            os.path.join(out_dir, "py_image_unconf.jpg"),
            kind="image",
            cmap="gray",
            vmin=float(np.percentile(res_unconf.image, 2)),
            vmax=float(np.percentile(res_unconf.image, 98)),
            salt=res_unconf.salt,
        )


if __name__ == "__main__":
    main()

