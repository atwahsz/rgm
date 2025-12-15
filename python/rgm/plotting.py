from __future__ import annotations

from typing import Literal

import numpy as np


def _normalize_for_cmap(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.clip((x - float(vmin)) / (float(vmax) - float(vmin) + 1.0e-8), 0.0, 1.0)


def plot_3_slices(
    volume: np.ndarray,
    out_path: str,
    *,
    kind: Literal["vp", "image", "dip", "strike"],
    cmap: str,
    vmin: float,
    vmax: float,
    xlabel: str = "X",
    ylabel: str = "Z",
) -> None:
    """
    Plot three Z-X slices at different Y indices (like the RGM doc figures).
    """
    import matplotlib.pyplot as plt

    vol = np.asarray(volume)
    n1, n2, n3 = vol.shape
    ys = [n2 // 4, n2 // 2, (3 * n2) // 4]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), constrained_layout=True)
    for ax, y in zip(axes, ys, strict=True):
        img = vol[:, y, :]
        im = ax.imshow(
            img,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            origin="upper",
        )
        ax.set_title(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
    if kind == "vp":
        cbar.set_label("P-wave Velocity (m/s)")
    elif kind == "image":
        cbar.set_label("Reflectivity")
    elif kind == "dip":
        cbar.set_label("Dip (degree)")
    elif kind == "strike":
        cbar.set_label("Strike (degree)")
    else:
        cbar.set_label(kind)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_cube_slices(
    volume: np.ndarray,
    out_path: str,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    salt: np.ndarray | None = None,
) -> None:
    """
    Plot a "cube" with three orthogonal slice planes (matplotlib mplot3d).

    This approximates the lower row panels in `doc/vp_faulted.jpg` and similar.
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    vol = np.asarray(volume, dtype=np.float32)
    n1, n2, n3 = vol.shape
    z0 = n1 // 2
    y0 = n2 // 2
    x0 = n3 // 2

    # Prepare coordinates
    X, Y = np.meshgrid(np.arange(n3), np.arange(n2))
    Xz, Zx = np.meshgrid(np.arange(n3), np.arange(n1))
    Yz, Zy = np.meshgrid(np.arange(n2), np.arange(n1))

    norm = lambda a: _normalize_for_cmap(a, vmin, vmax)
    cmap_obj = cm.get_cmap(cmap)

    fig = plt.figure(figsize=(13.5, 4), constrained_layout=True)
    axes = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]
    views = [(30, -60), (30, -35), (30, -15)]

    for ax, (elev, azim) in zip(axes, views, strict=True):
        # XY plane at z=z0
        c_xy = cmap_obj(norm(vol[z0, :, :]))
        ax.plot_surface(X, Y, np.full_like(X, z0), rstride=2, cstride=2, facecolors=c_xy, shade=False, linewidth=0)

        # XZ plane at y=y0
        c_xz = cmap_obj(norm(vol[:, y0, :]))
        ax.plot_surface(Xz, np.full_like(Xz, y0), Zx, rstride=2, cstride=2, facecolors=c_xz, shade=False, linewidth=0)

        # YZ plane at x=x0
        c_yz = cmap_obj(norm(vol[:, :, x0]))
        ax.plot_surface(np.full_like(Yz, x0), Yz, Zy, rstride=2, cstride=2, facecolors=c_yz, shade=False, linewidth=0)

        # Optional salt overlay (coarse voxels)
        if salt is not None:
            sm = np.asarray(salt, dtype=np.float32)
            step = max(1, min(n1, n2, n3) // 64)
            vv = sm[::step, ::step, ::step] > 0.5
            if np.any(vv):
                ax.voxels(vv, facecolors=(0.75, 0.95, 0.65, 0.4), edgecolor=None)

        ax.set_xlim(0, n3 - 1)
        ax.set_ylim(0, n2 - 1)
        ax.set_zlim(n1 - 1, 0)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=elev, azim=azim)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_doc_style_6panel(
    volume: np.ndarray,
    out_path: str,
    *,
    kind: Literal["vp", "image", "dip", "strike"],
    cmap: str,
    vmin: float,
    vmax: float,
    salt: np.ndarray | None = None,
) -> None:
    """
    Create a 2x3 panel figure:
    - top row: three Z-X slices at different Y
    - bottom row: three "cube slice" views

    This matches the overall layout of the figures in `doc/`.
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    vol = np.asarray(volume, dtype=np.float32)
    n1, n2, n3 = vol.shape
    ys = [n2 // 4, n2 // 2, (3 * n2) // 4]

    fig = plt.figure(figsize=(13.5, 7.5), constrained_layout=True)

    # --- Top row: 2D slices
    ims: list[object] = []
    for i, y in enumerate(ys):
        ax = fig.add_subplot(2, 3, i + 1)
        img = vol[:, y, :]
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", origin="upper")
        ims.append(im)
        ax.set_title("X")
        ax.set_ylabel("Z")

    cbar = fig.colorbar(ims[-1], ax=[fig.axes[0], fig.axes[1], fig.axes[2]], shrink=0.85, pad=0.02)
    if kind == "vp":
        cbar.set_label("P-wave Velocity (m/s)")
    elif kind == "image":
        cbar.set_label("Reflectivity")
    elif kind == "dip":
        cbar.set_label("Dip (degree)")
    elif kind == "strike":
        cbar.set_label("Strike (degree)")
    else:
        cbar.set_label(kind)

    # --- Bottom row: 3D cube slices
    z0 = n1 // 2
    y0 = n2 // 2
    x0 = n3 // 2
    X, Y = np.meshgrid(np.arange(n3), np.arange(n2))
    Xz, Zx = np.meshgrid(np.arange(n3), np.arange(n1))
    Yz, Zy = np.meshgrid(np.arange(n2), np.arange(n1))
    cmap_obj = cm.get_cmap(cmap)

    views = [(30, -60), (30, -35), (30, -15)]
    for i, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 3, 4 + i, projection="3d")
        c_xy = cmap_obj(_normalize_for_cmap(vol[z0, :, :], vmin, vmax))
        ax.plot_surface(X, Y, np.full_like(X, z0), rstride=2, cstride=2, facecolors=c_xy, shade=False, linewidth=0)

        c_xz = cmap_obj(_normalize_for_cmap(vol[:, y0, :], vmin, vmax))
        ax.plot_surface(Xz, np.full_like(Xz, y0), Zx, rstride=2, cstride=2, facecolors=c_xz, shade=False, linewidth=0)

        c_yz = cmap_obj(_normalize_for_cmap(vol[:, :, x0], vmin, vmax))
        ax.plot_surface(np.full_like(Yz, x0), Yz, Zy, rstride=2, cstride=2, facecolors=c_yz, shade=False, linewidth=0)

        if salt is not None:
            sm = np.asarray(salt, dtype=np.float32)
            step = max(1, min(n1, n2, n3) // 64)
            vv = sm[::step, ::step, ::step] > 0.5
            if np.any(vv):
                ax.voxels(vv, facecolors=(0.75, 0.95, 0.65, 0.4), edgecolor=None)

        ax.set_xlim(0, n3 - 1)
        ax.set_ylim(0, n2 - 1)
        ax.set_zlim(n1 - 1, 0)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=elev, azim=azim)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)

