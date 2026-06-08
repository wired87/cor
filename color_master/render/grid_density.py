"""
Prompt (2026-06): White-to-blue grid density main animation.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from color_master.types import PlotQuality
from color_master.render.helpers import _scatter3d_facecolors, check_plot_quality
def render_grid_white_to_blue(
    volume: np.ndarray,
    output_path: Path,
    quality: PlotQuality,
    *,
    amount_nodes: int,
    title: str = "Grid | feature density (white -> blue)",
) -> None:
    """White-grid main animation: per-grid-point dot fades from white to blue with summed feature value.

    `volume` has shape (T, N, N, N) and contains, for each timestep t and each
    integer grid coordinate (x, y, z), the sum of every feature value that
    linearly folds onto that grid point across all fields / equations. The
    figure background and axes are pure white; each grid point is rendered as
    a fixed scatter dot at its integer coordinate, and its colour is a
    `lerp(white, deep_blue, norm)` where `norm = volume[t] / global_max` is
    normalized once over the whole volume so brightness is comparable across
    timesteps. At t=0 (or any time a point's summed value is 0) the dot is
    pure white and visually indistinguishable from the background — exactly
    the "start = white grid with white dots" requirement.
    """
    # CHAR: defensive: empty / wrong-rank volume -> no animation, no crash
    if volume.ndim != 4 or volume.size == 0:
        print(f"[grid-w2b] skip: invalid volume shape {volume.shape}")
        return
    # CHAR: enforce cubic grid; we always render a 3D point cloud regardless of sim dims
    T, Nx, Ny, Nz = volume.shape
    if not (Nx == Ny == Nz == amount_nodes):
        print(f"[grid-w2b] skip: volume {volume.shape} != ({T},{amount_nodes}^3)")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # CHAR: global max so dot colour is comparable across all frames; clamp non-finite / 0
    g_max = float(np.nanmax(volume)) if np.isfinite(np.nanmax(volume)) else 0.0
    if g_max <= 0.0:
        g_max = 1.0

    # CHAR: fixed integer-grid positions — identical for every frame so dots only change colour
    xs_g, ys_g, zs_g = np.meshgrid(
        np.arange(amount_nodes), np.arange(amount_nodes), np.arange(amount_nodes), indexing="ij"
    )
    xs = xs_g.ravel().astype(float)
    ys = ys_g.ravel().astype(float)
    zs = zs_g.ravel().astype(float)
    n_pts = amount_nodes ** 3

    # CHAR: white -> deep-blue endpoints; alpha stays 1 so a pure-white dot is invisible on white
    WHITE = np.array([1.0, 1.0, 1.0], dtype=float)
    BLUE = np.array([0.05, 0.35, 1.0], dtype=float)

    use_rast = check_plot_quality("grid_w2b", n_pts, quality)

    # CHAR: white figure + axes (overrides _setup_axes' dark theme just for this animation)
    fig = plt.figure(figsize=(10, 7), dpi=quality.dpi)
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")

    def _setup_white_axes(ax_: Any) -> None:
        ax_.set_facecolor("white")
        ax_.grid(False)
        ax_.set_xticks([])
        ax_.set_yticks([])
        ax_.set_zticks([])
        # CHAR: hide pane fills + edges so nothing competes with the white "canvas"
        ax_.xaxis.pane.fill = False
        ax_.yaxis.pane.fill = False
        ax_.zaxis.pane.fill = False
        for pane_axis in (ax_.xaxis, ax_.yaxis, ax_.zaxis):
            pane_axis.pane.set_edgecolor("white")
        ax_.set_xlim(-0.5, amount_nodes - 0.5)
        ax_.set_ylim(-0.5, amount_nodes - 0.5)
        ax_.set_zlim(-0.5, amount_nodes - 0.5)

    point_size = max(quality.point_size_anim * 1.6, 60.0)

    def update(frame_idx: int) -> list[Any]:
        ax.cla()
        _setup_white_axes(ax)

        # CHAR: per-frame normalized values in [0,1] -> drive the white->blue lerp
        v = volume[frame_idx].ravel().astype(float)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        norm = np.clip(v / g_max, 0.0, 1.0)

        # CHAR: build per-dot RGBA. alpha=1 keeps norm=0 dots opaque white -> invisible on white bg
        rgba = np.zeros((n_pts, 4), dtype=float)
        rgba[:, :3] = (1.0 - norm[:, None]) * WHITE + norm[:, None] * BLUE
        rgba[:, 3] = 1.0

        _scatter3d_facecolors(
            ax,
            xs,
            ys,
            zs,
            rgba,
            s=point_size,
            depthshade=False,
            rasterized=use_rast,
        )

        ax.view_init(elev=22, azim=30 + (frame_idx * 1.5))

        # CHAR: dark text on white bg for contrast (overrides the dark-theme metrics block)
        fig.suptitle(title, color="#0B0B10", fontsize=11)
        fig.text(
            0.01,
            0.02,
            f"nodes={amount_nodes}^3 grid_pts={n_pts}",
            color="#3a3a3a",
            fontsize=9,
            va="bottom",
            ha="left",
        )
        fig.text(
            0.99,
            0.02,
            f"t {frame_idx + 1}/{T}  max={g_max:.3g}",
            color="#3a3a3a",
            fontsize=9,
            va="bottom",
            ha="right",
        )
        return []

    # CHAR: use a slightly slower fps so the white->blue transition stays readable
    ani = animation.FuncAnimation(fig, update, frames=T, interval=120, blit=False)
    ani.save(output_path, writer=animation.PillowWriter(fps=10))
    plt.close(fig)
    print(f"[grid-w2b] wrote {output_path}  (T={T}, N={amount_nodes}, max={g_max:.3g})")


