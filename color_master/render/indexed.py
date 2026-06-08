"""
Prompt (2026-06): Indexed controller-param unified 3D animation.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Sequence
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import to_rgb
from color_master.types import FramePoints, PlotQuality, QUALITY_HIGH, QUALITY_LIGHT
from color_master.render.helpers import (
    _alpha_from_values,
    _border_metrics,
    _scatter3d_facecolors,
    _setup_axes,
    check_plot_quality,
)


def _quality_from_preset(preset: str) -> PlotQuality:
    """Map workflow quality preset name to PlotQuality settings."""
    p = (preset or "default").lower()
    if p == "light":
        return QUALITY_LIGHT
    if p == "high":
        return QUALITY_HIGH
    return PlotQuality()


def render_indexed_unified_animation(
    frames: Sequence[FramePoints],
    labels: Sequence[str],
    value_range: tuple[float, float],
    output_path: Path,
    quality: PlotQuality,
    *,
    n_params: int,
) -> None:
    """
    One GIF: all indexed param points per timestep. X=param index, Y=value, Z=module/field band.
    Per-point color cycles tab20; alpha from |value| in global [min,max].
    """
    if not frames:
        return
    print(f"[render-indexed] n_params={n_params} frames={len(frames)} -> {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    val_min, val_max = value_range
    if not (np.isfinite(val_min) and np.isfinite(val_max)):
        val_min, val_max = 0.0, 1.0
    n_pts = n_params
    use_rast = check_plot_quality("indexed", n_pts, quality)
    n_show = min(18, n_pts)
    label_preview = ", ".join(labels[:n_show]) + ("..." if n_pts > n_show else "")

    fig = plt.figure(figsize=(11, 7), dpi=quality.dpi)
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.get_cmap("tab20", 20)

    def update(frame_idx: int) -> list[Any]:
        ax.cla()
        _setup_axes(fig, ax)
        ax.set_xlabel("param index (ctlr order)", color="#7E869B", fontsize=8)
        ax.set_ylabel("value", color="#7E869B", fontsize=8)
        ax.set_zlabel("module*band+field (z band)", color="#7E869B", fontsize=8)
        fr = frames[frame_idx]
        n_draw = int(len(fr.x))
        if n_draw == 0:
            return []
        pr = np.array([to_rgb(cmap(int(i) % 20)[:3]) for i in range(n_draw)], dtype=float)
        a = _alpha_from_values(
            fr.value[:n_draw], val_min, val_max, low=quality.alpha_low, high=quality.alpha_high
        )
        rgba = np.zeros((n_draw, 4), dtype=float)
        rgba[:, :3] = pr
        rgba[:, 3] = a
        _scatter3d_facecolors(
            ax,
            fr.x,
            fr.y,
            fr.z,
            rgba,
            s=max(8.0, quality.point_size_anim),
            depthshade=False,
            rasterized=use_rast,
        )
        ax.view_init(elev=18, azim=32 + (frame_idx * 1.1))
        fig.suptitle(
            "Indexed 3D | all ctlr param points (single animation)",
            color="#D6D8E6",
            fontsize=11,
        )
        fig.text(0.01, 0.96, label_preview, color="#9AA0B8", fontsize=6, va="top", ha="left")
        _border_metrics(
            fig,
            f"3D indexed | t={frame_idx + 1}/{len(frames)}",
            frame_idx,
            len(frames),
            n_params,
            0,
            val_min,
            val_max,
        )
        return []

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=90, blit=False)
    ani.save(output_path, writer=animation.PillowWriter(fps=11))
    plt.close(fig)
