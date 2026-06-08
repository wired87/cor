"""
Prompt (2026-06): Per-key and combined 3D time-series renders.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import to_rgb
from color_master.types import PlotQuality, QUALITY_HIGH, QUALITY_LIGHT
from color_master.render.helpers import (
    _alpha_from_values,
    _border_metrics,
    _frame_from_timestep,
    _normalize_series,
    _rgba,
    _scatter3d_facecolors,
    _setup_axes,
    check_plot_quality,
)
from color_master.types import FramePoints
def render_key_static(
    key: str,
    frames: list[FramePoints],
    value_range: tuple[float, float],
    color: tuple[float, float, float],
    output_path: Path,
    amount_nodes: int,
    dims: int,
    quality: PlotQuality,
) -> None:
    if not frames:
        return

    print(f"[render-static] key={key} -> {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    use_rasterized = check_plot_quality(key, amount_nodes, quality)

    fig = plt.figure(figsize=(9, 6), dpi=quality.dpi)
    ax = fig.add_subplot(111, projection="3d")
    _setup_axes(fig, ax)

    frame = frames[-1]
    val_min, val_max = value_range
    alpha = _alpha_from_values(frame.value, val_min, val_max, low=quality.alpha_low, high=quality.alpha_high)
    _scatter3d_facecolors(
        ax,
        frame.x,
        frame.y,
        frame.z,
        _rgba(color, alpha),
        s=quality.point_size_static,
        rasterized=use_rasterized,
    )
    ax.view_init(elev=27, azim=48)
    _border_metrics(fig, f"3D Static View | {key}", len(frames) - 1, len(frames), amount_nodes, dims, val_min, val_max)

    fig.tight_layout()
    fig.savefig(output_path, facecolor=fig.get_facecolor(), dpi=quality.dpi)
    plt.close(fig)


def render_key_animation(
    key: str,
    frames: list[FramePoints],
    value_range: tuple[float, float],
    color: tuple[float, float, float],
    output_path: Path,
    amount_nodes: int,
    dims: int,
    quality: PlotQuality,
) -> None:
    if not frames:
        return

    #print(f"[render-animation] key={key} -> {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    val_min, val_max = value_range
    use_rasterized = check_plot_quality(key, amount_nodes, quality)

    fig = plt.figure(figsize=(9, 6), dpi=quality.dpi)
    ax = fig.add_subplot(111, projection="3d")

    def update(frame_idx: int) -> list[Any]:
        ax.cla()
        _setup_axes(fig, ax)

        current = frames[frame_idx]
        alpha = _alpha_from_values(current.value, val_min, val_max, low=quality.alpha_low, high=quality.alpha_high)
        _scatter3d_facecolors(
            ax,
            current.x,
            current.y,
            current.z,
            _rgba(color, alpha),
            s=quality.point_size_anim,
            depthshade=True,
            rasterized=use_rasterized,
        )

        ax.view_init(elev=26, azim=36 + (frame_idx * 2.1))
        _border_metrics(
            fig,
            f"3D Time Series | {key}",
            frame_idx,
            len(frames),
            amount_nodes,
            dims,
            val_min,
            val_max,
        )
        return []

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=85, blit=False)
    ani.save(output_path, writer=animation.PillowWriter(fps=12))
    plt.close(fig)


def render_environment_animation(
    all_frames: dict[str, list[FramePoints]],
    value_ranges: dict[str, tuple[float, float]],
    colors: dict[str, tuple[float, float, float]],
    output_path: Path,
    amount_nodes: int,
    dims: int,
    quality: PlotQuality,
) -> None:
    if not all_frames:
        return

    max_frames = max(len(series) for series in all_frames.values())
    total_points_per_frame = amount_nodes * len(all_frames)
    print(f"[render-combined] keys={len(all_frames)} frames={max_frames} -> {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    use_rasterized = check_plot_quality("combined", total_points_per_frame, quality)

    gvals = [vr[0] for vr in value_ranges.values()] + [vr[1] for vr in value_ranges.values()]
    gvals_f = [float(x) for x in gvals if np.isfinite(x)]
    if gvals_f:
        global_min = min(gvals_f)
        global_max = max(gvals_f)
    else:
        global_min, global_max = 0.0, 1.0
    if np.isclose(global_min, global_max):
        global_max = global_min + 1.0

    fig = plt.figure(figsize=(10, 7), dpi=quality.dpi)
    ax = fig.add_subplot(111, projection="3d")

    def update(frame_idx: int) -> list[Any]:
        ax.cla()
        _setup_axes(fig, ax)

        for key, series in all_frames.items():
            if not series:
                continue

            local_idx = frame_idx % len(series)
            frame = series[local_idx]
            val_min, val_max = value_ranges[key]
            alpha = _alpha_from_values(frame.value, val_min, val_max, low=quality.alpha_low, high=quality.alpha_high)
            _scatter3d_facecolors(
                ax,
                frame.x,
                frame.y,
                frame.z,
                _rgba(colors[key], alpha),
                s=quality.point_size_combined,
                depthshade=True,
                rasterized=use_rasterized,
                label=key,
            )

        ax.view_init(elev=24, azim=30 + (frame_idx * 1.7))
        legend = ax.legend(loc="upper left", fontsize=8, frameon=False)
        for txt in legend.get_texts():
            txt.set_color("#CDD1E0")

        _border_metrics(
            fig,
            "3D Environment Animation | All Keys",
            frame_idx,
            max_frames,
            amount_nodes,
            dims,
            global_min,
            global_max,
        )
        return []

    ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=95, blit=False)
    ani.save(output_path, writer=animation.PillowWriter(fps=11))
    plt.close(fig)


def build_3d_time_series_visualization(
    data: dict[str, list[Any]],
    amount_nodes: int = 28,
    dims: int = 360,
    output_dir: str = "output_dir",
    quality: PlotQuality | None = None,
    quality_preset: str = "default",
) -> None:
    if quality is None:
        quality = {"default": PlotQuality(), "light": QUALITY_LIGHT, "high": QUALITY_HIGH}.get(
            quality_preset, PlotQuality()
        )
    print("[viz] timeseries start")
    print(f"[viz] amount_nodes={amount_nodes} dims={dims} output_dir={output_dir}")
    out_root = Path(output_dir)
    per_key_static_dir = out_root / "per_key_static"
    per_key_animation_dir = out_root / "per_key_animation"
    combined_dir = out_root / "combined"
    per_key_static_dir.mkdir(parents=True, exist_ok=True)
    per_key_animation_dir.mkdir(parents=True, exist_ok=True)
    combined_dir.mkdir(parents=True, exist_ok=True)
    normalized, value_ranges = _normalize_series(data, amount_nodes=amount_nodes, dims=dims)
    keys = list(normalized.keys())
    cmap = plt.get_cmap("tab10", max(3, len(keys)))
    colors = {key: to_rgb(cmap(idx)[:3]) for idx, key in enumerate(keys)}
    for key in keys:
        frames = normalized[key]
        render_key_static(
            key=key, frames=frames, value_range=value_ranges[key], color=colors[key],
            output_path=per_key_static_dir / f"{key}_3d.png",
            amount_nodes=amount_nodes, dims=dims, quality=quality,
        )
        render_key_animation(
            key=key, frames=frames, value_range=value_ranges[key], color=colors[key],
            output_path=per_key_animation_dir / f"{key}_3d.gif",
            amount_nodes=amount_nodes, dims=dims, quality=quality,
        )
    render_environment_animation(
        all_frames=normalized, value_ranges=value_ranges, colors=colors,
        output_path=combined_dir / "environment_3d.gif",
        amount_nodes=amount_nodes, dims=dims, quality=quality,
    )
    print("[viz] timeseries done")
