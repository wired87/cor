from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import to_rgb


@dataclass
class FramePoints:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    value: np.ndarray


@dataclass
class PlotQuality:
    dpi: int = 300
    min_dpi: int = 200
    point_size_static: float = 40
    point_size_anim: float = 32
    point_size_combined: float = 24
    alpha_low: float = 0.15
    alpha_high: float = 0.95
    rasterize_threshold: int = 500


QUALITY_LIGHT = PlotQuality(dpi=150, point_size_static=35, point_size_anim=28, point_size_combined=20)
QUALITY_HIGH = PlotQuality(dpi=400, point_size_static=45, point_size_anim=36, point_size_combined=28)


def check_plot_quality(key: str, n_points: int, quality: PlotQuality) -> bool:
    """Validate quality settings and return whether to use rasterized=True."""
    use_rasterized = n_points >= quality.rasterize_threshold
    print(f"[quality] key={key} points={n_points} dpi={quality.dpi} rasterized={use_rasterized}")
    if n_points < 100:
        print(f"[quality] warn: low point count ({n_points}) may reduce detail")
    if quality.dpi < quality.min_dpi:
        print(f"[quality] warn: dpi={quality.dpi} below min_dpi={quality.min_dpi}")
    return use_rasterized


def _flatten_numeric(payload: Any) -> list[complex]:
    if payload is None:
        return []

    if isinstance(payload, np.ndarray):
        flat = payload.ravel()
        return [complex(v) for v in flat]

    if np.isscalar(payload):
        return [complex(payload)]

    if isinstance(payload, (list, tuple, set)):
        out: list[complex] = []
        for item in payload:
            out.extend(_flatten_numeric(item))
        return out

    if isinstance(payload, dict):
        out: list[complex] = []
        for item in payload.values():
            out.extend(_flatten_numeric(item))
        return out

    try:
        return [complex(payload)]
    except (TypeError, ValueError):
        print("Err color_master.main::_flatten_numeric | handler_line=73 | (TypeError, ValueError) handler triggered")
        print("[exception] color_master.main._flatten_numeric: caught (TypeError, ValueError)")
        return []


def _frame_from_timestep(timestep: Any, amount_nodes: int, dims: int) -> FramePoints:
    raw = _flatten_numeric(timestep)
    if not raw:
        raw = [0j]

    n = len(raw)
    dims_eff = max(1, int(dims))
    stride_y = max(1, dims_eff // 3)
    stride_z = max(2, (2 * dims_eff) // 3)

    x = np.zeros(amount_nodes, dtype=float)
    y = np.zeros(amount_nodes, dtype=float)
    z = np.zeros(amount_nodes, dtype=float)
    value = np.zeros(amount_nodes, dtype=float)

    for node_idx in range(amount_nodes):
        base = (node_idx * dims_eff) % n
        cx = raw[base]
        cy = raw[(base + stride_y) % n]
        cz = raw[(base + stride_z) % n]

        x[node_idx] = float(np.real(cx))
        y[node_idx] = float(np.imag(cy) if np.imag(cy) != 0 else np.real(cy))
        z[node_idx] = float(np.abs(cz))
        value[node_idx] = (np.abs(cx) + np.abs(cy) + np.abs(cz)) / 3.0

    return FramePoints(x=x, y=y, z=z, value=value)


def _normalize_series(
    data: dict[str, list[Any]], amount_nodes: int, dims: int
) -> tuple[dict[str, list[FramePoints]], dict[str, tuple[float, float]]]:
    normalized: dict[str, list[FramePoints]] = {}
    ranges: dict[str, tuple[float, float]] = {}

    for key, series in data.items():
        print(f"[normalize] key={key} timesteps={len(series)}")
        frames = [_frame_from_timestep(timestep, amount_nodes, dims) for timestep in series]
        normalized[key] = frames

        all_values = np.concatenate([f.value for f in frames]) if frames else np.array([0.0])
        val_min = float(np.min(all_values))
        val_max = float(np.max(all_values))
        if np.isclose(val_min, val_max):
            val_max = val_min + 1.0
        ranges[key] = (val_min, val_max)
        print(f"[normalize] key={key} range=({val_min:.4f}, {val_max:.4f})")

    return normalized, ranges


def _alpha_from_values(
    values: np.ndarray, val_min: float, val_max: float, low: float = 0.10, high: float = 0.95
) -> np.ndarray:
    clipped = np.clip(values, val_min, val_max)
    norm = (clipped - val_min) / max(val_max - val_min, 1e-9)
    return low + norm * (high - low)


def _setup_axes(fig: plt.Figure, ax: Any) -> None:
    fig.patch.set_facecolor("#0B0B10")
    ax.set_facecolor("#0F111A")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_xlabel("X", color="#7E869B")
    ax.set_ylabel("Y", color="#7E869B")
    ax.set_zlabel("Z", color="#7E869B")
    ax.tick_params(colors="#6D7385")


def _rgba(color: tuple[float, float, float], alpha: np.ndarray) -> np.ndarray:
    arr = np.zeros((len(alpha), 4), dtype=float)
    arr[:, :3] = color
    arr[:, 3] = alpha
    return arr


def _border_metrics(
    fig: plt.Figure,
    title: str,
    frame_idx: int,
    total_frames: int,
    amount_nodes: int,
    dims: int,
    val_min: float,
    val_max: float,
) -> None:
    fig.text(0.01, 0.98, title, color="#D6D8E6", fontsize=10, va="top", ha="left")
    fig.text(
        0.99,
        0.98,
        f"frame {frame_idx + 1}/{total_frames}",
        color="#D6D8E6",
        fontsize=10,
        va="top",
        ha="right",
    )
    fig.text(
        0.01,
        0.02,
        f"nodes={amount_nodes} dims={dims}",
        color="#9AA0B8",
        fontsize=9,
        va="bottom",
        ha="left",
    )
    fig.text(
        0.99,
        0.02,
        f"value range [{val_min:.3f}, {val_max:.3f}]",
        color="#9AA0B8",
        fontsize=9,
        va="bottom",
        ha="right",
    )


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
    ax.scatter(
        frame.x,
        frame.y,
        frame.z,
        s=quality.point_size_static,
        c=_rgba(color, alpha),
        edgecolors="none",
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

    print(f"[render-animation] key={key} -> {output_path}")
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
        ax.scatter(
            current.x,
            current.y,
            current.z,
            s=quality.point_size_anim,
            c=_rgba(color, alpha),
            depthshade=True,
            edgecolors="none",
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

    global_min = min(vr[0] for vr in value_ranges.values())
    global_max = max(vr[1] for vr in value_ranges.values())

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
            ax.scatter(
                frame.x,
                frame.y,
                frame.z,
                s=quality.point_size_combined,
                c=_rgba(colors[key], alpha),
                label=key,
                depthshade=True,
                edgecolors="none",
                rasterized=use_rasterized,
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
    print("[engine] start")
    print(f"[engine] amount_nodes={amount_nodes} dims={dims} output_dir={output_dir}")
    print(f"[quality] preset={quality_preset} dpi={quality.dpi} rasterize_threshold={quality.rasterize_threshold}")

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
            key=key,
            frames=frames,
            value_range=value_ranges[key],
            color=colors[key],
            output_path=per_key_static_dir / f"{key}_3d.png",
            amount_nodes=amount_nodes,
            dims=dims,
            quality=quality,
        )
        render_key_animation(
            key=key,
            frames=frames,
            value_range=value_ranges[key],
            color=colors[key],
            output_path=per_key_animation_dir / f"{key}_3d.gif",
            amount_nodes=amount_nodes,
            dims=dims,
            quality=quality,
        )

    render_environment_animation(
        all_frames=normalized,
        value_ranges=value_ranges,
        colors=colors,
        output_path=combined_dir / "environment_3d.gif",
        amount_nodes=amount_nodes,
        dims=dims,
        quality=quality,
    )
    print("[engine] done")


def _demo_input_data(timesteps: int = 42) -> dict[str, list[Any]]:
    t = np.linspace(0, 6 * np.pi, timesteps)
    return {
        "sensor_A": [np.array([np.sin(v), np.cos(v), np.sin(v * 0.5)]) for v in t],
        "sensor_B": [float(1.8 + 0.9 * np.sin(v * 0.8)) for v in t],
        "sensor_C": [
            np.array([complex(np.sin(v + i * 0.2), np.cos(v * 0.6 - i * 0.15)) for i in range(5)])
            for v in t
        ],
        "sensor_D": [
            {
                "list_array_3": np.array([np.sin(v * 0.3), np.cos(v * 0.4), np.sin(v * 0.5)]),
                "float": float(v / np.pi),
                "array_complex_5": np.array([complex(np.cos(v + i), np.sin(v - i)) for i in range(5)]),
                "int_field": int((v * 100) % 17),
            }
            for v in t
        ],
    }


if __name__ == "__main__":
    print("[demo] generating hardcoded mixed time-series data")
    demo = _demo_input_data(timesteps=50)
    build_3d_time_series_visualization(demo, amount_nodes=36, dims=360, output_dir="output_dir")
