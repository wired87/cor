"""
Path-based color_master: `local.json` + `sim_cfg.json`

User prompt: Analyze color_master, local.json, sim_cfg.json — adapt color_master.main to visualize
all deserialized data points within a single animation, clearly referenced to specific (indexed)
data points over distributed controller (ctlr) components. Best practice.

"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import to_rgb

from color_master.viz_types import FramePoints, PlotQuality, QUALITY_HIGH, QUALITY_LIGHT


def check_plot_quality(key: str, n_points: int, quality: PlotQuality) -> bool:
    """Validate quality settings and return whether to use rasterized=True."""
    use_rasterized = n_points >= quality.rasterize_threshold
    #print(f"[quality] key={key} points={n_points} dpi={quality.dpi} rasterized={use_rasterized}")
    if n_points < 100:
        #print(f"[quality] warn: low point count ({n_points}) may reduce detail")
        pass
    if quality.dpi < quality.min_dpi:
        #print(f"[quality] warn: dpi={quality.dpi} below min_dpi={quality.min_dpi}")
        pass
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
    # CHAR: non-finite range (all-NaN data, bad export) would make alpha NaN → invisible in MPL
    if not (np.isfinite(val_min) and np.isfinite(val_max)):
        val_min, val_max = 0.0, 1.0
    v = np.asarray(values, dtype=float)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    clipped = np.clip(v, val_min, val_max)
    norm = (clipped - val_min) / max(val_max - val_min, 1e-9)
    out = low + norm * (high - low)
    return np.clip(out, low, high)


def _scatter3d_facecolors(
    ax: Any,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    facecolors: np.ndarray,
    *,
    s: float,
    depthshade: bool = False,
    rasterized: bool = False,
    label: str | None = None,
) -> None:
    """CHAR: mplot3d often fails to show per-point RGBA via `c=`; use `facecolors=` (see MPL #8897 / art3d)."""
    kw: dict[str, Any] = dict(
        s=s,
        facecolors=facecolors,
        depthshade=depthshade,
        edgecolors="none",
        rasterized=rasterized,
    )
    if label is not None:
        kw["label"] = label
    ax.scatter(x, y, z, **kw)


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
    #
    if not frames:
        return

    #print(f"[render-static] key={key} -> {output_path}")
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


def _quality_from_name_v2(preset: str) -> PlotQuality:
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


def build_indexed_viz_from_engine_dict(
    engine: dict[str, Any],
    output_dir: str | os.PathLike[str],
    *,
    quality_preset: str = "light",
    series: str = "values",
    max_indices: Optional[int] = None,
    max_frames: Optional[int] = None,
    subdir: str = "indexed",
) -> str:
    """Build single combined GIF from in-memory engine payload (param_series + ctlr)."""
    from color_master.engine_json import prepare_indexed_viz

    out = Path(output_dir) / subdir
    out.mkdir(parents=True, exist_ok=True)
    q = _quality_from_name_v2(quality_preset)
    frames, labels, g_min, g_max, _ct = prepare_indexed_viz(
        engine, series=series, max_indices=max_indices, max_frames=max_frames
    )
    n = len(labels)
    if not frames or n == 0:
        print("[indexed-viz] no frames; skip")
        return str(out)
    render_indexed_unified_animation(
        frames=frames,
        labels=labels,
        value_range=(g_min, g_max),
        output_path=out / "environment_3d.gif",
        quality=q,
        n_params=n,
    )
    print(f"[indexed-viz] wrote {out / 'environment_3d.gif'}")
    return str(out)


def run_path_based_viz(sim_cfg_path: str | None = None) -> str:
    """
    Path-based pipeline: `sim_cfg.json` points at enriched `local.json` (param_series + ctlr).
    """
    from color_master.config_loader import load_sim_viz_config
    from color_master.engine_json import load_engine_json, prepare_indexed_viz

    p = sim_cfg_path or os.environ.get("COLOR_MASTER_SIM_CFG", "sim_cfg.json")
    cfg = load_sim_viz_config(p)
    engine = load_engine_json(cfg.local_json)
    cm = cfg.color_master
    q = _quality_from_name_v2(cm.quality)
    fr, lab, g0, g1, _c = prepare_indexed_viz(
        engine,
        series=cm.series,
        max_indices=cm.max_indices,
        max_frames=cm.max_frames,
    )
    n = len(lab)
    base = Path(cfg.output_dir) / cm.output_subdir
    base.mkdir(parents=True, exist_ok=True)
    if n and fr:
        render_indexed_unified_animation(
            frames=fr,
            labels=lab,
            value_range=(g0, g1),
            output_path=base / "environment_3d.gif",
            quality=q,
            n_params=n,
        )
    else:
        print("[path-viz] no indexed frames; check param_series in local json")
    print(f"[path-viz] done -> {base}")
    return str(base)


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
    cfg_arg = None
    if len(sys.argv) > 1 and str(sys.argv[1]).lower().endswith(".json"):
        cfg_arg = sys.argv[1]
    elif os.environ.get("COLOR_MASTER_SIM_CFG"):
        cfg_arg = os.environ["COLOR_MASTER_SIM_CFG"]
    if cfg_arg and Path(cfg_arg).is_file():
        run_path_based_viz(cfg_arg)
    elif (not os.environ.get("COLOR_MASTER_DEMO")) and Path("sim_cfg.json").is_file():
        run_path_based_viz("sim_cfg.json")
    elif os.environ.get("COLOR_MASTER_DEMO") == "1":
        print("[demo] COLOR_MASTER_DEMO=1: legacy series visualization")
        demo = _demo_input_data(timesteps=50)
        build_3d_time_series_visualization(demo, amount_nodes=36, dims=360, output_dir="output_dir")
    else:
        print(
            "Path-based: set COLOR_MASTER_SIM_CFG to sim_cfg path, or pass a .json path, "
            "or place sim_cfg.json in CWD, or set COLOR_MASTER_DEMO=1 for legacy demo."
        )
