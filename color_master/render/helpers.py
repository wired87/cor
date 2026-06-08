"""
Prompt (2026-06): Shared matplotlib helpers for color_master renderers.
"""
from __future__ import annotations
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
from color_master.types import FramePoints, PlotQuality

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


