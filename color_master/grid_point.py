"""
Per-grid-point activity visualizations.

User prompt — "Within the color master create for each point in the grid an animation
that renders the feature values of all nodes (all fields correspond to the specific
grid point). Use here the specified controller from output dir to extract values
index based of specific fields for all nodes. The goal is to understand the activity
of each field on a specific point to understand e.g. gluon-gluon activity and their
results. Create for each point's activity a 3D animation graph (include edges based
on interaction — firegraph style) — file, and a 2D chart that shows the frequency
of interaction (split into specific colors based on the field — lines)."

Outputs (per grid point `g_x_y_z` under `<viz_root>/gridpoints/`):
  - `3d_activity.gif`   — Feynman/firegraph-style 3D animation. F field-nodes laid
                          out on a ring around the grid point; node size encodes
                          per-field activity at this grid point at time t; edges
                          are drawn between strongly co-active fields with width
                          ~ activity_i * activity_j and an edge colour that is a
                          blend of the two field colours. This visualises pairwise
                          interaction (gluon-gluon, photon-electron, ...) directly.
  - `freq_chart.png`    — 2D line chart, one line per field with the same field
                          palette as the 3D animation, showing |activity[t]| at
                          this grid point over the full time range.

Data sources:
  - `output/ctlr/db_ctlr.json` — MODULES / FIELDS / AMOUNT_PARAMS_PER_FIELD /
    DB_KEYS — for the field count F and per-field human-readable labels.
  - `gnn._features_history` (one entry per simulation timestep, populated inside
    `GNN.calc_batch`) — the live feature tensors. They are absolute-valued, padded
    to a multiple of `F * N**3` and folded into `(chunks, F, N, N, N)`, then
    summed across `chunks` so every linearly-folded "field-like partition" of
    the flat buffer collapses onto the same `(field, grid)` slot. This matches
    the user instruction "extract values index based of specific fields for all
    nodes".
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import to_rgb


# CHAR: stable colour palette. 20 distinct base colours via tab20; if F > 20 we cycle
# CHAR: but vary the brightness so cycled fields stay visually distinguishable.
def _field_palette(num_fields: int) -> np.ndarray:
    base_cmap = plt.get_cmap("tab20", 20)
    out = np.zeros((num_fields, 3), dtype=float)
    for i in range(num_fields):
        rgb = np.asarray(to_rgb(base_cmap(i % 20)[:3]), dtype=float)
        # CHAR: cycle-aware brightness shift so the (i+20)th field is darker than the i-th
        shift = 0.85 ** (i // 20)
        out[i] = np.clip(rgb * shift, 0.0, 1.0)
    return out


# CHAR: build per-field labels from the controller. AMOUNT_PARAMS_PER_FIELD tells us
# CHAR: how many DB_KEYS belong to each field; the first key in each field group is
# CHAR: a stable, human-readable proxy name (e.g. "gterm", "psi", "y", ...). MODULES
# CHAR: gives a `mod{m}_<key>` prefix so duplicate keys across modules stay unique.
def _field_labels_from_ctlr(ctlr: Dict[str, Any]) -> List[str]:
    fields_per_module: List[int] = list(ctlr.get("FIELDS") or [])
    params_per_field: List[int] = list(ctlr.get("AMOUNT_PARAMS_PER_FIELD") or [])
    db_keys: List[str] = list(ctlr.get("DB_KEYS") or [])

    f_total = sum(int(x) for x in fields_per_module) if fields_per_module else len(params_per_field)
    if f_total <= 0:
        f_total = max(1, len(params_per_field))

    # CHAR: walk the (module, field-within-module) tuples in the same order the engine emitted them
    labels: List[str] = []
    cursor = 0
    f_idx = 0
    for m_idx, n_fields_in_module in enumerate(fields_per_module):
        for fi_in_m in range(int(n_fields_in_module)):
            n_params = int(params_per_field[f_idx]) if f_idx < len(params_per_field) else 0
            head_key = db_keys[cursor] if cursor < len(db_keys) else f"f{f_idx:02d}"
            labels.append(f"m{m_idx}_f{fi_in_m:02d}_{head_key}")
            cursor += max(n_params, 0)
            f_idx += 1

    # CHAR: pad / truncate so caller always gets exactly the field count it asked about
    while len(labels) < f_total:
        labels.append(f"field_{len(labels):02d}")
    return labels[:f_total]


# CHAR: load the controller dict; missing-file is non-fatal — caller falls back to
# CHAR: generic field count + names so the viz pipeline never blocks on metadata.
def _load_db_ctlr(ctlr_path: Path) -> Dict[str, Any]:
    if not ctlr_path.is_file():
        return {}
    try:
        with open(ctlr_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"[gridpoint-viz] could not read controller {ctlr_path}: {type(exc).__name__}: {exc}")
        return {}


def _to_numpy(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    try:
        # CHAR: lazy jax import — color_master may run in environments without jax during static viz
        import jax.numpy as jnp  # type: ignore
        if isinstance(x, jnp.ndarray):
            return np.asarray(x)
    except Exception:
        pass
    try:
        return np.asarray(x)
    except Exception:
        return None


# CHAR: build (T, F, N, N, N) field-resolved activity volume from gnn feature history.
# CHAR: same fold strategy as the white->blue grid volume, but the fold target is now
# CHAR: F * N^3 (extra axis for fields) so the field dimension survives the chunk-sum.
def build_field_volume(
    jax_guard: Any, amount_nodes: int, num_fields: int
) -> Optional[np.ndarray]:
    if jax_guard is None or amount_nodes < 1 or num_fields < 1:
        return None
    gnn = getattr(jax_guard, "gnn_layer", None)
    history = getattr(gnn, "_features_history", None) if gnn is not None else None
    if not history:
        return None

    grid_n = int(amount_nodes)
    grid_size = grid_n ** 3
    fold_target = num_fields * grid_size
    if fold_target <= 0:
        return None

    per_step: List[np.ndarray] = []
    for step_features in history:
        flat_chunks: List[np.ndarray] = []
        for feat in step_features or []:
            arr = _to_numpy(feat)
            if arr is None:
                continue
            v = np.abs(np.asarray(arr, dtype=np.float32).ravel())
            if v.size == 0:
                continue
            if not np.all(np.isfinite(v)):
                v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            flat_chunks.append(v)

        if not flat_chunks:
            per_step.append(np.zeros((num_fields, grid_n, grid_n, grid_n), dtype=np.float32))
            continue

        flat = np.concatenate(flat_chunks)
        # CHAR: pad up to a multiple of F * N^3 so the reshape is exact; padding with
        # CHAR: zeros never inflates field activities.
        pad = (-flat.size) % fold_target
        if pad:
            flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
        chunks = flat.reshape(-1, num_fields, grid_n, grid_n, grid_n)
        per_step.append(chunks.sum(axis=0).astype(np.float32))

    if not per_step:
        return None
    return np.stack(per_step, axis=0)


# CHAR: per-grid-point time-series tensor — extract a (T, F) slice from the volume
# CHAR: at a fixed (x, y, z). This is the input every per-point renderer consumes.
def _series_at_grid(volume: np.ndarray, x: int, y: int, z: int) -> np.ndarray:
    return np.asarray(volume[:, :, x, y, z], dtype=float)


# -----------------------------------------------------------------------------
#  2D frequency chart — one PNG per grid point
# -----------------------------------------------------------------------------

def render_freq_chart_2d(
    series: np.ndarray,
    field_labels: Sequence[str],
    field_colors: np.ndarray,
    grid_xyz: Tuple[int, int, int],
    output_path: Path,
    *,
    dpi: int = 130,
    max_legend_fields: int = 16,
) -> None:
    """2D line chart — one line per field of the per-grid-point activity time series.

    `series` has shape (T, F). Each field f gets a coloured polyline; the legend
    is truncated to the top-`max_legend_fields` most active fields (by total
    energy) so it stays readable when F is large.
    """
    if series.ndim != 2 or series.size == 0:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)

    T, F = series.shape
    # CHAR: rank fields by integrated activity so the legend surfaces the "loudest" fields first
    energy = np.sum(np.abs(series), axis=0)
    rank = np.argsort(-energy)
    legend_idx = set(rank[:max_legend_fields].tolist())

    fig = plt.figure(figsize=(10, 5), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#FAFAFA")
    ax.grid(True, color="#E2E2E2", linewidth=0.6, alpha=0.8)

    xs = np.arange(T)
    for f in range(F):
        color = tuple(float(c) for c in field_colors[f])
        label = field_labels[f] if (f < len(field_labels) and f in legend_idx) else None
        # CHAR: faded lines for fields that are not in the legend so the eye still groups by colour
        alpha = 0.95 if f in legend_idx else 0.35
        lw = 1.6 if f in legend_idx else 0.9
        ax.plot(xs, series[:, f], color=color, alpha=alpha, linewidth=lw, label=label)

    ax.set_xlabel("timestep")
    ax.set_ylabel("|feature activity|")
    ax.set_title(
        f"Per-field activity frequency at grid point (x={grid_xyz[0]}, y={grid_xyz[1]}, z={grid_xyz[2]})  |  "
        f"F={F}, T={T}",
        fontsize=11,
    )
    if any(label is not None for label in [field_labels[i] for i in legend_idx if i < len(field_labels)]):
        ax.legend(loc="upper right", fontsize=7, ncol=2, frameon=False, title=f"top {min(F, max_legend_fields)} fields")

    fig.tight_layout()
    fig.savefig(output_path, facecolor=fig.get_facecolor(), dpi=dpi)
    plt.close(fig)


# -----------------------------------------------------------------------------
#  3D activity animation — one GIF per grid point (firegraph / Feynman style)
# -----------------------------------------------------------------------------

def _ring_layout(num_fields: int, center: Tuple[float, float, float], radius: float = 1.0) -> np.ndarray:
    """Place F field-nodes on a tilted ring around `center` in 3D.

    A flat ring layout in 3D keeps the firegraph readable (no node-occlusion
    blowups) and gives a clear pairwise-interaction surface — every edge is a
    chord of the same circle, exactly the visual idiom of a Feynman-style
    interaction diagram.
    """
    cx, cy, cz = center
    out = np.zeros((num_fields, 3), dtype=float)
    if num_fields <= 0:
        return out
    angles = np.linspace(0.0, 2.0 * np.pi, num=num_fields, endpoint=False)
    # CHAR: small z tilt so the ring is visible from a default 3D camera without flattening edges
    tilt = 0.18
    for i, a in enumerate(angles):
        out[i, 0] = cx + radius * math.cos(a)
        out[i, 1] = cy + radius * math.sin(a)
        out[i, 2] = cz + tilt * math.sin(2.0 * a)
    return out


def render_gridpoint_3d_animation(
    series: np.ndarray,
    field_labels: Sequence[str],
    field_colors: np.ndarray,
    grid_xyz: Tuple[int, int, int],
    output_path: Path,
    *,
    dpi: int = 120,
    edge_top_k: int = 30,
    fps: int = 6,
) -> None:
    """3D firegraph animation showing per-field activity + pairwise interaction edges.

    `series` shape (T, F). For each frame t:
      * F nodes are placed on a fixed ring around the grid point.
      * node `f` is drawn with marker-size scaled by `series[t, f]`.
      * the top-`edge_top_k` strongest pairs are drawn as edges with
          width  ~  s_i * s_j         (interaction "strength")
          colour ~  0.5*(c_i + c_j)   (blend of the two field colours)
        — this is the canonical firegraph rendering of co-activation.
    """
    if series.ndim != 2 or series.size == 0:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)

    T, F = series.shape
    if F < 2:
        return

    # CHAR: per-frame normalisation by the global max across (T, F) keeps node sizes
    # CHAR: comparable across timesteps; tiny floor avoids div-by-0 on all-zero runs.
    g_max = float(np.nanmax(series)) if np.isfinite(np.nanmax(series)) else 0.0
    if g_max <= 0.0:
        g_max = 1.0

    pos = _ring_layout(F, center=(float(grid_xyz[0]), float(grid_xyz[1]), float(grid_xyz[2])), radius=0.45)

    # CHAR: precompute every unordered pair (i, j) once — cheaper than rebuilding per frame.
    pairs = np.array([(i, j) for i in range(F) for j in range(i + 1, F)], dtype=np.int32)
    n_pairs = len(pairs)
    edge_top_k = max(1, min(edge_top_k, n_pairs))

    fig = plt.figure(figsize=(8, 7), dpi=dpi)
    fig.patch.set_facecolor("#0B0B10")
    ax = fig.add_subplot(111, projection="3d")

    def _setup_dark_axes(ax_: Any) -> None:
        ax_.set_facecolor("#0F111A")
        ax_.grid(False)
        ax_.set_xticks([])
        ax_.set_yticks([])
        ax_.set_zticks([])
        ax_.xaxis.pane.fill = False
        ax_.yaxis.pane.fill = False
        ax_.zaxis.pane.fill = False
        cx, cy, cz = grid_xyz
        ax_.set_xlim(cx - 0.7, cx + 0.7)
        ax_.set_ylim(cy - 0.7, cy + 0.7)
        ax_.set_zlim(cz - 0.7, cz + 0.7)

    def update(frame_idx: int) -> List[Any]:
        ax.cla()
        _setup_dark_axes(ax)

        s = np.clip(series[frame_idx] / g_max, 0.0, 1.0)

        # CHAR: pairwise co-activation = element-wise product. Equivalent to the
        # CHAR: classical "rate at vertex × rate at vertex" coupling -> Feynman-vertex weight.
        pair_weight = s[pairs[:, 0]] * s[pairs[:, 1]]
        if n_pairs > edge_top_k:
            top_idx = np.argpartition(-pair_weight, edge_top_k - 1)[:edge_top_k]
        else:
            top_idx = np.arange(n_pairs)

        # CHAR: edges first (so node markers stay on top); width + alpha both follow weight
        for k in top_idx:
            i, j = int(pairs[k, 0]), int(pairs[k, 1])
            w = float(pair_weight[k])
            if w <= 0.0:
                continue
            blend = 0.5 * (field_colors[i] + field_colors[j])
            ax.plot(
                [pos[i, 0], pos[j, 0]],
                [pos[i, 1], pos[j, 1]],
                [pos[i, 2], pos[j, 2]],
                color=tuple(blend),
                linewidth=0.6 + 3.5 * w,
                alpha=min(0.95, 0.15 + 0.85 * w),
                solid_capstyle="round",
            )

        # CHAR: nodes — facecolor uses field palette; size scaled by activity (with a
        # CHAR: visible floor so quiet fields stay locatable on the diagram).
        sizes = 18.0 + 240.0 * s
        ax.scatter(
            pos[:, 0],
            pos[:, 1],
            pos[:, 2],
            s=sizes,
            facecolors=field_colors,
            edgecolors="white",
            linewidths=0.4,
            depthshade=False,
        )

        # CHAR: anchor the grid point itself as a small "origin" marker so the user
        # CHAR: always sees which (x,y,z) point this firegraph is centered on.
        ax.scatter(
            [grid_xyz[0]],
            [grid_xyz[1]],
            [grid_xyz[2]],
            s=22.0,
            facecolors=[(1.0, 1.0, 1.0)],
            edgecolors=[(0.7, 0.7, 0.7)],
            depthshade=False,
        )

        ax.view_init(elev=20, azim=30 + frame_idx * 6.0)
        fig.suptitle(
            f"Firegraph | grid point ({grid_xyz[0]},{grid_xyz[1]},{grid_xyz[2]})  |  F={F}",
            color="#D6D8E6",
            fontsize=11,
        )
        fig.text(
            0.99, 0.02, f"t {frame_idx + 1}/{T}",
            color="#9AA0B8", fontsize=9, va="bottom", ha="right",
        )
        return []

    ani = animation.FuncAnimation(fig, update, frames=T, interval=140, blit=False)
    ani.save(output_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


# -----------------------------------------------------------------------------
#  Orchestrator — produce a (3D-gif + 2D-png) pair for every grid point
# -----------------------------------------------------------------------------

def render_gridpoint_visualizations(
    jax_guard: Any,
    output_root: str | os.PathLike[str],
    *,
    amount_nodes: int,
    ctlr_path: Optional[str | os.PathLike[str]] = None,
    edge_top_k: int = 30,
    max_legend_fields: int = 16,
) -> Optional[str]:
    """End-to-end driver: build the field volume and render every grid point.

    Returns the absolute path of the directory that received the per-grid-point
    artifacts (`<output_root>/gridpoints`), or None if no data could be built.
    """
    # CHAR: step 1 — controller + field count. Falls back to FIELDS sum from db_ctlr.
    # gien: flat layout — db_ctlr.json lives in the same `output/` root as viz files
    ctlr_path_obj = Path(ctlr_path) if ctlr_path is not None else (
        Path(output_root) / "db_ctlr.json"
    )
    ctlr = _load_db_ctlr(ctlr_path_obj)
    fields_per_module = list(ctlr.get("FIELDS") or [])
    num_fields = int(sum(int(x) for x in fields_per_module)) if fields_per_module else 0
    if num_fields <= 0:
        # CHAR: minimal safe default if controller has no FIELDS list — won't crash, just
        # CHAR: produces a single-line chart and a 2-node firegraph (still valid output).
        num_fields = 8
    field_labels = _field_labels_from_ctlr(ctlr) if ctlr else [f"field_{i:02d}" for i in range(num_fields)]
    field_colors = _field_palette(num_fields)

    # CHAR: step 2 — extract (T, F, N, N, N) field-resolved activity volume
    volume = build_field_volume(jax_guard, amount_nodes=amount_nodes, num_fields=num_fields)
    if volume is None:
        print("[gridpoint-viz] no _features_history; skipping per-grid-point renders")
        return None

    T, F, Nx, Ny, Nz = volume.shape
    if F != num_fields or Nx != amount_nodes or Ny != amount_nodes or Nz != amount_nodes:
        print(
            f"[gridpoint-viz] unexpected volume shape {volume.shape}; expected (T, {num_fields}, "
            f"{amount_nodes}, {amount_nodes}, {amount_nodes}) — aborting"
        )
        return None

    out_dir = Path(output_root) / "gridpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    # CHAR: step 3 — loop over every (x,y,z); for each one emit one GIF + one PNG.
    rendered = 0
    for x in range(amount_nodes):
        for y in range(amount_nodes):
            for z in range(amount_nodes):
                series = _series_at_grid(volume, x, y, z)  # (T, F)
                cell_dir = out_dir / f"g_{x}_{y}_{z}"
                cell_dir.mkdir(parents=True, exist_ok=True)

                # CHAR: 2D frequency line chart (per-field activity over time)
                try:
                    render_freq_chart_2d(
                        series=series,
                        field_labels=field_labels,
                        field_colors=field_colors,
                        grid_xyz=(x, y, z),
                        output_path=cell_dir / "freq_chart.png",
                        max_legend_fields=max_legend_fields,
                    )
                except Exception as exc:
                    print(f"[gridpoint-viz] freq_chart {x},{y},{z} failed: {type(exc).__name__}: {exc}")

                # CHAR: 3D firegraph animation (per-field activity + pairwise edges)
                try:
                    render_gridpoint_3d_animation(
                        series=series,
                        field_labels=field_labels,
                        field_colors=field_colors,
                        grid_xyz=(x, y, z),
                        output_path=cell_dir / "3d_activity.gif",
                        edge_top_k=edge_top_k,
                    )
                except Exception as exc:
                    print(f"[gridpoint-viz] 3d_activity {x},{y},{z} failed: {type(exc).__name__}: {exc}")

                rendered += 1

    # CHAR: step 4 — write a tiny index file mapping (x,y,z) -> dir + field palette legend
    legend = [
        {"field_index": i, "label": field_labels[i], "rgb": [float(c) for c in field_colors[i]]}
        for i in range(num_fields)
    ]
    index_payload = {
        "amount_nodes": amount_nodes,
        "num_fields": num_fields,
        "timesteps": T,
        "edge_top_k": edge_top_k,
        "field_palette": legend,
        "grid_points": [
            {
                "x": x, "y": y, "z": z,
                "dir": f"g_{x}_{y}_{z}",
                "freq_chart": f"g_{x}_{y}_{z}/freq_chart.png",
                "anim_3d":   f"g_{x}_{y}_{z}/3d_activity.gif",
            }
            for x in range(amount_nodes)
            for y in range(amount_nodes)
            for z in range(amount_nodes)
        ],
    }
    try:
        with open(out_dir / "index.json", "w", encoding="utf-8") as fh:
            json.dump(index_payload, fh, indent=2)
    except Exception as exc:
        print(f"[gridpoint-viz] index.json write failed: {type(exc).__name__}: {exc}")

    print(f"[gridpoint-viz] rendered {rendered} grid-point pairs (T={T}, F={F}, N={amount_nodes}) -> {out_dir}")
    return str(out_dir)


# gien: clearer public alias for per-grid-point activity renders (2026-06 refactor)
render_grid_point_activity_views = render_gridpoint_visualizations
