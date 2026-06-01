"""
Bridge JAX simulation output -> color_master 3D time-series visualizations.

Loaded with color_master on sys.path so `from main import ...` resolves to color_master/main.py.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

# gien: resolve color_master entrypoints once so workflow visualization stays import-safe at runtime
from color_master.main import (
    QUALITY_LIGHT,
    _demo_input_data,
    build_3d_time_series_visualization,
    build_indexed_viz_from_engine_dict,
    render_grid_white_to_blue,
)
# CHAR: PlotQuality is needed when no preset is requested (passes a sane default to render_grid_white_to_blue)
from color_master.viz_types import PlotQuality, QUALITY_HIGH


def _to_numpy(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    try:
        import jax.numpy as jnp

        if isinstance(x, jnp.ndarray):
            return np.asarray(x)
    except Exception:
        print("Err color_master.sim_bridge::_to_numpy | handler_line=23 | Exception handler triggered")
        print("[exception] color_master.sim_bridge._to_numpy: caught Exception")
        pass
    try:
        return np.asarray(x)
    except Exception:
        print("Err color_master.sim_bridge::_to_numpy | handler_line=28 | Exception handler triggered")
        print("[exception] color_master.sim_bridge._to_numpy: caught Exception")
        return None


def _series_from_param_histories(
    dl: Any, max_keys: Optional[int] = None
) -> Dict[str, List[Any]]:
    out: Dict[str, List[Any]] = {}
    pv = getattr(dl, "param_values_history", None) or {}
    pf = getattr(dl, "param_features_history", None) or {}
    added = 0
    for name, src in (("value", pv), ("feature", pf)):
        sk = sorted(src.keys(), key=lambda x: (str(type(x).__name__), str(x)))
        for k in sk:
            if max_keys is not None and added >= max_keys:
                break
            series = src.get(k)
            if not series:
                continue
            out[f"param_{name}_{k}"] = [float(v) for v in series]
            added += 1
        if max_keys is not None and added >= max_keys:
            break
    return out


def _series_from_feature_encoder(gnn: Any, max_eq: int, max_steps: int) -> Dict[str, List[Any]]:
    out: Dict[str, List[Any]] = {}
    enc = getattr(gnn, "feature_encoder", None)
    if enc is None:
        return out
    in_store = getattr(enc, "in_store", None) or []
    for eq_i, bucket in enumerate(in_store[:max_eq]):
        if not bucket:
            continue
        slot0 = bucket[0] if isinstance(bucket[0], list) else None
        if not slot0:
            continue
        series: List[Any] = []
        for t, feat in enumerate(slot0[:max_steps]):
            arr = _to_numpy(feat)
            if arr is None:
                continue
            flat = np.asarray(arr, dtype=np.float32).ravel()
            if flat.size == 0:
                continue
            series.append(flat)
        if len(series) >= 2:
            out[f"encoder_in_eq{eq_i}"] = series
    return out


# prompt - "Adapt color_master to generate the main animation based on feature values of all
# grid points separated (start = grid with white background and white dots). The more value a
# grid point has (sum of feature values of all fields on that grid point), the more it
# transitions from white to blue, animated over all timesteps."
def _build_grid_value_volume(
    jax_guard: Any, amount_nodes: int
) -> Optional[np.ndarray]:
    """Aggregate per-timestep scalar value at every (x, y, z) grid point.

    Walks `gnn._features_history` (per-step list of input feature tensors that
    `calc_batch` snapshots one-per-timestep) and folds the absolute-valued flat
    float buffer into chunks of `N**3`, then sums chunks along the partition
    axis. The chunk-sum collapses every "field-like partition" of the flat
    feature stream onto the same grid coordinates, which matches the user's
    instruction "sum the features of all fields on a specific node grid point".

    Returns an ndarray of shape (T, N, N, N) with non-negative float32 values,
    or None if no feature history is available (e.g. demo-data run).
    """
    # CHAR: defensive guards - without a guard or a sane node count we cannot build a volume
    if jax_guard is None or amount_nodes < 1:
        return None
    gnn = getattr(jax_guard, "gnn_layer", None)
    history = getattr(gnn, "_features_history", None) if gnn is not None else None
    if not history:
        return None

    grid_size = int(amount_nodes) ** 3
    if grid_size <= 0:
        return None

    volume_steps: List[np.ndarray] = []
    # CHAR: iterate per timestep - each `step_features` is a list of jax/np feature tensors
    for step_features in history:
        flat_chunks: List[np.ndarray] = []
        for feat in step_features or []:
            arr = _to_numpy(feat)
            if arr is None:
                continue
            # CHAR: |feature| -> non-negative magnitudes; "more value = more blue" needs >=0 inputs
            v = np.abs(np.asarray(arr, dtype=np.float32).ravel())
            if v.size == 0:
                continue
            # CHAR: nan/inf clamp at the viz boundary (matches the same policy as serialization)
            if not np.all(np.isfinite(v)):
                v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            flat_chunks.append(v)

        # CHAR: empty step -> all-zero slice (renders as pure-white grid, exactly the desired t=0)
        if not flat_chunks:
            volume_steps.append(np.zeros((amount_nodes, amount_nodes, amount_nodes), dtype=np.float32))
            continue

        flat = np.concatenate(flat_chunks)
        # CHAR: pad to a multiple of grid_size so reshape is exact; padding zeros add nothing
        pad = (-flat.size) % grid_size
        if pad:
            flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
        # CHAR: reshape into (chunks, N, N, N) and sum across chunks -> per-grid-point summed value
        chunks = flat.reshape(-1, amount_nodes, amount_nodes, amount_nodes)
        volume_steps.append(chunks.sum(axis=0).astype(np.float32))

    if not volume_steps:
        return None
    return np.stack(volume_steps, axis=0)


def collect_engine_payload_from_jax_guard(guard: Any) -> Optional[Dict[str, Any]]:
    """
    Full ctlr + param_series (same shape as enriched local.json) when DBLayer has histories
    and JaxGuard exports are available.
    """
    if guard is None:
        return None
    dl = getattr(getattr(guard, "gnn_layer", None), "db_layer", None)
    if dl is None or not hasattr(guard, "_build_param_series_payload"):
        return None
    ps = guard._build_param_series_payload(dl)
    if not ps or not ps.get("series"):
        return None
    return {"param_series": ps, "ctlr": guard._build_ctlr_for_export()}


def collect_series_from_jax_guard(
    guard: Any, max_param_keys: Optional[int] = None
) -> Dict[str, List[Any]]:
    """
    Build dict[str, list[timestep_payload]] for build_3d_time_series_visualization.
    If max_param_keys is None, no cap (full parity with file export order).
    """
    gnn = getattr(guard, "gnn_layer", None)
    if gnn is None:
        return {}
    dl = getattr(gnn, "db_layer", None)
    if dl is None:
        return {}

    data = _series_from_param_histories(dl, max_param_keys)
    if not data:
        data = _series_from_feature_encoder(gnn, max_eq=6, max_steps=500)
    return data

def run_workflow_visualization(
    output_dir: Optional[str] = None,
    *,
    jax_guard: Any = None,
    amount_nodes: Optional[int] = None,
    dims: Optional[int] = None,
    quality_preset: str = "light",
    out_name: str = "single_animation"
) -> str:

    import os
    import numpy as np
    import json

    repo_parent = os.path.dirname(os.path.abspath(__file__))
    default_out = os.path.join(os.path.dirname(repo_parent), "color_master_output")
    out = output_dir or os.environ.get("COLOR_MASTER_OUT", default_out)
    os.makedirs(out, exist_ok=True)

    n_nodes = amount_nodes if amount_nodes is not None else int(os.environ.get("AMOUNT_NODES", "4"))
    n_dims = dims if dims is not None else int(os.environ.get("DIMS", "3"))

    # -------------------------
    # COLLECT DATA
    # -------------------------
    data: Dict[str, List[Any]] = {}

    if jax_guard is not None:
        data = collect_series_from_jax_guard(jax_guard)

    if not data:
        t_steps = max(8, int(os.environ.get("SIM_TIME", "24")))
        print(f"[color_master] no simulation series; using demo data (timesteps={t_steps})")
        data = _demo_input_data(timesteps=t_steps)

    # -------------------------
    # MERGE INTO SINGLE TIMELINE
    # -------------------------
    max_len = max(len(v) for v in data.values())

    frames = []

    for t in range(max_len):
        frame = []

        for key, series in data.items():
            if t >= len(series):
                continue

            val = series[t]

            arr = _to_numpy(val)
            if arr is None:
                continue

            flat = arr.astype(np.float32).ravel()

            # encode: [x,y,z,val]
            for i, v in enumerate(flat):
                frame.append([
                    i % n_nodes,
                    (i // n_nodes) % n_nodes,
                    (i // (n_nodes * n_nodes)) % max(n_dims, 3),
                    float(v)
                ])

        frames.append(frame)

    print(f"[color_master] unified frames={len(frames)}")

    # -------------------------
    # SAVE BRAINMASTER JSON
    # -------------------------
    brainmaster_path = os.path.join(out, f"{out_name}.json")

    with open(brainmaster_path, "w") as f:
        json.dump(frames, f)

    print(f"[color_master] brainmaster saved -> {brainmaster_path}")

    # -------------------------
    # BUILD SINGLE ANIMATION
    # -------------------------
    vkw: Dict[str, Any] = dict(
        data={"merged": frames},
        amount_nodes=n_nodes,
        dims=max(n_dims, 3),
        output_dir=out,
    )

    if quality_preset == "light":
        vkw["quality"] = QUALITY_LIGHT
        vkw["quality_preset"] = "default"
    else:
        vkw["quality"] = None
        vkw["quality_preset"] = quality_preset

    build_3d_time_series_visualization(**vkw)

    print(f"[color_master] single animation output -> {out}")

    # CHAR: prompt - main animation = white grid that fades to blue per grid point as feature
    # CHAR: values accumulate. We build a (T, N, N, N) volume from the GNN feature history and
    # CHAR: overwrite `combined/environment_3d.gif` so the primary animation is now the
    # CHAR: requested per-grid-point white->blue view; per-key static + per-key animation flows
    # CHAR: above stay intact for supplementary inspection.
    try:
        # CHAR: dimension for the visual grid is fixed to 3D (always N^3 dots) for a stable cube
        # CHAR: layout regardless of the simulation's `dims` value.
        volume = _build_grid_value_volume(jax_guard, n_nodes)
        if volume is None:
            print("[color_master] grid_white_to_blue: no feature history; main animation kept as default combined view")
        else:
            # CHAR: pick the same quality preset as the rest of the workflow so dpi/sizes match
            if quality_preset == "light":
                grid_quality = QUALITY_LIGHT
            elif quality_preset == "high":
                grid_quality = QUALITY_HIGH
            else:
                grid_quality = PlotQuality()
            # CHAR: same target path as the default combined animation -> overwrites it intentionally
            from pathlib import Path as _Path
            grid_out = _Path(out) / "combined" / "environment_3d.gif"
            render_grid_white_to_blue(
                volume=volume,
                output_path=grid_out,
                quality=grid_quality,
                amount_nodes=n_nodes,
                title="Main animation | grid feature density (white -> blue)",
            )
            print(f"[color_master] main animation (white->blue grid) -> {grid_out}")
    except Exception as exc:
        # CHAR: never let the new viz crash the whole workflow; the default combined view above is fine fallback
        print(f"[color_master] grid_white_to_blue failed; main animation falls back to default. {type(exc).__name__}: {exc}")

    # CHAR: prompt - per-grid-point activity views: for each (x,y,z), render a Feynman-style
    # CHAR: firegraph 3D animation of field activity (with edges based on pairwise interaction)
    # CHAR: plus a 2D line chart of per-field activity over time. Field count + labels are
    # CHAR: pulled from `output/ctlr/db_ctlr.json` (MODULES / FIELDS / AMOUNT_PARAMS_PER_FIELD
    # CHAR: / DB_KEYS) so the visualisation honours the live controller mapping.
    try:
        from color_master.grid_point_viz import render_gridpoint_visualizations
        # CHAR: ctlr lives next to the viz tree (same `output/` root). Compute path via repo layout
        # CHAR: instead of hardcoding so this works whether `out` is `output/visualizations/` or
        # CHAR: a custom override.
        repo_out_root = os.path.dirname(os.path.abspath(out))
        ctlr_candidate = os.path.join(repo_out_root, "ctlr", "db_ctlr.json")
        gridpoint_dir = render_gridpoint_visualizations(
            jax_guard=jax_guard,
            output_root=out,
            amount_nodes=n_nodes,
            ctlr_path=ctlr_candidate if os.path.isfile(ctlr_candidate) else None,
        )
        if gridpoint_dir:
            print(f"[color_master] per-grid-point viz -> {gridpoint_dir}")
    except Exception as exc:
        # CHAR: per-point viz is supplementary; never break the main workflow on its failure.
        print(f"[color_master] gridpoint viz failed: {type(exc).__name__}: {exc}")

    return out