"""
Bridge JAX simulation output -> color_master 3D time-series visualizations.

Loaded with color_master on sys.path so `from main import ...` resolves to color_master/main.py.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np


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


def _series_from_param_histories(dl: Any, max_keys: int) -> Dict[str, List[Any]]:
    out: Dict[str, List[Any]] = {}
    pv = getattr(dl, "param_values_history", None) or {}
    pf = getattr(dl, "param_features_history", None) or {}
    added = 0
    for name, src in (("value", pv), ("feature", pf)):
        sk = sorted(src.keys(), key=lambda x: (str(type(x).__name__), str(x)))
        for k in sk:
            if added >= max_keys:
                break
            series = src.get(k)
            if not series:
                continue
            out[f"param_{name}_{k}"] = [float(v) for v in series]
            added += 1
        if added >= max_keys:
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


def collect_series_from_jax_guard(guard: Any, max_param_keys: int = 16) -> Dict[str, List[Any]]:
    """Build dict[str, list[timestep_payload]] for build_3d_time_series_visualization."""
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
    jax_guard: Optional[Any],
    *,
    output_dir: Optional[str] = None,
    amount_nodes: Optional[int] = None,
    dims: Optional[int] = None,
    quality_preset: str = "light",
    _demo_input_data,
    QUALITY_LIGHT,
    build_3d_time_series_visualization,
) -> str:
    """
    Run color_master 3D exports. Returns output directory used.
    """
    repo_parent = os.path.dirname(os.path.abspath(__file__))
    default_out = os.path.join(os.path.dirname(repo_parent), "color_master_output")
    out = output_dir or os.environ.get("COLOR_MASTER_OUT", default_out)

    n_nodes = amount_nodes if amount_nodes is not None else int(os.environ.get("AMOUNT_NODES", "4"))
    n_dims = dims if dims is not None else int(os.environ.get("DIMS", "3"))

    data: Dict[str, List[Any]] = {}
    if jax_guard is not None:
        data = collect_series_from_jax_guard(jax_guard)

    if not data:
        t_steps = max(8, int(os.environ.get("SIM_TIME", "24")))
        print(f"[color_master] no simulation series; using demo data (timesteps={t_steps})")
        data = _demo_input_data(timesteps=t_steps)

    vkw: Dict[str, Any] = dict(
        data=data,
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
    print(f"[color_master] visualization output -> {out}")
    return out
