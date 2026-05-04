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
)


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
) -> str:
    """
    Run color_master 3D exports. Returns output directory used.
    """
    # gien: default viz root is sibling of `color_master/` inside the repo (cor2/color_master_output)
    repo_parent = os.path.dirname(os.path.abspath(__file__))
    default_out = os.path.join(os.path.dirname(repo_parent), "color_master_output")
    # gien: allow explicit dir, else env COLOR_MASTER_OUT, else deterministic default
    out = output_dir or os.environ.get("COLOR_MASTER_OUT", default_out)

    # gien: caller wins over env so main.py sim geometry matches charts without extra shell exports
    n_nodes = amount_nodes if amount_nodes is not None else int(os.environ.get("AMOUNT_NODES", "4"))
    n_dims = dims if dims is not None else int(os.environ.get("DIMS", "3"))

    if jax_guard is not None:
        eng = collect_engine_payload_from_jax_guard(jax_guard)
        if eng:
            _mxi = os.environ.get("COLOR_MASTER_MAX_INDICES")
            _mxf = os.environ.get("COLOR_MASTER_MAX_FRAMES")
            max_i = int(_mxi) if _mxi and str(_mxi).isdigit() else None
            max_f = int(_mxf) if _mxf and str(_mxf).isdigit() else None
            _ser = (os.environ.get("COLOR_MASTER_SERIES") or "values").lower()
            if _ser not in ("values", "features"):
                _ser = "values"
            build_indexed_viz_from_engine_dict(
                eng,
                out,
                quality_preset=quality_preset,
                series=_ser,
                max_indices=max_i,
                max_frames=max_f,
                subdir="indexed",
            )
            print(f"[color_master] indexed ctlr visualization output -> {out}/indexed")
            return out

    data: Dict[str, List[Any]] = {}
    if jax_guard is not None:
        data = collect_series_from_jax_guard(jax_guard)

    if not data:
        # gien: keep demo horizon loosely aligned with engine SIM_TIME when series are empty
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
