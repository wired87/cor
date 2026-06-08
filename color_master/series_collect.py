"""
Prompt (2026-06): Collect simulation time-series and grid volumes from JaxGuard for visualization.

Extracts param histories, feature-encoder series, and per-grid-point feature density volumes
from the live JAX engine without requiring file exports.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def to_numpy(x: Any) -> Optional[np.ndarray]:
    """Normalize JAX/NumPy/scalar payloads to a NumPy ndarray."""
    if x is None:
        return None
    try:
        import jax.numpy as jnp

        if isinstance(x, jnp.ndarray):
            return np.asarray(x)
    except Exception:
        pass
    try:
        return np.asarray(x)
    except Exception:
        return None


def series_from_param_histories(dl: Any, max_keys: Optional[int] = None) -> Dict[str, List[Any]]:
    """Build keyed time-series from DBLayer param value/feature histories."""
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


def series_from_feature_encoder(gnn: Any, max_eq: int, max_steps: int) -> Dict[str, List[Any]]:
    """Fallback series from feature_encoder.in_store when param histories are empty."""
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
        for _t, feat in enumerate(slot0[:max_steps]):
            arr = to_numpy(feat)
            if arr is None:
                continue
            flat = np.asarray(arr, dtype=np.float32).ravel()
            if flat.size == 0:
                continue
            series.append(flat)
        if len(series) >= 2:
            out[f"encoder_in_eq{eq_i}"] = series
    return out


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

    data = series_from_param_histories(dl, max_param_keys)
    if not data:
        data = series_from_feature_encoder(gnn, max_eq=6, max_steps=500)
    return data


def build_grid_value_volume(jax_guard: Any, amount_nodes: int) -> Optional[np.ndarray]:
    """Aggregate per-timestep scalar value at every (x, y, z) grid point.

    Walks `gnn._features_history` and folds absolute-valued flat buffers into chunks
    of `N**3`, then sums chunks along the partition axis.

    Returns ndarray shape (T, N, N, N) or None when no feature history is available.
    """
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
    for step_features in history:
        flat_chunks: List[np.ndarray] = []
        for feat in step_features or []:
            arr = to_numpy(feat)
            if arr is None:
                continue
            v = np.abs(np.asarray(arr, dtype=np.float32).ravel())
            if v.size == 0:
                continue
            if not np.all(np.isfinite(v)):
                v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            flat_chunks.append(v)

        if not flat_chunks:
            volume_steps.append(np.zeros((amount_nodes, amount_nodes, amount_nodes), dtype=np.float32))
            continue

        flat = np.concatenate(flat_chunks)
        pad = (-flat.size) % grid_size
        if pad:
            flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
        chunks = flat.reshape(-1, amount_nodes, amount_nodes, amount_nodes)
        volume_steps.append(chunks.sum(axis=0).astype(np.float32))

    if not volume_steps:
        return None
    return np.stack(volume_steps, axis=0)
