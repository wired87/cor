"""
Load enriched `local.json` and build indexed 3D frames (param index x value x module/field band).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ParamSeriesUnpacked:
    order: List[str]
    param_indices: List[int]
    series: Dict[str, Dict[str, List[float]]]
    field: str  # values | features


def _as_list(x: Any) -> list:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def ctlr_mf_bands(ctlr_db: Dict[str, Any]) -> List[Tuple[int, int]]:
    """
    Same (module, field) order as jax_test.grid.live_payload._flat_keys_from_cfg
    for one scalar per (mi,fi,param_slot).
    """
    amount_per_field = [int(x) for x in _as_list(ctlr_db.get("AMOUNT_PARAMS_PER_FIELD"))]
    modules = [int(x) for x in _as_list(ctlr_db.get("MODULES") or [0])]
    fields = [int(x) for x in _as_list(ctlr_db.get("FIELDS") or [1])]
    n_modules = max(1, max(modules) + 1) if modules else 1
    n_fields = max(1, max(fields)) if fields else 1
    bands: List[Tuple[int, int]] = []
    for mi in range(n_modules):
        for fi in range(n_fields):
            flat_idx = mi * n_fields + fi
            n_params = int(amount_per_field[flat_idx]) if flat_idx < len(amount_per_field) else 1
            for _ in range(n_params):
                bands.append((mi, fi))
    return bands


def unpack_param_series(obj: Any) -> Optional[ParamSeriesUnpacked]:
    if not isinstance(obj, dict) or not obj:
        return None
    if "order" in obj and "series" in obj:
        order = [str(c) for c in obj.get("order") or []]
        series = obj.get("series") or {}
        if not isinstance(series, dict):
            return None
        pi = obj.get("param_indices")
        param_indices = [int(x) for x in pi] if isinstance(pi, list) and pi else list(range(len(order)))
        return ParamSeriesUnpacked(order=order, param_indices=param_indices, series=series, field="values")
    # --- legacy flat: { col: { values, features } } (no stable order) ---
    keys = [k for k in obj if not k.startswith("_") and isinstance(obj[k], dict)]
    if not keys:
        return None
    keys = sorted(keys)
    series: Dict[str, Any] = {k: obj[k] for k in keys if isinstance(obj[k], dict)}
    return ParamSeriesUnpacked(
        order=keys,
        param_indices=list(range(len(keys))),
        series=series,
        field="values",
    )


def min_timestep_count(ps: ParamSeriesUnpacked) -> int:
    order = ps.order
    s = ps.series
    f = ps.field
    if f not in ("values", "features"):
        f = "values"
    lens: List[int] = []
    for c in order:
        row = s.get(c) or {}
        a = row.get(f) if f in row else row.get("values")
        if a is not None and isinstance(a, (list, tuple)) and len(a) > 0:
            lens.append(len(a))
    return min(lens) if lens else 0


def build_timestep_value_matrix(
    ps: ParamSeriesUnpacked, T: int, max_indices: Optional[int] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    shape (T, N) in param index order. Shorter series rows are padded with NaN up to T.
    """
    order = ps.order
    s = ps.series
    f = ps.field if ps.field in ("values", "features") else "values"
    n_all = len(order)
    if max_indices is not None and max_indices > 0:
        n = min(n_all, int(max_indices))
    else:
        n = n_all
    order2 = order[:n]
    mat = np.zeros((T, n), dtype=np.float64)
    for j, c in enumerate(order2):
        row = s.get(c) or {}
        arr = list(row.get(f) or row.get("values") or [])
        for t in range(T):
            mat[t, j] = float(arr[t]) if t < len(arr) else float("nan")
    return mat, order2


def load_engine_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"local engine json not found: {p}")
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def prepare_indexed_viz(
    engine: Dict[str, Any],
    *,
    series: str = "values",
    max_indices: Optional[int] = None,
    max_frames: Optional[int] = None,
) -> Tuple[List[Any], List[str], float, float, Dict[str, Any]]:
    from color_master.viz_types import FramePoints

    ps_raw = engine.get("param_series")
    if not ps_raw:
        raise ValueError("param_series missing in engine json (re-run simulation with EXPORT_PARAM_SERIES or export enabled)")
    ps = unpack_param_series(ps_raw)
    if not ps or not ps.order:
        raise ValueError("param_series is empty or invalid")
    ps.field = series if series in ("values", "features") else "values"
    T0 = min_timestep_count(ps)
    if T0 <= 0 and ps.field == "values":
        # DBLayer often records features every step; values need nodes slice and may be empty.
        ps.field = "features"
        T0 = min_timestep_count(ps)
        if T0 > 0:
            print("[engine_json] param_series: using 'features' (no 'values' timesteps)")
    if T0 <= 0:
        raise ValueError("param_series has no timestep data (values and features both empty?)")
    T = T0
    if max_frames is not None and max_frames > 0:
        T = min(T, int(max_frames))
    mat, labels = build_timestep_value_matrix(ps, T, max_indices=max_indices)
    T, n = mat.shape
    ctlr = engine.get("ctlr") or {}
    if not isinstance(ctlr, dict):
        ctlr = {}
    db = ctlr.get("db") or ctlr
    bands = ctlr_mf_bands(db if isinstance(db, dict) else {})
    n_fields = max((b[1] for b in bands[:n]), default=0) + 1
    z_scale = float(max(1, n_fields))
    frames: List[FramePoints] = []
    valid = mat[np.isfinite(mat)]
    if valid.size:
        g_min = float(np.min(valid))
        g_max = float(np.max(valid))
    else:
        g_min, g_max = 0.0, 1.0
    if not (np.isfinite(g_min) and np.isfinite(g_max)):
        g_min, g_max = 0.0, 1.0
    if np.isclose(g_min, g_max):
        g_max = g_min + 1.0
    for t in range(T):
        x = np.arange(n, dtype=np.float64)
        y = mat[t, :].copy()
        z = np.zeros(n, dtype=np.float64)
        for i in range(n):
            mi, fi = bands[i] if i < len(bands) else (0, 0)
            z[i] = float(mi) * z_scale + float(fi)
        val = np.abs(np.nan_to_num(y, nan=0.0))
        frames.append(FramePoints(x=x, y=y, z=z, value=val))
    return frames, labels, g_min, g_max, ctlr
