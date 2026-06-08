"""Backward-compat re-export — use color_master.engine_payload."""
from color_master.engine_payload import (
    ParamSeriesUnpacked,
    load_engine_json,
    prepare_indexed_viz,
    unpack_param_series,
)

__all__ = [
    "ParamSeriesUnpacked",
    "load_engine_json",
    "prepare_indexed_viz",
    "unpack_param_series",
]
