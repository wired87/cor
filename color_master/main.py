"""
Prompt (2026-06): Backward-compatible entry shim — public API lives in submodules.

Legacy imports (`from color_master.main import ...`) re-export the refactored modules
so repo `main.py` and external callers keep working without touching `color_master/main.py`
name collision with the repo root entry.
"""
from __future__ import annotations

from color_master.path_viz import build_indexed_viz_from_engine_dict, run_path_based_viz
from color_master.render.demo_data import _demo_input_data
from color_master.render.grid_density import render_grid_white_to_blue
from color_master.render.timeseries import build_3d_time_series_visualization
from color_master.types import QUALITY_HIGH, QUALITY_LIGHT, FramePoints, PlotQuality
from color_master.workflow import run_workflow_visualization

__all__ = [
    "QUALITY_HIGH",
    "QUALITY_LIGHT",
    "FramePoints",
    "PlotQuality",
    "_demo_input_data",
    "build_3d_time_series_visualization",
    "build_indexed_viz_from_engine_dict",
    "render_grid_white_to_blue",
    "run_path_based_viz",
    "run_workflow_visualization",
]
