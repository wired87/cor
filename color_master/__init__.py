"""
Prompt (2026-06): color_master — 3D visualization for COR2 simulation output.

Public API:
  - run_workflow_visualization  — post-JaxGuard pipeline (main white→blue GIF + optional extras)
  - run_path_based_viz          — offline indexed GIF from sim_cfg + local.json
  - build_indexed_viz_from_engine_dict — in-memory indexed GIF from engine payload
"""
from __future__ import annotations

from color_master.path_viz import build_indexed_viz_from_engine_dict, run_path_based_viz
from color_master.workflow import run_workflow_visualization

__all__ = [
    "build_indexed_viz_from_engine_dict",
    "run_path_based_viz",
    "run_workflow_visualization",
]
