"""
Prompt (2026-06): Backward-compat alias — use color_master.workflow instead.

Bridge JAX simulation output -> color_master 3D time-series visualizations.
"""
from __future__ import annotations

from color_master.workflow import run_workflow_visualization

__all__ = ["run_workflow_visualization"]
