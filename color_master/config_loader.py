"""Backward-compat re-export — use color_master.viz_config."""
from color_master.viz_config import ColorMasterVizConfig, SimVizConfig, load_sim_viz_config

__all__ = ["ColorMasterVizConfig", "SimVizConfig", "load_sim_viz_config"]
