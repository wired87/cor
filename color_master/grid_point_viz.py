"""Backward-compat re-export — use color_master.grid_point."""
from color_master.grid_point import (
    build_field_volume,
    render_freq_chart_2d,
    render_grid_point_activity_views,
    render_gridpoint_3d_animation,
    render_gridpoint_visualizations,
)

__all__ = [
    "build_field_volume",
    "render_freq_chart_2d",
    "render_grid_point_activity_views",
    "render_gridpoint_3d_animation",
    "render_gridpoint_visualizations",
]
