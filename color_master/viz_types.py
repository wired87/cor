"""Shared dataclasses for color_master (avoids circular imports with engine_json)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FramePoints:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    value: np.ndarray


@dataclass
class PlotQuality:
    dpi: int = 300
    min_dpi: int = 200
    point_size_static: float = 40
    point_size_anim: float = 32
    point_size_combined: float = 24
    alpha_low: float = 0.15
    alpha_high: float = 0.95
    rasterize_threshold: int = 500


QUALITY_LIGHT = PlotQuality(dpi=150, point_size_static=35, point_size_anim=28, point_size_combined=20)
QUALITY_HIGH = PlotQuality(dpi=400, point_size_static=45, point_size_anim=36, point_size_combined=28)
