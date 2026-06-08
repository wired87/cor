"""
Path-based color_master: load and merge `sim_cfg.json` with defaults.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ColorMasterVizConfig:
    quality: str = "light"  # light | default | high
    series: str = "values"  # values | features
    max_indices: Optional[int] = None
    max_frames: Optional[int] = None
    unified_gif: bool = True
    output_subdir: str = "indexed"


@dataclass
class SimVizConfig:
    local_json: str
    output_dir: str
    color_master: ColorMasterVizConfig = field(default_factory=ColorMasterVizConfig)


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _default_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_sim_viz_config(
    sim_cfg_path: str | os.PathLike[str],
) -> SimVizConfig:
    p = Path(sim_cfg_path)
    base_dir = p.resolve().parent if p.is_file() else p.resolve()
    raw: Dict[str, Any] = {}
    if p.is_file():
        raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("sim_cfg must be a JSON object")
    d = _deep_merge(
        {
            "local_json": str(_default_root() / "local.json"),
            "output_dir": str(_default_root() / "color_master_output"),
            "color_master": {
                "quality": "light",
                "series": "values",
                "max_indices": None,
                "max_frames": None,
                "unified_gif": True,
                "output_subdir": "indexed",
            },
        },
        raw,
    )
    cm = d.get("color_master") or {}
    if not isinstance(cm, dict):
        cm = {}
    cm_cfg = ColorMasterVizConfig(
        quality=str(cm.get("quality", "light") or "light"),
        series=str(cm.get("series", "values") or "values").lower(),
        max_indices=cm.get("max_indices"),
        max_frames=cm.get("max_frames"),
        unified_gif=bool(cm.get("unified_gif", True)),
        output_subdir=str(cm.get("output_subdir", "indexed") or "indexed"),
    )
    if cm_cfg.series not in ("values", "features"):
        cm_cfg.series = "values"

    lj = d.get("local_json") or ""
    odir = d.get("output_dir") or "color_master_output"
    if lj and not Path(str(lj)).is_absolute():
        lj = str((base_dir / str(lj)).resolve())
    if odir and not Path(str(odir)).is_absolute():
        odir = str((base_dir / str(odir)).resolve())

    return SimVizConfig(
        local_json=str(lj),
        output_dir=str(odir),
        color_master=cm_cfg,
    )
