"""
Prompt (2026-06): Path-based viz from sim_cfg + local.json engine payload.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Optional
from color_master.render.indexed import _quality_from_preset, render_indexed_unified_animation
def build_indexed_viz_from_engine_dict(
    engine: dict[str, Any],
    output_dir: str | os.PathLike[str],
    *,
    quality_preset: str = "light",
    series: str = "values",
    max_indices: Optional[int] = None,
    max_frames: Optional[int] = None,
    subdir: str = "indexed",
) -> str:
    """Build single combined GIF from in-memory engine payload (param_series + ctlr)."""
    from color_master.engine_payload import prepare_indexed_viz

    out = Path(output_dir) / subdir
    out.mkdir(parents=True, exist_ok=True)
    q = _quality_from_preset(quality_preset)
    frames, labels, g_min, g_max, _ct = prepare_indexed_viz(
        engine, series=series, max_indices=max_indices, max_frames=max_frames
    )
    n = len(labels)
    if not frames or n == 0:
        print("[indexed-viz] no frames; skip")
        return str(out)
    render_indexed_unified_animation(
        frames=frames,
        labels=labels,
        value_range=(g_min, g_max),
        output_path=out / "environment_3d.gif",
        quality=q,
        n_params=n,
    )
    print(f"[indexed-viz] wrote {out / 'environment_3d.gif'}")
    return str(out)


def run_path_based_viz(sim_cfg_path: str | None = None) -> str:
    """
    Path-based pipeline: `sim_cfg.json` points at enriched `local.json` (param_series + ctlr).
    """
    from color_master.viz_config import load_sim_viz_config
    from color_master.engine_payload import load_engine_json, prepare_indexed_viz

    p = sim_cfg_path or os.environ.get("COLOR_MASTER_SIM_CFG", "sim_cfg.json")
    cfg = load_sim_viz_config(p)
    engine = load_engine_json(cfg.local_json)
    cm = cfg.color_master
    q = _quality_from_preset(cm.quality)
    fr, lab, g0, g1, _c = prepare_indexed_viz(
        engine,
        series=cm.series,
        max_indices=cm.max_indices,
        max_frames=cm.max_frames,
    )
    n = len(lab)
    base = Path(cfg.output_dir) / cm.output_subdir
    base.mkdir(parents=True, exist_ok=True)
    if n and fr:
        render_indexed_unified_animation(
            frames=fr,
            labels=lab,
            value_range=(g0, g1),
            output_path=base / "environment_3d.gif",
            quality=q,
            n_params=n,
        )
    else:
        print("[path-viz] no indexed frames; check param_series in local json")
    print(f"[path-viz] done -> {base}")
    return str(base)

