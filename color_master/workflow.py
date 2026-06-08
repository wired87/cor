"""
Prompt (2026-06): Post-simulation visualization workflow for the COR2 pipeline.

Collects series from JaxGuard, writes unified animation JSON, renders the main white→blue
grid GIF, and optionally per-key 3D series + per-grid-point activity views.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from color_master.render.demo_data import _demo_input_data
from color_master.render.grid_density import render_grid_white_to_blue
from color_master.render.timeseries import build_3d_time_series_visualization
from color_master.series_collect import (
    build_grid_value_volume,
    collect_series_from_jax_guard,
    to_numpy,
)
from color_master.types import QUALITY_HIGH, QUALITY_LIGHT, PlotQuality


def _gridpoint_viz_enabled() -> bool:
    # gien: per-grid-point viz is 27× heavy; skip unless COR_GRIDPOINT_VIZ=1
    raw = (os.environ.get("COR_GRIDPOINT_VIZ") or "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def run_workflow_visualization(
    output_dir: Optional[str] = None,
    *,
    jax_guard: Any = None,
    amount_nodes: Optional[int] = None,
    dims: Optional[int] = None,
    quality_preset: str = "light",
    out_name: str = "single_animation",
) -> str:
    repo_parent = os.path.dirname(os.path.abspath(__file__))
    default_out = os.path.join(os.path.dirname(repo_parent), "color_master_output")
    out = output_dir or os.environ.get("COLOR_MASTER_OUT", default_out)
    os.makedirs(out, exist_ok=True)

    n_nodes = amount_nodes if amount_nodes is not None else int(os.environ.get("AMOUNT_NODES", "4"))
    n_dims = dims if dims is not None else int(os.environ.get("DIMS", "3"))

    data: Dict[str, List[Any]] = {}
    if jax_guard is not None:
        data = collect_series_from_jax_guard(jax_guard)

    if not data:
        t_steps = max(8, int(os.environ.get("SIM_TIME", "24")))
        print(f"[color_master] no simulation series; using demo data (timesteps={t_steps})")
        data = _demo_input_data(timesteps=t_steps)

    max_len = max(len(v) for v in data.values())
    frames: List[List[List[float]]] = []

    for t in range(max_len):
        frame: List[List[float]] = []
        for key, series in data.items():
            if t >= len(series):
                continue
            val = series[t]
            arr = to_numpy(val)
            if arr is None:
                continue
            flat = arr.astype(np.float32).ravel()
            for i, v in enumerate(flat):
                frame.append([
                    i % n_nodes,
                    (i // n_nodes) % n_nodes,
                    (i // (n_nodes * n_nodes)) % max(n_dims, 3),
                    float(v),
                ])
        frames.append(frame)

    print(f"[color_master] unified frames={len(frames)}")

    brainmaster_path = os.path.join(out, f"{out_name}.json")
    with open(brainmaster_path, "w") as f:
        json.dump(frames, f)
    print(f"[color_master] brainmaster saved -> {brainmaster_path}")

    if quality_preset != "light":
        build_3d_time_series_visualization(
            data={"merged": frames},
            amount_nodes=n_nodes,
            dims=max(n_dims, 3),
            output_dir=out,
            quality=None,
            quality_preset=quality_preset,
        )
        print(f"[color_master] single animation output -> {out}")
    else:
        print("[color_master] light preset: skipping per-key 3D build (main white->blue gif only)")

    try:
        volume = build_grid_value_volume(jax_guard, n_nodes)
        if volume is None:
            print("[color_master] grid_white_to_blue: no feature history; main animation kept as default combined view")
        else:
            if quality_preset == "light":
                grid_quality = QUALITY_LIGHT
            elif quality_preset == "high":
                grid_quality = QUALITY_HIGH
            else:
                grid_quality = PlotQuality()
            grid_out = Path(out) / "environment_3d.gif"
            render_grid_white_to_blue(
                volume=volume,
                output_path=grid_out,
                quality=grid_quality,
                amount_nodes=n_nodes,
                title="Main animation | grid feature density (white -> blue)",
            )
            print(f"[color_master] main animation (white->blue grid) -> {grid_out}")
    except Exception as exc:
        print(f"[color_master] grid_white_to_blue failed; main animation falls back to default. {type(exc).__name__}: {exc}")

    if _gridpoint_viz_enabled():
        try:
            from color_master.grid_point import render_gridpoint_visualizations

            repo_out_root = os.path.abspath(out)
            ctlr_candidate = os.path.join(repo_out_root, "db_ctlr.json")
            gridpoint_dir = render_gridpoint_visualizations(
                jax_guard=jax_guard,
                output_root=out,
                amount_nodes=n_nodes,
                ctlr_path=ctlr_candidate if os.path.isfile(ctlr_candidate) else None,
            )
            if gridpoint_dir:
                print(f"[color_master] per-grid-point viz -> {gridpoint_dir}")
        except Exception as exc:
            print(f"[color_master] gridpoint viz failed: {type(exc).__name__}: {exc}")
    else:
        print("[color_master] gridpoint viz skipped (set COR_GRIDPOINT_VIZ=1 to enable)")

    return out
