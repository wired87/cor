# Color Master

3D visualization for COR2 simulation output: per-key time series, white→blue grid density, indexed controller animations, and optional per-grid-point activity views.

## Progress

- 2026-06-07: Modular refactor — removed unused MCP/Docker artifacts; split monolithic `main.py` into focused modules:
  - `workflow.py` — `run_workflow_visualization` (post-JaxGuard pipeline)
  - `series_collect.py` — JaxGuard series + grid volume extraction
  - `path_viz.py` — offline indexed GIF from `sim_cfg.json` + `local.json`
  - `render/{timeseries,indexed,grid_density,helpers,demo_data}.py` — matplotlib renderers
  - `types.py`, `engine_payload.py`, `viz_config.py`, `grid_point.py` — shared types and I/O
  - `main.py` / `sim_bridge.py` / `viz_types.py` / `engine_json.py` / `config_loader.py` / `grid_point_viz.py` — backward-compat shims
  - Repo `main.py` imports `color_master.workflow` directly.

## Layout

```
color_master/
  workflow.py           # run_workflow_visualization
  series_collect.py     # collect_series_from_jax_guard, build_grid_value_volume
  path_viz.py           # run_path_based_viz, build_indexed_viz_from_engine_dict
  render/               # matplotlib renderers
  types.py              # FramePoints, PlotQuality
  engine_payload.py     # local.json / param_series unpacking
  viz_config.py         # sim_cfg.json loader
  grid_point.py         # per-grid-point 3D + 2D activity views
```

## Quick Start

```bash
pip install -r requirements.txt
```

From the repo root after a simulation run:

```python
from color_master.workflow import run_workflow_visualization
run_workflow_visualization(output_dir="output", jax_guard=guard, amount_nodes=3, dims=3)
```

## Env

| Var | Default | Effect |
|-----|---------|--------|
| `COLOR_MASTER_OUT` | `../color_master_output` | Default viz output dir |
| `COR_GRIDPOINT_VIZ` | `0` | Set `1` to enable per-grid-point GIFs (27× heavier) |
| `COLOR_MASTER_SIM_CFG` | `sim_cfg.json` | Path-based indexed viz config |
