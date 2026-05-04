# Color Master

3D time-series visualization with MCP server. Single `create` tool: `dict[str, list]` (SOA) → PNG/GIF outputs.

## Progress

- 2026-05-03: Repo `main.py` now calls `run_workflow_visualization` after `JaxGuard` (see `color_master/sim_bridge.py`); bridge had broken free variables (`jax_guard`, `amount_nodes`, …) and is restored with explicit kwargs. `Guard` writes `sim_config.json` next to repo root again (no double `.json`, no parent-folder path). Windows: `main._configure_stdio_utf8` avoids `UnicodeEncodeError` when printing large `components` (e.g. μ in strings). `run_main_process` accepts MCP’s `injection_cfg`, `run_visualization`, `visualization_dir` and omits non-JSON `jax_guard` from the return dict. Full run log: `test_out/main_run_console.txt` (use `MPLBACKEND=Agg` headless). Env: `COLOR_MASTER_VIZ=0` skips viz; `SIM_TIME` aligns demo fallback timesteps when DB series are empty.

## Workflow

```
data (dict[str, list]) → create() → output_dir/
  ├── per_key_static/
  ├── per_key_animation/
  └── combined/environment_3d.gif
```

## Quick Start

```bash
pip install -r r.txt
python main.py
```

## MCP Server

```bash
cp .env.example .env   # optional
python mcp_server/routes.py
```

**Tool:** `create(data, amount_nodes=28, dims=360, output_dir="output_dir", quality_preset="default", use_demo_if_empty=True)`

- `data`: `dict[str, list]` – each key maps to a list of timestep items (lists, scalars, dicts, arrays)
- If `data` is empty and `use_demo_if_empty=True`, runs hardcoded demo
- Returns: `{ok, out, static[], anim[], combined}` or `{ok: false, err}`

**Endpoints:**
- MCP: `http://localhost:8080/mcp`
- Status: `http://localhost:8080/status`
- Health: `http://localhost:8080/health`

## Docker

```bash
docker build -t color-master-mcp .
docker run -p 8000:8000 color-master-mcp
```

Check status at startup: `curl http://localhost:8080/status`

## Test

```bash
python test_mcp.py
```

## Env

| Var | Default |
|-----|---------|
| MCP_HOST | 0.0.0.0 |
| MCP_PORT | 8080 |
| MCP_PATH | /mcp |
