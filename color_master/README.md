# Color Master

3D time-series visualization with MCP server. Single `create` tool: `dict[str, list]` (SOA) → PNG/GIF outputs.

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
