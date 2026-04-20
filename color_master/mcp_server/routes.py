"""MCP server: single create route for 3D time-series visualization."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from main import build_3d_time_series_visualization, _demo_input_data

APP_NAME = "color-master-viz"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080
DEFAULT_PATH = "/mcp"


def _port() -> int:
    raw = os.getenv("PORT") or os.getenv("MCP_PORT") or str(DEFAULT_PORT)
    try:
        return int(raw)
    except Exception:
        print("Err color_master.mcp_server.routes::_port | handler_line=31 | Exception handler triggered")
        print("[exception] color_master.mcp_server.routes._port: caught Exception")
        return DEFAULT_PORT


def _ensure_numpy(obj: Any) -> Any:
    if isinstance(obj, list):
        return np.array(obj)
    if isinstance(obj, dict):
        return {k: _ensure_numpy(v) for k, v in obj.items()}
    return obj


def _prepare(data: dict[str, list[Any]]) -> dict[str, list[Any]]:
    return {k: [_ensure_numpy(item) for item in series] for k, series in data.items()}


app = FastMCP(
    name=APP_NAME,
    instructions="Single create tool: pass dict[str, list] (SOA time-series). Runs 3D viz pipeline, returns output paths.",
    host=os.getenv("MCP_HOST", DEFAULT_HOST),
    port=_port(),
    streamable_http_path=os.getenv("MCP_PATH", DEFAULT_PATH),
    json_response=True,
    stateless_http=True,
)


@app.custom_route("/health", methods=["GET"], include_in_schema=False)
async def _health(_: Request) -> Response:
    return JSONResponse({"status": "ok", "app": APP_NAME, "mcp_path": os.getenv("MCP_PATH", DEFAULT_PATH)})


@app.custom_route("/status", methods=["GET"], include_in_schema=False)
async def _status(_: Request) -> Response:
    """FastMCP application status for Docker startup checks."""
    return JSONResponse({
        "status": "ok",
        "app": APP_NAME,
        "mcp_path": os.getenv("MCP_PATH", DEFAULT_PATH),
        "host": os.getenv("MCP_HOST", DEFAULT_HOST),
        "port": _port(),
    })


@app.tool()
def create(
    data: dict[str, list[Any]],
    amount_nodes: int = 28,
    dims: int = 360,
    output_dir: str = "output_dir",
    quality_preset: str = "default",
    use_demo_if_empty: bool = True,
) -> dict[str, Any]:
    """
    Run 3D time-series visualization. Input: dict[str, list] (SOA).
    Each key -> list of timestep items (lists, scalars, dicts, arrays).
    If data is empty and use_demo_if_empty=True, runs hardcoded demo.
    Returns minimal: {ok, out, static[], anim[], combined}.
    """
    try:
        if not data and use_demo_if_empty:
            data = _demo_input_data(timesteps=20)
        if not data:
            return {"ok": False, "err": "empty data"}
        prepared = _prepare(data)
        build_3d_time_series_visualization(
            data=prepared,
            amount_nodes=amount_nodes,
            dims=dims,
            output_dir=output_dir,
            quality_preset=quality_preset,
        )
        out = Path(output_dir)
        keys = list(data.keys())
        return {
            "ok": True,
            "out": str(out),
            "static": [str(out / "per_key_static" / f"{k}_3d.png") for k in keys],
            "anim": [str(out / "per_key_animation" / f"{k}_3d.gif") for k in keys],
            "combined": str(out / "combined" / "environment_3d.gif"),
        }
    except Exception as e:
        print(f"Err color_master.mcp_server.routes::create | handler_line=113 | {type(e).__name__}: {e}")
        print(f"[exception] color_master.mcp_server.routes.create: {e}")
        return {"ok": False, "err": str(e)}


if __name__ == "__main__":
    host = os.getenv("MCP_HOST", DEFAULT_HOST)
    port = _port()
    path = os.getenv("MCP_PATH", DEFAULT_PATH)
    print(f"[mcp] {APP_NAME} host={host} port={port} path={path}", file=sys.stderr)
    print(f"[mcp] status http://{host}:{port}/status", file=sys.stderr)
    app.run(transport="streamable-http")
