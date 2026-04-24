"""
Single-file MCP server wrapper for the repo `main.py` workflow.

Exposes ONE route/tool that runs the wrapped `run_main_process()` and returns its result.
Uses FastMCP (streamable HTTP by default).
"""

from __future__ import annotations

import os
import sys
import importlib.util
from pathlib import Path
from typing import Any, Optional, Dict, cast, Tuple

# gien: this file name is `mcp.py` which would shadow the installed `mcp` package.
# We temporarily remove the repo root from sys.path so `from mcp...` resolves to site-packages.
_THIS_DIR = Path(__file__).resolve().parent
_REMOVED_SYSPATH: list[str] = []
for _p in (str(_THIS_DIR), ""):
    while _p in sys.path:
        sys.path.remove(_p)
        _REMOVED_SYSPATH.append(_p)

from mcp.server.fastmcp import FastMCP  # noqa: E402
from starlette.requests import Request  # noqa: E402
from starlette.responses import JSONResponse, Response  # noqa: E402

for _p in reversed(_REMOVED_SYSPATH):
    sys.path.insert(0, _p)

# gien: tree bootstrap (repo root + cor + color_master) before importing main workflow.
_REPO_ROOT = _THIS_DIR
_CORE = str(_REPO_ROOT / "cor")
_COLOR_MASTER = str(_REPO_ROOT / "color_master")
for _p in (str(_REPO_ROOT), _CORE, _COLOR_MASTER):
    if _p not in sys.path:
        sys.path.insert(0, _p)

def _load_repo_root_main() -> Any:
    """
    Load repo-root main.py explicitly to avoid shadowing by color_master/main.py
    when color_master is on sys.path (common in container images).
    """
    main_path = _REPO_ROOT / "main.py"
    spec = importlib.util.spec_from_file_location("cor2_repo_root_main", main_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load repo root main.py from {main_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


_root_main = _load_repo_root_main()
run_main_process = getattr(_root_main, "run_main_process")

APP_NAME = "cor2-main"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080
DEFAULT_PATH = "/mcp"

def _port() -> int:
    """
    Cloud Run sets PORT. We also support MCP_PORT for local dev.
    """
    raw = os.getenv("PORT") or os.getenv("MCP_PORT") or str(DEFAULT_PORT)
    try:
        return int(raw)
    except Exception:
        print("Err mcp::_port | handler_line=69 | Exception handler triggered")
        print("[exception] mcp._port: caught Exception")
        return DEFAULT_PORT

app = FastMCP(
    name=APP_NAME,
    instructions="Single-route wrapper around repo main.py run_main_process().",
    host=os.getenv("MCP_HOST", DEFAULT_HOST),
    port=_port(),
    streamable_http_path=os.getenv("MCP_PATH", DEFAULT_PATH),
    json_response=True,
    stateless_http=True,
)


@app.custom_route("/status", methods=["GET"], include_in_schema=False)
async def _status(_: Request) -> Response:
    return JSONResponse(
        {
            "status": "ok",
            "app": APP_NAME,
            "mcp_path": os.getenv("MCP_PATH", DEFAULT_PATH),
            "host": os.getenv("MCP_HOST", DEFAULT_HOST),
            "port": _port(),
            "transport": (os.getenv("MCP_TRANSPORT", "streamable-http") or "streamable-http"),
        }
    )


@app.custom_route("/health", methods=["GET"], include_in_schema=False)
async def _health(_: Request) -> Response:
    # Cloud Run readiness/liveness probes often prefer a stable /health endpoint.
    return JSONResponse({"status": "ok"})


def _as_int(v: Any, default: Optional[int] = None) -> Optional[int]:
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        print("Err mcp::_as_int | handler_line=109 | Exception handler triggered")
        print("[exception] mcp._as_int: caught Exception")
        return default


def _normalize_run_payload(payload: Dict[str, Any]) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Preferred body: { "sim_spec": {...}, "injection_file": {...} }.
    Legacy flat keys still accepted (mapped into sim_spec).
    """
    sim_spec = payload.get("sim_spec")
    if not isinstance(sim_spec, dict):
        sim_spec = {}
    if not sim_spec and any(
        k in payload for k in ("output_dir", "amount_nodes", "sim_time", "dims", "user_id")
    ):
        sim_spec = {
            k: payload[k]
            for k in ("output_dir", "amount_nodes", "sim_time", "dims", "user_id")
            if k in payload
        }
    inj = payload.get("injection_file")
    injection_file = cast(Optional[Dict[str, Any]], inj if isinstance(inj, dict) else None)
    return sim_spec, injection_file


@app.custom_route("/run", methods=["POST"], include_in_schema=False)
async def _run_route(req: Request) -> Response:
    payload: Dict[str, Any] = {}
    try:
        raw = await req.json()
        if isinstance(raw, dict):
            payload = raw
    except Exception:
        print("Err mcp::_run_route | handler_line=142 | Exception handler triggered")
        print("[exception] mcp._run_route: caught Exception")
        payload = {}

    sim_spec, injection_file = _normalize_run_payload(payload)
    result = run_main_process(
        output_dir=sim_spec.get("output_dir") if isinstance(sim_spec.get("output_dir"), (str, type(None))) else None,
        amount_nodes=_as_int(sim_spec.get("amount_nodes")),
        sim_time=_as_int(sim_spec.get("sim_time")),
        dims=_as_int(sim_spec.get("dims")),
        user_id=int(sim_spec.get("user_id", 1)),
        injection_cfg=injection_file,
    )
    return JSONResponse(result)


@app.tool(name="run")
def run(
    sim_spec: Optional[Dict[str, Any]] = None,
    injection_file: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run workflow: sim_spec (dims, sim_time, …) + injection_file merged into final sim cfg."""
    spec = sim_spec if isinstance(sim_spec, dict) else {}
    inj = injection_file if isinstance(injection_file, dict) else None
    return run_main_process(
        output_dir=spec.get("output_dir") if isinstance(spec.get("output_dir"), (str, type(None))) else None,
        amount_nodes=_as_int(spec.get("amount_nodes")),
        sim_time=_as_int(spec.get("sim_time")),
        dims=_as_int(spec.get("dims")),
        user_id=int(spec.get("user_id", 1)),
        injection_cfg=inj,
    )


def main() -> None:
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    transport = (os.getenv("MCP_TRANSPORT", "streamable-http") or "streamable-http").strip().lower()
    if transport not in {"stdio", "streamable-http"}:
        print(f"[mcp] invalid MCP_TRANSPORT={transport!r}; falling back to streamable-http")
        transport = "streamable-http"
    print(f"[mcp] starting FastMCP transport={transport!r} http=http://{app.settings.host}:{app.settings.port}{app.settings.streamable_http_path}")
    app.run(transport=transport)


if __name__ == "__main__":
    main()

