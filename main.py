"""
Create CFG from Module G
Simulate (JAX)
color_master 3D visualizations -> color_master_output/ (sim series or demo fallback)


Err Node.__call__.cor 'NoneType' object is not callable
Err calc_batch: 'NoneType' object is not iterable
save_t_step...
Err flatten_result: 'NoneType' object is not iterable
sum_results...
Err sum_results: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (37,) + inhomogeneous part.
stack_tdb... started
"""

import os
import sys
import json
import base64
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from firegraph.graph import GUtils
from guard import Guard as CG
from injector import Injector

from qfu.qf_utils import QFUtils
from sm_manager.sm_manager import SMManager
from jax_test.guard import JaxGuard

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_CORE = os.path.join(_REPO_ROOT, "cor")
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)
_COLOR_MASTER = os.path.join(_REPO_ROOT, "color_master")
if _COLOR_MASTER not in sys.path:
    sys.path.insert(0, _COLOR_MASTER)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _step(name: str, **details: Any) -> None:
    if details:
        print(f"[main][step] {name} | {details}", flush=True)
    else:
        print(f"[main][step] {name}", flush=True)


def _slurp_visualizations(tmp_out_dir: str) -> Dict[str, Any]:
    """
    Read rendered artifacts from a color_master output folder and return them inline (base64).
    Layout expected:
      per_key_static/*.png
      per_key_animation/*.gif
      combined/*.gif
    """
    root = Path(tmp_out_dir)
    out: Dict[str, Any] = {"static": {}, "anim": {}, "combined": {}}

    static_dir = root / "per_key_static"
    anim_dir = root / "per_key_animation"
    combined_dir = root / "combined"

    if static_dir.is_dir():
        for p in sorted(static_dir.glob("*.png")):
            key = p.stem.replace("_3d", "")
            out["static"][key] = {"filename": p.name, "mime": "image/png", "b64": _b64(p.read_bytes())}

    if anim_dir.is_dir():
        for p in sorted(anim_dir.glob("*.gif")):
            key = p.stem.replace("_3d", "")
            out["anim"][key] = {"filename": p.name, "mime": "image/gif", "b64": _b64(p.read_bytes())}

    if combined_dir.is_dir():
        for p in sorted(combined_dir.glob("*.gif")):
            out["combined"][p.stem] = {"filename": p.name, "mime": "image/gif", "b64": _b64(p.read_bytes())}

    return out


def _deep_merge_sim_cfg(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge overlay into base; nested dicts merge, scalars/lists replaced by overlay."""
    if not isinstance(overlay, dict):
        raise TypeError(f"overlay must be a dict, got {type(overlay).__name__}")
    out: Dict[str, Any] = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge_sim_cfg(out[k], v)
        else:
            out[k] = v
    return out


def _parse_injection_cfg(injection_cfg: Optional[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    injection_cfg (single MCP payload field) supports:
      - {"json": { ... }}  — overlay dict
      - {"b64": "<base64>"} — UTF-8 JSON text after decode
      - {"text": "<json string>"}
      - { ... } — bare dict treated as overlay (no wrapper keys)
    """
    if not injection_cfg:
        return None, None
    if not isinstance(injection_cfg, dict):
        return None, "injection_cfg must be an object"
    try:
        if "json" in injection_cfg:
            inner = injection_cfg["json"]
            if isinstance(inner, dict):
                return inner, None
            return None, "injection_cfg.json must be an object"
        if "b64" in injection_cfg:
            raw = base64.b64decode(str(injection_cfg["b64"])).decode("utf-8")
            data = json.loads(raw)
            if not isinstance(data, dict):
                return None, "decoded injection b64 must be a JSON object"
            return data, None
        if "text" in injection_cfg:
            data = json.loads(str(injection_cfg["text"]))
            if not isinstance(data, dict):
                return None, "injection_cfg.text must parse to a JSON object"
            return data, None
        # gien: whole object is the overlay (no b64/text/json wrapper).
        return dict(injection_cfg), None
    except Exception as exc:
        print(f"Err main::_parse_injection_cfg | handler_line=123 | {type(exc).__name__}: {exc}")
        print(f"[exception] main._parse_injection_cfg: {exc}")
        return None, repr(exc)

def visualize(
    jax_guard: Any = None,
    output_dir: Optional[str] = None,
    amount_nodes_eff: int = 0,
    sim_time_eff: int = 0,
    dims_eff: int = 0,
) -> Optional[Dict[str, Any]]:
    """
    Optional color_master 3D pipeline. Call with `jax_guard` (and optional `output_dir`) from
    `run_main_process` after a successful JAX run. With `jax_guard is None` this is a no-op
    and returns None (e.g. when imported but not used from __main__).
    """
    if jax_guard is None:
        return None

    amount_nodes_eff = amount_nodes_eff or int(os.environ.get("AMOUNT_NODES", "0") or 0)
    sim_time_eff = sim_time_eff or int(os.environ.get("SIM_TIME", "0") or 0)
    dims_eff = dims_eff or int(os.environ.get("DIMS", "0") or 0)

    visualizations: Optional[Dict[str, Any]] = None
    series_used: Optional[Dict[str, List[Any]]] = None
    out_dir_used: Optional[str] = None
    visualization_error: Optional[str] = None

    try:
        _step("visualization.import.start")
        from color_master.main import _demo_input_data, QUALITY_LIGHT, build_3d_time_series_visualization
        from color_master.sim_bridge import collect_series_from_jax_guard, run_workflow_visualization
        _step("visualization.import.done")
    except Exception as exc:
        print(f"Err main::visualize | {type(exc).__name__}: {exc}")
        print(f"[exception] main.visualize: {exc}")
        visualization_error = repr(exc)
        _step("visualization.import.failed", error=visualization_error)

    if output_dir:
        if visualization_error is None:
            _step("visualization.output_dir_mode.start", output_dir=output_dir)
            out_dir_used = run_workflow_visualization(
                jax_guard,
                output_dir=output_dir,
                _demo_input_data=_demo_input_data,
                QUALITY_LIGHT=QUALITY_LIGHT,
                build_3d_time_series_visualization=build_3d_time_series_visualization,
            )
            _step("visualization.output_dir_mode.done", output_dir=out_dir_used)
    else:
        if visualization_error is None:
            _step("visualization.inline_mode.start")
            data: Dict[str, List[Any]] = {}
            if jax_guard is not None:
                _step("visualization.collect_series.start")
                data = collect_series_from_jax_guard(jax_guard)
                _step("visualization.collect_series.done", series_keys=sorted(data.keys()))
            if not data:
                t_steps = max(8, sim_time_eff or 24)
                _step("visualization.demo_fallback.start", timesteps=t_steps)
                data = _demo_input_data(timesteps=t_steps)
                _step("visualization.demo_fallback.done", series_keys=sorted(data.keys()))
            series_used = data

            with tempfile.TemporaryDirectory(prefix="color_master_") as td:
                _step("visualization.render.start", temp_dir=td)
                build_3d_time_series_visualization(
                    data=data,
                    amount_nodes=amount_nodes_eff or 4,
                    dims=max(dims_eff or 3, 3),
                    output_dir=td,
                    quality=QUALITY_LIGHT,
                    quality_preset="default",
                )
                visualizations = _slurp_visualizations(td)
                _step(
                    "visualization.render.done",
                    static=len((visualizations or {}).get("static", {})),
                    anim=len((visualizations or {}).get("anim", {})),
                    combined=len((visualizations or {}).get("combined", {})),
                )
    return visualizations

def run_main_process(
    amount_nodes: int,
    sim_time: int,
    dims: int,
    injection_cfg: dict[
        str,  # field
        list[tuple[tuple[int],  # pos
        list[list[int], list[int]]  # data
        ]]
    ] = None,
    output_dir: Optional[str] = None,
    user_id: Any = "public",
    injection_file: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    # MCP/legacy: `injection_file` is the same payload as `injection_cfg`
    if injection_cfg is None and injection_file is not None:
        injection_cfg = injection_file
    _step(
        "run_main_process.start",
        output_dir=output_dir,
        amount_nodes=amount_nodes,
        sim_time=sim_time,
        dims=dims,
        user_id=user_id,
        has_injection=bool(injection_cfg),
    )

    # Build G
    g = GUtils()
    qfu = QFUtils(G=g.G)
    injector = Injector(g, amount_nodes)

    # DEFAULT INJECTION PATTERN
    fields = ["ELECTRON", "PHOTON"]
    if injection_cfg is None:
        injection_cfg = injector.rainbow(
            sim_time,
            amount_nodes,
            fields,
            dims,
        )

    # BUILD SM
    _step("workflow.graph.initialize.start")
    SMManager()._initialize_graph(
        env_id="public",
        g=g,
        qf=qfu
    )

    # INCLUDE INJECTIONS
    injector.set_inj_pattern(
        inj_struct=injection_cfg
    )

    components = CG(amount_nodes, sim_time, dims, qfu, g, user_id=user_id, injector=injector).main()

    jax_guard = JaxGuard(cfg=components).main()

    _local_json = os.path.join(_REPO_ROOT, "local.json")
    result = {
        "ok": True,
        "jax_ok": jax_guard is not None,
        "user_id": user_id,
        "local_json": _local_json if os.path.isfile(_local_json) else None,
    }
    return result

if __name__ == "__main__":
    run_main_process(
        amount_nodes=3,
        sim_time=1,
        dims=3,
    )

