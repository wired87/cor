"""Cor2 primary workflow: Firegraph + SM + Injector → Guard components → Jax simulation → optional color_master.

Prompt (2026-05): Understand `main.py` entry, run and capture console under `test_out/`, analyze failures,
apply clean minimal fixes so the engine exits 0; preserve existing comments; update README progress.
"""

import os
import pprint
import sys
import json
import base64

from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np

from firegraph.graph import GUtils
from guard import Guard as CG
from in_parser import convert_img_to_energy_map
from injector import Injector

from qfu.qf_utils import QFUtils
from sm_manager.sm_manager import SMManager
from jax_test.guard import JaxGuard
# gien: single import surface for post-simulation visualization (avoids duplicate trailing import block)
from color_master.sim_bridge import run_workflow_visualization

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


def _configure_stdio_utf8() -> None:
    # gien: Windows terminals often use cp1252; huge `components` dicts include μ etc. and crash prints
    _streams = (getattr(sys, "stdout", None), getattr(sys, "stderr", None))
    for _s in _streams:
        if _s is not None and hasattr(_s, "reconfigure"):
            try:
                _s.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


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

    idx_dir = root / "indexed"
    if idx_dir.is_dir():
        out.setdefault("indexed", {})
        for p in sorted(idx_dir.glob("*.gif")):
            out["indexed"][p.stem] = {
                "filename": p.name,
                "mime": "image/gif",
                "b64": _b64(p.read_bytes()),
            }
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


def run_color_master_from_config(
    sim_cfg_path: str,
) -> str:
    """
    Path-based color_master: load `sim_cfg.json` and enriched `local.json` (param_series + ctlr).
    Does not run JAX. Returns output subdirectory path (indexed GIF).
    """
    from color_master.main import run_path_based_viz

    return run_path_based_viz(sim_cfg_path)


def _parse_inj_cfg(inj_cfg: Optional[Dict[str, Any]]):
    if not inj_cfg:
        return None, None
    if not isinstance(inj_cfg, dict):
        return None, "inj_cfg must be an object"
    try:
        if "json" in inj_cfg:
            inner = inj_cfg["json"]
            if isinstance(inner, dict):
                return inner, None
            return None, "inj_cfg.json must be an object"
        if "b64" in inj_cfg:
            raw = base64.b64decode(str(inj_cfg["b64"])).decode("utf-8")
            data = json.loads(raw)
            if not isinstance(data, dict):
                return None, "decoded injection b64 must be a JSON object"
            return data, None
        if "text" in inj_cfg:
            data = json.loads(str(inj_cfg["text"]))
            if not isinstance(data, dict):
                return None, "inj_cfg.text must parse to a JSON object"
            return data, None
        return dict(inj_cfg), None
    except Exception as exc:
        print(f"Err main::_parse_inj_cfg | handler_line=123 | {type(exc).__name__}: {exc}")
        print(f"[exception] main._parse_inj_cfg: {exc}")
        return None, repr(exc)

def visualize(tmp_out_dir: str) -> Optional[Dict[str, Any]]:
    # gien: read latest engine export beside this repo (not process CWD) so IDE/sandbox runs stay stable
    local_path = os.path.join(_REPO_ROOT, "local.json")
    sim_results = json.loads(
        open(local_path, "r", encoding="utf-8").read()
    )
    print("SIM RESULTS, CFG LOAD...")
    serialized_f_out = np.frombuffer(
        base64.b64decode(sim_results["serialized_f_out"]),
        dtype=np.float64
    )

    serialized_raw_out = np.frombuffer(
        base64.b64decode(sim_results["serialized_raw_out"]),
        dtype=np.float64
    )
    pprint.pp(serialized_f_out)
    pprint.pp(serialized_raw_out)
    print("SIM RESULTS, CFG DESERIALIZED...")

    # gien: slurp requires the directory produced by color_master (PNGs/GIFs)
    visualizations = _slurp_visualizations(tmp_out_dir)

    return visualizations

def _run_visualization_enabled(explicit: Optional[bool]) -> bool:
    """Default on unless COLOR_MASTER_VIZ=0/false/off; explicit bool overrides."""
    if explicit is not None:
        return bool(explicit)
    raw = (os.environ.get("COLOR_MASTER_VIZ") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def run_main_process(
    amount_nodes: int,
    sim_time: int,
    dims: int,
    inj_cfg: dict[
        str,  # field
        list[tuple[tuple[int],  # pos
        list[list[int], list[int]]  # data
        ]]
    ] = None,
    output_dir: Optional[str] = None,
    user_id: Any = "public",
    run_visualization: Optional[bool] = None,
    visualization_dir: Optional[str] = None,
) -> Dict[str, Any]:

    _configure_stdio_utf8()
    # gien: MCP `run` route uses `injection_cfg`; CLI path uses `inj_cfg` — merge without mutating callers
    _step(
        "run_main_process.start",
        output_dir=output_dir,
        amount_nodes=amount_nodes,
        sim_time=sim_time,
        dims=dims,
        user_id=user_id,
        has_injection=bool(inj_cfg),
    )

    # Build G
    g = GUtils()
    qfu = QFUtils(G=g.G, dims=dims)
    injector = Injector(g, amount_nodes)

    # BUILD SM
    _step("workflow.graph.initialize.start")
    SMManager()._initialize_graph(
        env_id="public",
        g=g,
        qf=qfu
    )

    # INCLUDE INJECTIONS
    if inj_cfg is not None:
        injector.set_inj_pattern(
            inj_struct=inj_cfg
        )

    components = CG(amount_nodes, sim_time, dims, qfu, g, user_id=user_id, injector=injector).main()

    jax_guard = JaxGuard(cfg=components).main()


    result: Dict[str, Any] = {"components": components, "jax_finished": True}
    if _run_visualization_enabled(run_visualization):
        os.environ["SIM_TIME"] = str(max(1, int(sim_time)))

        viz_root = visualization_dir or output_dir or os.path.join(_REPO_ROOT, "color_master_output")
        viz_dir = run_workflow_visualization(
            viz_root,
            jax_guard=jax_guard,
            amount_nodes=amount_nodes,
            dims=dims,
            quality_preset="light",
        )
        result["visualization_dir"] = viz_dir
        # gien: optional heavy payload (base64) — still useful for integrated clients
        result["visualizations"] = _slurp_visualizations(viz_dir)
    else:
        _step("visualization.skipped", reason="COLOR_MASTER_VIZ disabled")
        result["visualization_dir"] = None
        result["visualizations"] = None
    return result


if __name__ == "__main__":
    inj_cfg, max_len_inj = convert_img_to_energy_map()

    run_main_process(
        amount_nodes=3,
        sim_time=max_len_inj + 2,
        dims=3,
        inj_cfg=inj_cfg
    )