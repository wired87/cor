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

def visualize() -> Optional[Dict[str, Any]]:
    sim_results = json.loads(
        open("local.json", "r").read()
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

    visualizations = _slurp_visualizations()

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
) -> Dict[str, Any]:

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
    injector.set_inj_pattern(
        inj_struct=inj_cfg
    )

    components = CG(amount_nodes, sim_time, dims, qfu, g, user_id=user_id, injector=injector).main()

    jax_guard = JaxGuard(cfg=components).main()

    _step("visualization.run.start")
    result["visualizations"] = visualize()
    return result

if __name__ == "__main__":
    # das system empfängt input img

    inj_cfgs:list[dict] = convert_img_to_energy_map()

    for item in inj_cfgs:
        run_main_process(
            amount_nodes=1,
            sim_time=1,
            dims=1,
            inj_cfg=item
        )

"""


from color_master.main import (
_demo_input_data,
QUALITY_LIGHT,
build_3d_time_series_visualization,
build_indexed_viz_from_engine_dict,
)
from color_master.sim_bridge import (
collect_engine_payload_from_jax_guard,
collect_series_from_jax_guard,
run_workflow_visualization,
)

"""