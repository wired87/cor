import base64
import json
import os
import shutil
import tempfile
from typing import Any, Dict, List

import jax.numpy as jnp

import jax

from jax_test.data_handler.main import load_data
from jax_test.gnn.gnn import GNN
from jax_test.jax_utils.deserialize_in import parse_value



def _to_json_serializable(data):
    """Convert JAX/numpy arrays and complex to JSON-serializable (list/dict)."""
    if isinstance(data, (str, int, float, bool, type(None))):
        return data
    if isinstance(data, list):
        return [_to_json_serializable(x) for x in data]
    if isinstance(data, tuple):
        return [_to_json_serializable(x) for x in data]
    if isinstance(data, dict):
        return {k: _to_json_serializable(v) for k, v in data.items()}
    if isinstance(data, bytes):
        # --- FIX: bytes (e.g. from base64.b64encode) -> decode to string for JSON ---
        # base64.b64encode returns ASCII-safe bytes, so decode as ASCII
        return data.decode('ascii')
    if hasattr(data, "shape") and hasattr(data, "tolist"):
        arr = jnp.asarray(data)
        if jnp.iscomplexobj(arr):
            return {"real": jnp.real(arr).tolist(), "imag": jnp.imag(arr).tolist()}
        return arr.tolist()
    if hasattr(data, "item"):
        v = data.item()
        if isinstance(v, (complex, jnp.complexfloating)):
            return {"real": float(jnp.real(v)), "imag": float(jnp.imag(v))}
        return float(v) if hasattr(v, "real") else v
    return f"<{type(data).__name__} object>"


def _sanitize_param_column_name(raw_id: str) -> str:
    """
    Map a param id to a safe column name for the envs table.
    Mirrors logic from ParamsManager._sanitize_param_column_name.
    """
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(raw_id))
    if not safe:
        safe = "param"
    if safe[0].isdigit():
        safe = f"p_{safe}"
    return f"p_{safe}"


class JaxGuard:
    """JAX simulation orchestrator.

    Prompt (2026-05-30): "create a output dir which includes sim output files (controller (ctlr)
    and results) separated, sim config files and all other products of the sim." — `JaxGuard`
    now lays out a single `output/` root with disjoint sub-folders (`results/`, `ctlr/`,
    `config/`, `visualizations/`) plus a top-level `manifest.json`. Each `main()` call
    re-exports into the same layout in-place; concerns stay separated for downstream consumers.
    """
    # todo prevaliate features to avoid double calculations
    def __init__(self, cfg):
        #JAX
        platform = "cpu" if os.name == "nt" else "gpu"
        jax.config.update("jax_platform_name", platform)  # must be run before jnp
        self.gpu = jax.devices(platform)[0]

        self.cfg = cfg
        print("cfg:")

        # CHAR: structured output layout — `output/{results,ctlr,config,visualizations}/`.
        # `self.save_path` is kept for backwards-compat with callers that referenced the old
        # flat `output/results.json` path; it now points to the engine-state file inside
        # `results/`. Sub-folders are created up-front so every export branch can write blindly.
        self._init_output_layout()

        for k, v in self.cfg.items():
            self.cfg[k] = parse_value(v)

        # layers
        self.gnn_layer = GNN(
            gpu=self.gpu,
            **self.cfg
        )

    def _init_output_layout(self) -> None:
        """Create flat `output/` root — all engine JSON artifacts live at top level."""
        # gien: prompt — flat output folder; anchor at repo root so layout is CWD-independent
        _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.repo_root = _repo_root
        self.out_root = os.path.join(_repo_root, "output")
        # CHAR: single flat dir for results, ctlr, config, and viz file names (no nested sub-trees)
        self.dir_results = self.out_root
        self.dir_ctlr = self.out_root
        self.dir_config = self.out_root
        self.dir_viz = self.out_root
        os.makedirs(self.out_root, exist_ok=True)
        self.save_path = os.path.join(self.out_root, "engine_state.json")

    def divide_vector(self, vec, divisor):
        """Divide all values of a given vector by divisor. Returns array same shape as vec."""
        v = jnp.asarray(vec)
        d = jnp.asarray(divisor)
        return v / d


    def main(self):
        serialized_raw_out, serialized_f_out = self.gnn_layer.main()
        # CHAR: separated exports — config snapshot first (so it reflects what was consumed),
        # then controller metadata, then engine state, then a top-level manifest pointing at
        # everything that was produced. Each branch is wrapped in its own try/except inside the
        # method so a single failing export cannot abort the others.
        self._export_config_snapshot()
        self._export_ctlr()
        self._export_engine_state(
            serialized_raw_out, serialized_f_out
        )
        self._write_manifest()
        print("SIMULATION PROCESS FINISHED")
        return self


    def _export_data(self):
        print("_export_data...")
        dl = self.gnn_layer.db_layer

        history = []
        env_id = os.getenv("ENV_ID")

        for i, item in enumerate(dl.history_nodes):
            history.append(
                {
                    "id": f"{env_id}_{i}",
                    "data":_to_json_serializable(item),
                    "env_id": env_id
                }
            )

        print("_export_data... done (local-only: no BigQuery)")

    def _build_ctlr_for_export(self) -> Dict[str, Any]:
        """DB + model controller metadata for local.json (same shape as _export_ctlr)."""
        dl = self.gnn_layer.db_layer
        env_id = os.getenv("ENV_ID")
        db_ctlr = {
            "id": env_id,
            "OUT_SHAPES": _to_json_serializable(dl.OUT_SHAPES),
            "SCALED_PARAMS": _to_json_serializable(dl.SCALED_PARAMS),
            "METHOD_TO_DB": _to_json_serializable(dl.METHOD_TO_DB),
            "AMOUNT_PARAMS_PER_FIELD": _to_json_serializable(dl.AMOUNT_PARAMS_PER_FIELD),
            "DB_PARAM_CONTROLLER": _to_json_serializable(dl.DB_PARAM_CONTROLLER),
            "DB_KEYS": _to_json_serializable(self.cfg.get("DB_KEYS")),
            "FIELD_KEYS": _to_json_serializable(self.cfg.get("FIELD_KEYS")),
            "MODULES": _to_json_serializable(self.cfg.get("MODULES")),
            "FIELDS": _to_json_serializable(self.cfg.get("FIELDS")),
        }
        model_ctlr = {
            "id": env_id,
            "VARIATION_KEYS": _to_json_serializable(self.cfg.get("VARIATION_KEYS")),
        }
        return {"db": db_ctlr, "model": model_ctlr}

    def _export_ctlr(self):
        # CHAR: persist the DB + model controller bundles as two JSON files under `output/ctlr/`
        # — keeps controller metadata cleanly separated from raw simulation results, so downstream
        # consumers (visualizers, BQ uploaders) can load only what they need without touching the
        # multi-MB engine_state.json.
        print("_export_ctlr...")
        try:
            bundle = self._build_ctlr_for_export()
            with open(os.path.join(self.dir_ctlr, "db_ctlr.json"), "w", encoding="utf-8") as f:
                json.dump(bundle["db"], f, indent=2)
            with open(os.path.join(self.dir_ctlr, "model_ctlr.json"), "w", encoding="utf-8") as f:
                json.dump(bundle["model"], f, indent=2)
            print("_export_ctlr... done →", self.dir_ctlr)
        except Exception as e:
            print(f"Warn _export_ctlr: {type(e).__name__}: {e}")

    def _export_config_snapshot(self):
        # CHAR: snapshot config artifacts for a self-contained `output/` run.
        # runtime.json is written by root `guard.py::Guard._export_runtime` from class args — not here.
        print("_export_config_snapshot...")
        try:
            src_cfg = os.path.join(self.repo_root, "sim_config.json")
            if os.path.isfile(src_cfg):
                shutil.copyfile(src_cfg, os.path.join(self.dir_config, "sim_config.json"))

            with open(os.path.join(self.dir_config, "components.json"), "w", encoding="utf-8") as f:
                json.dump(_to_json_serializable(self.cfg), f, indent=2)

            print("_export_config_snapshot... done →", self.dir_config)
        except Exception as e:
            print(f"Warn _export_config_snapshot: {type(e).__name__}: {e}")

    def _write_manifest(self):
        # CHAR: tiny pointer-file that lists every product the sim wrote in this run. Walks
        # `output/` once, records relative path → byte size; cheap and gives downstream tools a
        # single index to discover artifacts without filesystem probing.
        try:
            files: Dict[str, int] = {}
            for _root, _, _names in os.walk(self.out_root):
                for fn in _names:
                    if fn == "manifest.json":
                        continue
                    full = os.path.join(_root, fn)
                    try:
                        size = os.path.getsize(full)
                    except OSError:
                        size = -1
                    rel = os.path.relpath(full, self.out_root).replace("\\", "/")
                    files[rel] = size
            # gien: flat layout — manifest is a single files map (rel_path -> byte_size)
            manifest = {
                "layout": "flat",
                "root":   ".",
                "files":  files,
            }
            with open(os.path.join(self.out_root, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            print("manifest written →", os.path.join(self.out_root, "manifest.json"))
        except Exception as e:
            print(f"Warn _write_manifest: {type(e).__name__}: {e}")



    def _export_engine_state(self, serialized_raw_out, serialized_f_out):
        """Save all generated engine data (history, db, tdb, ctlr, param_series, etc.) to a local .json file."""
        #
        dl = self.gnn_layer.db_layer

        try:
            payload = {
                "serialized_raw_out": _to_json_serializable(base64.b64encode(serialized_raw_out).decode('ascii')),
                "serialized_f_out": _to_json_serializable(base64.b64encode(serialized_f_out).decode('ascii')),
            }

            with open(self.save_path, "w") as f:
                f.write(json.dumps(payload))

            print("engine state saved to", self.save_path)
        except Exception as e:
            print(f"Warn _export_engine_state: {type(e).__name__}: {e}")

        # Local-only: param series stay in engine JSON / memory, not envs table.

    def _build_param_keys(self) -> List[str]:
        """
        Build a flat list of param keys in the same order as controller metadata.
        Mirrors jax_test.grid.live_payload._flat_keys_from_cfg.
        """
        gnn = self.gnn_layer
        param_ctrl = getattr(gnn, "DB_PARAM_CONTROLLER", []) or []
        amount_per_field = getattr(gnn, "AMOUNT_PARAMS_PER_FIELD", []) or []
        modules = getattr(gnn, "MODULES", None) or [0]
        fields = getattr(gnn, "FIELDS", None) or [1]
        db_keys = getattr(gnn, "DB_KEYS", None)
        field_keys = getattr(gnn, "FIELD_KEYS", None)

        try:
            param_ctrl = list(param_ctrl)
        except TypeError:
            print("Err cor.jax_test.guard::Guard._build_param_keys | handler_line=232 | TypeError handler triggered")
            print("[exception] cor.jax_test.guard.Guard._build_param_keys: caught TypeError")
            param_ctrl = [param_ctrl]
        try:
            amount_per_field = list(amount_per_field)
        except TypeError:
            print("Err cor.jax_test.guard::Guard._build_param_keys | handler_line=237 | TypeError handler triggered")
            print("[exception] cor.jax_test.guard.Guard._build_param_keys: caught TypeError")
            amount_per_field = [amount_per_field]
        try:
            modules = list(modules)
        except TypeError:
            print("Err cor.jax_test.guard::Guard._build_param_keys | handler_line=242 | TypeError handler triggered")
            print("[exception] cor.jax_test.guard.Guard._build_param_keys: caught TypeError")
            modules = [modules]
        try:
            fields = list(fields)
        except TypeError:
            print("Err cor.jax_test.guard::Guard._build_param_keys | handler_line=247 | TypeError handler triggered")
            print("[exception] cor.jax_test.guard.Guard._build_param_keys: caught TypeError")
            fields = [fields]

        n_modules = max(1, max(modules) + 1) if modules else 1
        n_fields = max(1, max(fields)) if fields else 1

        keys: List[str] = []
        idx = 0
        for _mi in range(n_modules):
            for _fi in range(n_fields):
                flat_idx = _mi * n_fields + _fi
                n_params = amount_per_field[flat_idx] if flat_idx < len(amount_per_field) else 1
                for _pi in range(n_params):
                    if db_keys and idx < len(db_keys):
                        keys.append(str(db_keys[idx]))
                    elif field_keys and idx < len(field_keys):
                        keys.append(str(field_keys[idx]))
                    else:
                        keys.append(f"p_{idx}")
                    idx += 1
        return keys

    def _build_param_series_payload(self, dl) -> Dict[str, Any]:
        """
        Combine DBLayer per-param histories. Shape:
        { "order", "param_indices", "series" } with series[col] = {values, features} in param index order.
        """
        # todo paste parameter history

    def _persist_param_series_to_env(self, dl) -> None:
        """No-op: local pipeline does not write param series to a remote DB."""
        return

    def finish(self):
        # Collect data
        history_nodes = self.gnn_layer.db_layer.history_nodes
        model_skeleton = getattr(self.gnn_layer, "model_skeleton", None)
        if model_skeleton is None:
            model_skeleton = getattr(self.gnn_layer.db_layer, "model_skeleton", None)
        if model_skeleton is None:
            # Some local runs do not materialize a model skeleton. Keep `finish()`
            # usable by returning an empty model payload instead of failing late.
            model_skeleton = []

        # Serialization helper
        def serialize(data):
            if isinstance(data, list):
                return [serialize(x) for x in data]
            if isinstance(data, tuple):
                return tuple(serialize(x) for x in data)
            if isinstance(data, dict):
                 return {k: serialize(v) for k, v in data.items()}

            # Check for JAX/Numpy array
            if hasattr(data, 'dtype') and hasattr(data, 'real') and hasattr(data, 'imag'):
                # Check directly if complex dtype
                if jnp.iscomplexobj(data):
                    return (data.real, data.imag)
            return data

        serialized_history = serialize(history_nodes)
        serialized_model = serialize(model_skeleton)

        try:
            model_bytes = self.gnn_layer.serialize(model_skeleton)
        except Exception:
            print("Err cor.jax_test.guard::Guard.finish | handler_line=335 | Exception handler triggered")
            print("[exception] cor.jax_test.guard.Guard.finish: caught Exception")
            model_bytes = b""

        try:
            history_bytes = self.gnn_layer.serialize(history_nodes)
        except Exception:
            print("Err cor.jax_test.guard::Guard.finish | handler_line=341 | Exception handler triggered")
            print("[exception] cor.jax_test.guard.Guard.finish: caught Exception")
            history_bytes = b""
        ctrl_payload = {
            "cfg": self.cfg,
            "AMOUNT_NODES": int(os.getenv("AMOUNT_NODES", "0") or 0),
            "SIM_TIME": int(os.getenv("SIM_TIME", "0") or 0),
            "DIMS": int(os.getenv("DIMS", "0") or 0),
        }
        try:
            ctrl_bytes = self.gnn_layer.serialize(ctrl_payload)
        except Exception:
            print("Err cor.jax_test.guard::Guard.finish | handler_line=352 | Exception handler triggered")
            print("[exception] cor.jax_test.guard.Guard.finish: caught Exception")
            ctrl_bytes = b""

        row = {
            "model": model_bytes,
            "time": history_bytes,
            "ctlr": ctrl_bytes,
        }
        # todo save
        print("DATA DISTRIBUTED")
        return row

if __name__ == "__main__":
    print("Run via main.py (repo root) with graph + components; this entry is a stub.")
