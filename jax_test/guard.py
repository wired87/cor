import base64
import json
import os
import pprint
from typing import Any, Dict, List

import jax.numpy as jnp

import jax

from jax_test.data_handler.main import load_data
from jax_test.gnn.gnn import GNN
from jax_test.jax_utils.deserialize_in import parse_value



def _to_json_serializable(data):
    """Convert JAX/numpy arrays and complex to JSON-serializable (list/dict)."""
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
    return data


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
    # todo prevaliate features to avoid double calculations
    def __init__(self, cfg):
        #JAX
        platform = "cpu" if os.name == "nt" else "gpu"
        jax.config.update("jax_platform_name", platform)  # must be run before jnp
        self.gpu = jax.devices(platform)[0]

        self.cfg = cfg
        print("cfg:")
        pprint.pp(cfg)
        AMOUNT_NODES = int(self.cfg.get("AMOUNT_NODES"))
        SIM_TIME = int(self.cfg.get("SIM_TIME"))
        DIMS = int(self.cfg.get("DIMS"))

        for k, v in self.cfg.items():
            self.cfg[k] = parse_value(v)

            if isinstance(self.cfg[k], dict):
                for i, o in self.cfg[k].items():
                    self.cfg[k][i] = parse_value(o)

        # layers
        self.gnn_layer = GNN(
            gpu=self.gpu,
            **self.cfg
        )

        self._live_ws = None

    def divide_vector(self, vec, divisor):
        """Divide all values of a given vector by divisor. Returns array same shape as vec."""
        v = jnp.asarray(vec)
        d = jnp.asarray(divisor)
        return v / d

    def _make_send_live(self, vis_fps):
        """Return a callable (gnn_self, step) -> None that sends LIVE_DATA via self._live_ws."""
        import numpy as np
        from grid.live_payload import build_live_data_payload

        ws = self._live_ws
        user_id = os.getenv("USER_ID")
        env_id = os.getenv("ENV_ID")
        if not ws or not user_id or not env_id:
            return None

        def send_live(gnn_self, step):
            if ws is None:
                return
            # Throttle: send roughly every N steps to cap frame rate
            counter = getattr(gnn_self, "_vis_step_counter", 0)
            gnn_self._vis_step_counter = counter + 1
            n = max(1, 100 // max(1, vis_fps))
            if counter % n != 0:
                return
            try:
                dl = gnn_self.db_layer
                flat = np.asarray(dl.time_construct[0]).ravel()
                if np.iscomplexobj(flat):
                    flat = np.abs(flat.astype(np.complex64)).astype(np.float32)
                cfg = {
                    k: getattr(gnn_self, k, None)
                    for k in ("DB_PARAM_CONTROLLER", "AMOUNT_PARAMS_PER_FIELD", "MODULES", "FIELDS", "DB_KEYS", "FIELD_KEYS")
                }
                data = build_live_data_payload(cfg, flat)
                payload = {
                    "type": "LIVE_DATA",
                    "auth": {"user_id": user_id, "env_id": env_id},
                    "data": data,
                }
                ws.send(json.dumps(payload, default=str))
            except Exception as e:
                print(f"Err cor.jax_test.guard::Guard._make_send_live.send_live | handler_line=129 | {type(e).__name__}: {e}")
                print(f"[exception] cor.jax_test.guard.Guard.send_live: {e}")
                print("[jax_test.Guard] LIVE_DATA send error:", e)

        return send_live

    def main(self):
        self.gnn_layer.main()
        #results = self.finish()
        print("SIMULATION PROCESS FINISHED")
        return self


    def run(self):

        print("run... done")



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

    def _export_ctlr(self):
        print("_export_ctlr...")
        dl = self.gnn_layer.db_layer
        env_id = os.getenv("ENV_ID")

        db_ctlr = {
            "id": env_id,
            "OUT_SHAPES": _to_json_serializable(dl.OUT_SHAPES),
            "SCALED_PARAMS": _to_json_serializable(dl.SCALED_PARAMS),
            "METHOD_TO_DB": _to_json_serializable(dl.METHOD_TO_DB),
            "AMOUNT_PARAMS_PER_FIELD": _to_json_serializable(dl.AMOUNT_PARAMS_PER_FIELD),
            "DB_PARAM_CONTROLLER": _to_json_serializable(dl.DB_PARAM_CONTROLLER),
            "DB_KEYS": _to_json_serializable(self.cfg["DB_KEYS"]),
            "FIELD_KEYS": _to_json_serializable(self.cfg["FIELD_KEYS"])
        }


        model_ctlr = {
            "id": env_id,
            "VARIATION_KEYS": _to_json_serializable(self.cfg["VARIATION_KEYS"]),
        }



        print("_export_ctlr... done (local-only: no BigQuery)")



    def _export_engine_state(self, serialized_in, serialized_out, out_path: str = "engine_output.json"):
        """Save all generated engine data (history, db, tdb, etc.) to a local .json file."""
        dl = self.gnn_layer.db_layer
        try:
            payload = {
                "serialized_out": _to_json_serializable(base64.b64encode(serialized_out).decode('ascii')),
                "serialized_in": _to_json_serializable(base64.b64encode(serialized_in).decode('ascii')),

                # CTLR
                "ENERGY_MAP": None,
            }
            if hasattr(dl, "tdb") and dl.tdb is not None:
                payload["tdb"] = _to_json_serializable(dl.tdb)

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, allow_nan=True)
            print("engine state saved to", out_path)
        except Exception as e:
            print(f"Err cor.jax_test.guard::Guard._export_engine_state | handler_line=211 | {type(e).__name__}: {e}")
            print(f"[exception] cor.jax_test.guard.Guard._export_engine_state: {e}")
            print("export_engine_state failed:", e)

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
        Combine DBLayer per-param histories into column payloads:
            { env_col_name: {\"values\": [...], \"features\": [...] }, ... }
        """
        values_hist: Dict[int, list] = getattr(dl, "param_values_history", {}) or {}
        features_hist: Dict[int, list] = getattr(dl, "param_features_history", {}) or {}
        if not values_hist and not features_hist:
            return {}

        keys = self._build_param_keys()
        all_indices = sorted(set(list(values_hist.keys()) + list(features_hist.keys())))
        out: Dict[str, Any] = {}
        for idx in all_indices:
            if idx < 0:
                continue
            key = keys[idx] if idx < len(keys) else f"p_{idx}"
            col_name = _sanitize_param_column_name(key)
            vals = values_hist.get(idx, []) or []
            feats = features_hist.get(idx, []) or []
            # Ensure JSON-serializable primitives.
            vals_f = [float(v) for v in vals]
            feats_f = [float(f) for f in feats]
            out[col_name] = {
                "values": vals_f,
                "features": feats_f,
            }
        return out

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
    Guard().main()
