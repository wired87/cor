import json, base64, binascii
import numpy as np
import jax.numpy as jnp

def parse_value(o):
    if isinstance(o, str):
        s = o.strip()

        # JSON?
        if s.startswith(("{", "[", '"', "true", "false", "null")):
            try:
                o = json.loads(s)
                return o
            except json.JSONDecodeError:
                print("Err cor.jax_test.jax_utils.deserialize_in::parse_value | handler_line=14 | json.JSONDecodeError handler triggered")
                print("[exception] cor.jax_test.jax_utils.deserialize_in.parse_value: caught json.JSONDecodeError")
                pass

        # Base64?
        try:
            raw = base64.b64decode(s, validate=True)
            arr = np.frombuffer(raw, dtype=np.complex64)
            return jnp.array(arr)
        except (binascii.Error, ValueError):
            print("Err cor.jax_test.jax_utils.deserialize_in::parse_value | handler_line=23 | (binascii.Error, ValueError) handler triggered")
            print("[exception] cor.jax_test.jax_utils.deserialize_in.parse_value: caught (binascii.Error, ValueError)")
            return o

    return o
