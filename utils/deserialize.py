import json


def deserialize(val):
    try:
        if isinstance(val, str):
            val = json.loads(val)
    except Exception as e:
        print(f"Err core.utils.deserialize::deserialize | handler_line=8 | {type(e).__name__}: {e}")
        print(f"[exception] core.utils.deserialize.deserialize: {e}")
        print(f"Err deserialize: {e}")
    print("deserialized", val, type(val))
    return val
