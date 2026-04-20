"""Flatten nested node attrs for graph export (minimal local implementation)."""


def flatten_attributes(attrs):
    if attrs is None:
        return {}
    if not isinstance(attrs, dict):
        return {}
    flat = {}
    for k, v in attrs.items():
        if isinstance(v, dict):
            for nk, nv in v.items():
                flat[f"{k}.{nk}"] = nv
        else:
            flat[k] = v
    return flat
