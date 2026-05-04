"""Alias qbrain.qf_utils -> top-level qfu package (sources live under cor/qfu)."""
import importlib.util

_spec = importlib.util.find_spec("qfu")
if _spec is None or not _spec.submodule_search_locations:
    raise ImportError("qfu must be on PYTHONPATH before qbrain.qf_utils")
__path__ = list(_spec.submodule_search_locations)
