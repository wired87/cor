"""Injection-pattern schemas — the two distinct shapes used in the project.

Prompt (2026-05-30): "Create a types dir which includes exact detailed types for all
objects inside of the project (analyze under condition of data processes and infer
based on that the correct schema)." — `Injector.set_inj_pattern` (CLI / MCP entry)
accepts one shape; the engine rewrites it into a 5-element row consumed by
`GNN.inject(step, ...)`. Both shapes are typed here so callers don't conflate them.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, TypedDict, Union

from .aliases import (
    Coord4,
    EquationIndex,
    FieldIndex,
    ModuleIndex,
    NodeIndex,
    ParamIndex,
    TimeStep,
)

# CHAR: external (request) shape — a dict keyed by field id with positional injection
# entries. Source of truth: `Injector.set_inj_pattern`'s annotation:
#   dict[
#     str,                      # field id
#     list[
#       tuple[
#         tuple[int],            # pos
#         list[list[int], list[int]],  # data == [frequency, amplitude]
#       ]
#     ]
#   ]
InjectionPos = Tuple[int, ...]                                    # spatial position
InjectionData = List[List[int]]                                   # [frequency, amplitude]
InjectionFieldEntry = Tuple[InjectionPos, InjectionData]
InjectionRequestCfg = Dict[str, List[InjectionFieldEntry]]        # field_id -> entries


# CHAR: optional dict-wrapping shapes that `main.py::_parse_inj_cfg` accepts. The
# function unwraps one of three keys ("json", "b64", "text") to the actual dict above.
class InjectionCfgWrapped(TypedDict, total=False):
    json: InjectionRequestCfg          # already-parsed dict
    b64:  str                          # base64-encoded JSON
    text: str                          # raw JSON text


# CHAR: internal (engine) shape consumed by `GNN.inject` —
#   item = [mod_idx, field_idx, param_idx, node_idx, schedule]
#   schedule = [[time_point, value], ...]
ScheduleEvent = Tuple[TimeStep, Union[int, float]]      # (time, value) per scheduled spike
Schedule      = List[List[Union[int, float]]]          # legacy: list of 2-int rows

InjectionPatternEntry = List[Union[
    ModuleIndex,
    FieldIndex,
    ParamIndex,
    NodeIndex,
    Schedule,
]]
"""Five-element row used inside `inject()`:
[mod_idx, field_idx, param_idx, node_idx, schedule].
The fifth element is a `list[list[int|float]]` schedule of `[time, value]` pairs.
"""

InjectionPattern = List[InjectionPatternEntry]
"""Full per-step injection table consumed by `GNN.inject(step, db_layer)`."""
