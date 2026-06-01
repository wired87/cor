"""Atomic type aliases shared by every schema module.

Prompt (2026-05-30): "Create a types dir which includes exact detailed types for all
objects inside of the project (analyze under condition of data processes and infer
based on that the correct schema)." — `aliases.py` holds the small, reusable atomic
names that every other schema imports. Heavy types (jax / numpy) stay behind
``TYPE_CHECKING`` so importing the package does not pay any runtime cost.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple, Union

# CHAR: keep stdlib-only at runtime — numpy / jax are imported lazily under
# TYPE_CHECKING so this module can be loaded by tooling (lsp, doc generators,
# CI lint) without pulling the whole numerical stack into memory.
if TYPE_CHECKING:
    import jax.numpy as _jnp
    import numpy as _np

    FloatArray = Union[_np.ndarray, _jnp.ndarray]
    IntArray   = Union[_np.ndarray, _jnp.ndarray]
    BoolArray  = Union[_np.ndarray, _jnp.ndarray]
    AnyArray   = Union[_np.ndarray, _jnp.ndarray]
else:
    # gien: at runtime these collapse to `Any` so no isinstance check ever rejects an
    # input that came as either a jax.Array or a np.ndarray (the engine uses both).
    FloatArray = Any
    IntArray   = Any
    BoolArray  = Any
    AnyArray   = Any

# --- identifiers --------------------------------------------------------------------
EnvId      = str            # `os.getenv("ENV_ID")` — used to scope env-keyed payloads
ParamKey   = str            # human-readable key, e.g. an entry of `DB_KEYS`
Base64Str  = str            # ASCII-only b64 payload (e.g. `serialized_raw_out`)
PathStr    = str            # file system path, always stored as `str` in payloads

# --- index spaces (all int but distinct semantics — encoded as aliases for docs) ----
ModuleIndex   = int   # 0..len(MODULES)-1
FieldIndex    = int   # 0..len(FIELDS)-1
ParamIndex    = int   # 0..AMOUNT_PARAMS_PER_FIELD[..]-1
NodeIndex     = int   # 0..AMOUNT_NODES-1
EquationIndex = int   # 0..len(METHODS)-1
VariationIdx  = int   # variation row inside an equation (V dimension)
TimeStep      = int   # 0..SIM_TIME-1
DimAxis       = int   # 0..DIMS-1, or -1 / None for "broadcast across this axis"

# --- coordinate tuples --------------------------------------------------------------
# CHAR: 4-int address `(module, field, param, node)` is the canonical address used by
# `METHOD_TO_DB`, `DB_TO_METHOD_EDGES`, `INJECTOR_INDICES`, and inside `inject()`.
Coord4 = Tuple[ModuleIndex, FieldIndex, ParamIndex, NodeIndex]
# 3-int spatial coord (x, y, z) used by `_coord_to_abs_idx` after stripping leading dims.
CoordXYZ = Tuple[int, int, int]
