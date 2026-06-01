"""SimConfig — exact schema of `sim_config.json` (and the cfg dict consumed by `JaxGuard`).

Prompt (2026-05-30): "Create a types dir which includes exact detailed types for all
objects inside of the project (analyze under condition of data processes and infer
based on that the correct schema)." — every field below was inferred from the live
`sim_config.json` (24 top-level keys, 545 DB params, 37 equations, 4 fields, etc.) and
cross-checked against the consumers in `jax_test/gnn/db_layer.py`, `gnn.py` and
`jax_test/guard.py::_build_ctlr_for_export`.
"""
from __future__ import annotations

from typing import List, TypedDict

# Python 3.11 has `NotRequired` directly on `typing`; older runtimes would need
# `typing_extensions`. The project pins Python 3.11, so we use the stdlib import.
from typing import NotRequired


class SimConfig(TypedDict):
    """Top-level schema of `sim_config.json` and the cfg dict consumed by `JaxGuard`.

    Shape annotations (where applicable) describe length relationships — they are
    contracts between fields, not size limits. Symbols:
      P  = total scaled-param count    (== len(DB_PARAM_CONTROLLER))
      U  = total unscaled-param count  (== sum(AMOUNT_PARAMS_PER_FIELD))
      M  = number of modules           (== len(MODULES))
      F  = number of fields            (== len(FIELDS))
      E  = number of equations         (== len(METHODS))
      A  = total axis-row count        (== len(AXIS))
      D  = number of spatial dims      (== DIMS)
      N  = nodes per dim               (== AMOUNT_NODES)
      T  = simulation timesteps        (== SIM_TIME)
      I  = injection-event count       (== len(INJECTOR_TIME))
    """

    # --- runtime knobs ---------------------------------------------------------------
    AMOUNT_NODES: int                          # N — nodes per spatial dimension
    SIM_TIME: int                              # T — number of timesteps to run
    DIMS: int                                  # D — number of spatial dimensions

    # --- raw DB payload (numpy float buffer, base64-encoded) -------------------------
    DB: str                                    # b64-encoded float buffer (`np.frombuffer`)

    # --- shape / controller arrays (length P unless noted) ---------------------------
    AXIS: List[int]                            # length A; per-axis-row int (-1/0/1) — see DimAxis
    DB_SHAPE: List[List[int]]                  # length P, each entry list[int] (rank-1 shape)
    DB_KEYS: List[str]                         # length P; human-readable param keys
    DB_PARAM_CONTROLLER: List[int]             # length P; per-param controller flag

    # --- module / field / equation tables --------------------------------------------
    MODULES: List[int]                         # length M; per-module int code
    FIELDS: List[int]                          # length F; per-field int code
    AMOUNT_PARAMS_PER_FIELD: List[int]         # length U; params-per-(module,field) flat
    E_KEY_MAP_PER_FIELD: List[int]             # length U; equation-key map per field-slot

    # --- equation set ----------------------------------------------------------------
    METHODS: List[str]                         # length E; equation source code (Python text)
    METHODS_PER_MOD_LEN_CTLR: List[int]        # length M; equations-per-module count
    METHOD_PARAM_LEN_CTLR: List[int]           # length E; param count per equation
    LEN_FEATURES_PER_EQ: List[List[int]]       # length E, inner list of feature group lengths

    # --- variation tables ------------------------------------------------------------
    DB_CTL_VARIATION_LEN_PER_EQUATION: List[int]   # length E; total variations per equation
    DB_CTL_VARIATION_LEN_PER_FIELD: List[int]      # length U; total variations per field-slot

    # --- graph edges (each entry is a 4-int (module, field, param, node) address) ----
    DB_TO_METHOD_EDGES: List[List[int]]        # ragged list of [4]-int rows
    METHOD_TO_DB: List[List[int]]              # ragged list of [4]-int rows

    # --- injection schedule (parallel arrays of length I) ----------------------------
    INJECTOR_TIME: List[int]                   # length I; absolute timestep
    INJECTOR_INDICES: List[List[List[int]]]    # length I; each entry list of [4]-int addrs
    INJECTOR_VALUES: List[List[int]]           # length I; each entry list of values

    # --- optional metadata only present in `components.json` (post-parse cfg) --------
    FIELD_KEYS: NotRequired[List[str]]         # human-readable field keys
    VARIATION_KEYS: NotRequired[List[str]]     # human-readable variation keys
