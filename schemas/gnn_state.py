"""Runtime in-memory schemas for the GNN engine objects.

Prompt (2026-05-30): "Create a types dir which includes exact detailed types for all
objects inside of the project (analyze under condition of data processes and infer
based on that the correct schema)." — these `Protocol` types describe the *living*
objects that flow through the simulation (`DBLayer`, `FeatureEncoder`, `GNN`, `Node`)
inferred from their actual attribute usage in `jax_test/gnn/{db_layer,feature_encoder,gnn}.py`.
They are intentionally `Protocol`s, not subclasses, so the existing concrete classes
satisfy them structurally without inheritance.
"""
from __future__ import annotations

from typing import Any, List, Protocol, Tuple

from .aliases import (
    AnyArray,
    Coord4,
    EquationIndex,
    FloatArray,
    IntArray,
    TimeStep,
)


# --- DBLayer ---------------------------------------------------------------------------

class DBLayerState(Protocol):
    """Structural type of `jax_test.gnn.db_layer.DBLayer` after `build_db()`.

    Every attribute below is set in `__init__` or `build_db()`. JAX device placement
    happens after construction (`jax.device_put`); types remain `*Array` either way.
    """

    # raw config copies (mostly jnp.int64 arrays after `__init__`)
    DB:                                   FloatArray   # the unscaled-param flat buffer
    AXIS:                                 List[int]
    AMOUNT_PARAMS_PER_FIELD:              IntArray
    DB_PARAM_CONTROLLER:                  IntArray
    DB_SHAPE:                             List[List[int]]
    FIELDS:                               IntArray
    METHODS_PER_MOD_LEN_CTLR:             IntArray
    METHOD_TO_DB:                         IntArray     # (R,4) of (mod, field, param, node)
    DB_TO_METHOD_EDGES:                   IntArray     # (R,4)
    DB_CTL_VARIATION_LEN_PER_FIELD:       IntArray
    DB_CTL_VARIATION_LEN_PER_EQUATION_CUMSUM: int
    LEN_FEATURES_PER_EQ:                  List[List[int]]

    # cumulative-sum tables (set in `__init__` / `build_db`)
    FIELDS_CUMSUM:                        IntArray
    AMOUNT_PARAMS_PER_FIELD_CUMSUM:       IntArray
    DB_PARAM_CONTROLLER_CUMSUM:           IntArray
    SCALED_PARAMS:                        IntArray     # set after build_db
    SCALED_PARAMS_CUMSUM:                 IntArray
    SCALED_PARAMS_CUMSUM_UNPADDED:        IntArray
    UNSCALED_CUMSUM_PADDED:               IntArray

    # runtime state (mutated each timestep)
    nodes:           FloatArray                        # (P,) flat node grid
    tdb:             AnyArray                          # (T,...) time-database
    time_construct:  FloatArray                        # (2, P) — current + prev step
    history_nodes:   List[FloatArray]                  # appended snapshots
    out_idx_map:     IntArray                          # vmapped abs DB index map
    out_shapes_sum:  List[int]

    # legacy / accumulator stores (kept for backwards-compat — see GNN.serialie_input)
    store:           List[Any]                         # never populated in current code
    out_store:       List[List[Any]]                   # per-equation flat results
    out_f_store:     List[List[Any]]                   # per-equation feature embeddings
    in_features:     List[List[Any]]                   # per-method input embeddings

    # runtime knobs copied from cfg
    SIM_TIME:        int
    DIMS:            int
    gpu:             Any                               # `jax.Device`


# --- FeatureEncoder --------------------------------------------------------------------

class FeatureEncoderState(Protocol):
    """Structural type of `jax_test.gnn.feature_encoder.FeatureEncoder`."""

    rngs:               Any                            # `jax.random.PRNGKey`
    d_model:            int
    amount_variations:  int
    db_layer:           DBLayerState
    AXIS:               List[int]
    result_blur:        float

    # per-equation linear stacks (built lazily in `prep`)
    in_linears:         List[List[Any]]                # nested by [eq][variation] — `eqx.nn.Linear`
    out_linears:        List[Any]                      # appended top-level by `create_out_linear`

    # per-equation feature stores (timestep-indexed)
    in_store:           List[Any]
    out_store:          List[List[Any]]                # per-equation outputs
    in_f_store:         List[List[List[Any]]]          # nested [eq][var][step]
    out_f_store:        List[List[Any]]
    in_ts:              List[List[Any]]                # per-timestep input embeddings list

    # bookkeeping
    out_skeleton:       List[Any]
    in_skeleton:        List[Any]
    feature_controller: List[Any]


# --- Node (the dynamic equation runnable) ----------------------------------------------

class NodeProtocol(Protocol):
    """Structural type of `jax_test.mod.Node` — a JAX-callable equation runnable."""

    eq_idx: EquationIndex

    def __call__(
        self,
        *,
        unprocessed_in: List[FloatArray],
        precomputed_grid: Any,
        in_axes_def: Tuple[int, ...],
        eq_idx: EquationIndex,
    ) -> FloatArray: ...


# --- GNN -------------------------------------------------------------------------------

class GNNState(Protocol):
    """Structural type of `jax_test.gnn.gnn.GNN` — the orchestrator class.

    Holds the layered sub-systems plus the per-step capture buffers used by
    `serialie_input` (see `jax_test/gnn/gnn.py::calc_batch`).
    """

    gpu:                Any
    db_layer:           DBLayerState
    feature_encoder:    FeatureEncoderState
    schema_grid:        List[Tuple[int, int, int]]
    model_feature_dims: int

    # per-equation tables built in `prep`
    METHODS:            List[NodeProtocol]
    in_shapes_all_eqs:  List[List[List[Any]]]          # nested [eq][var][shape_entry]
    axs_all_eqs:        List[Any]
    FEATURES_CUMSUM:    IntArray

    # runtime knobs copied from cfg
    AMOUNT_NODES:       int
    SIM_TIME:           int
    DIMS:               int
    LEN_FEATURES_PER_EQ_CUMSUM: IntArray
    LEN_FEATURES_PER_EQ:        List[List[int]]

    # serialization buffers populated inside `calc_batch`
    _raw_outs_history:  List[List[FloatArray]]         # one entry per timestep
    _features_history:  List[List[FloatArray]]         # one entry per timestep

    # injection
    injection_pattern:  List[Any]                      # see schemas.injection.InjectionPattern

    def main(self) -> Tuple[bytes, bytes]: ...
    def calc_batch(self) -> None: ...
    def simulate(self) -> None: ...
    def prepare(self) -> None: ...
    def serialie_input(self) -> Tuple[bytes, bytes]: ...
