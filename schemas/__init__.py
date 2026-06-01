"""schemas — exact, inferred type definitions for every data object in the project.

Prompt (2026-05-30): "Create a types dir which includes exact detailed types for all
objects inside of the project (analyze under condition of data processes and infer
based on that the correct schema)."

Why this package is named `schemas/` and not `types/`:
    `main.py` does `sys.path.insert(0, _REPO_ROOT)`, so a top-level package literally
    named `types` would shadow the Python stdlib `types` module (used by `inspect`,
    `functools`, `dataclasses`, `pickle`, ...). `schemas` is the standard term for
    "type schemas / data shapes" and avoids that latent runtime bug — same intent.

Layout:
    schemas/
        aliases.py            — atomic type aliases (Coord4, Base64Str, FloatArray, …)
        sim_config.py         — SimConfig (sim_config.json + post-`parse_value` cfg)
        injection.py          — InjectionRequestCfg / InjectionPattern (two distinct shapes)
        controller.py         — DBController, ModelController, ControllerBundle
        output_artifacts.py   — EngineState, RuntimeConfig, Manifest, Visualization*, RunResult
        gnn_state.py          — DBLayerState, FeatureEncoderState, GNNState, NodeProtocol

Inference method:
    - JSON schemas (SimConfig, controller, manifest, runtime, engine_state) were
      inferred from the *actual* live files under `output/` and `sim_config.json`,
      then cross-checked with the writers (`JaxGuard._build_ctlr_for_export`,
      `_export_engine_state`, `_export_config_snapshot`, `_write_manifest`) and the
      readers (`main.py::visualize`, `_slurp_visualizations`).
    - In-memory schemas (DBLayerState, FeatureEncoderState, GNNState) were inferred
      from the actual `self.X = ...` attribute assignments in
      `jax_test/gnn/{db_layer,feature_encoder,gnn}.py`.
    - Injection schemas were taken from `Injector.set_inj_pattern`'s declared
      signature (CLI/MCP shape) and `GNN.inject`'s consumer code (engine shape).

Usage:
    from schemas import SimConfig, EngineState, DBController, GNNState, ...
"""
from __future__ import annotations

# --- atomic aliases -------------------------------------------------------------------
from .aliases import (
    AnyArray,
    Base64Str,
    BoolArray,
    Coord4,
    CoordXYZ,
    DimAxis,
    EnvId,
    EquationIndex,
    FieldIndex,
    FloatArray,
    IntArray,
    ModuleIndex,
    NodeIndex,
    ParamIndex,
    ParamKey,
    PathStr,
    TimeStep,
    VariationIdx,
)

# --- sim configuration ----------------------------------------------------------------
from .sim_config import SimConfig

# --- injection ------------------------------------------------------------------------
from .injection import (
    InjectionCfgWrapped,
    InjectionData,
    InjectionFieldEntry,
    InjectionPattern,
    InjectionPatternEntry,
    InjectionPos,
    InjectionRequestCfg,
    Schedule,
    ScheduleEvent,
)

# --- controller (output/ctlr/) --------------------------------------------------------
from .controller import ControllerBundle, DBController, ModelController

# --- output artifacts (output/results, output/manifest, output/config, output/viz) ----
from .output_artifacts import (
    EngineState,
    Manifest,
    RunResult,
    RuntimeConfig,
    VisualizationArtifact,
    VisualizationBundle,
)

# --- in-memory engine state (DBLayer / FeatureEncoder / GNN / Node) -------------------
from .gnn_state import (
    DBLayerState,
    FeatureEncoderState,
    GNNState,
    NodeProtocol,
)

__all__ = [
    # aliases
    "AnyArray", "Base64Str", "BoolArray", "Coord4", "CoordXYZ", "DimAxis",
    "EnvId", "EquationIndex", "FieldIndex", "FloatArray", "IntArray",
    "ModuleIndex", "NodeIndex", "ParamIndex", "ParamKey", "PathStr",
    "TimeStep", "VariationIdx",
    # sim_config
    "SimConfig",
    # injection
    "InjectionCfgWrapped", "InjectionData", "InjectionFieldEntry",
    "InjectionPattern", "InjectionPatternEntry", "InjectionPos",
    "InjectionRequestCfg", "Schedule", "ScheduleEvent",
    # controller
    "ControllerBundle", "DBController", "ModelController",
    # output artifacts
    "EngineState", "Manifest", "RunResult", "RuntimeConfig",
    "VisualizationArtifact", "VisualizationBundle",
    # in-memory state
    "DBLayerState", "FeatureEncoderState", "GNNState", "NodeProtocol",
]
