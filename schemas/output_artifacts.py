"""Output artifact schemas — every file `JaxGuard` + `color_master` write under `output/`.

Prompt (2026-05-30): "Create a types dir which includes exact detailed types for all
objects inside of the project (analyze under condition of data processes and infer
based on that the correct schema)." — covers:
  output/manifest.json                  → `Manifest`
  output/results/engine_state.json      → `EngineState`
  output/config/runtime.json            → `RuntimeConfig`
  visualization slurp dict (in-memory)  → `VisualizationBundle` + `VisualizationArtifact`
Schemas inferred from the writers in `jax_test/guard.py` and the slurper in
`main.py::_slurp_visualizations`. Consumer contract: `np.frombuffer(b64decode(x), dtype=np.float64)`.
"""
from __future__ import annotations

from typing import Dict, List, Optional, TypedDict

from .aliases import Base64Str, EnvId, PathStr
from .controller import ControllerBundle  # re-exported via __init__ for completeness


# --- output/results/engine_state.json --------------------------------------------------

class EngineState(TypedDict):
    """Body of `output/results/engine_state.json` written by `_export_engine_state`.

    Both fields are base64-encoded `float64` byte buffers — decode via
    `np.frombuffer(base64.b64decode(value), dtype=np.float64)`. Empty string is a
    valid (degenerate) state when the simulation produced no samples.
    """

    serialized_raw_out: Base64Str    # flattened per-step `all_outs` history
    serialized_f_out:   Base64Str    # flattened per-step `all_features` history


# --- output/config/runtime.json --------------------------------------------------------

class RuntimeConfig(TypedDict):
    """Body of `output/config/runtime.json` — process-level knobs of one run."""

    AMOUNT_NODES: int                # nodes per spatial dim — also in SimConfig
    SIM_TIME:     int                # number of timesteps    — also in SimConfig
    DIMS:         int                # spatial dimensions     — also in SimConfig
    ENV_ID:       Optional[EnvId]    # `os.getenv("ENV_ID")`, may be None
    platform:     str                # "cpu" on Windows, "gpu" elsewhere


# --- output/manifest.json --------------------------------------------------------------

class Manifest(TypedDict):
    """Body of `output/manifest.json` written by `JaxGuard._write_manifest`.

    `files` is a flat map of `<rel_path>` → byte size. Path separators are normalized
    to forward slashes for portability across OS.
    """

    results:        PathStr                 # rel path to results sub-folder ("results")
    ctlr:           PathStr                 # rel path to controller sub-folder ("ctlr")
    config:         PathStr                 # rel path to config sub-folder ("config")
    visualizations: PathStr                 # rel path to viz sub-folder ("visualizations")
    files:          Dict[PathStr, int]      # flat <rel_path> -> byte_size index


# --- visualization slurp (in-memory, not persisted as one file) ------------------------

class VisualizationArtifact(TypedDict):
    """One image / animation entry inside the viz bundle returned by `_slurp_visualizations`."""

    filename: str                    # bare filename (no directory part)
    mime:     str                    # "image/png" or "image/gif"
    b64:      Base64Str              # raw asset bytes, base64-encoded


class VisualizationBundle(TypedDict, total=False):
    """In-memory dict returned by `main.py::_slurp_visualizations`.

    All fields are sub-dicts keyed by stem (filename without extension and without the
    `_3d` suffix). `static` and `anim` are derived from `per_key_*`; `combined` and
    `indexed` mirror their respective sub-folders verbatim.
    """

    static:   Dict[str, VisualizationArtifact]
    anim:     Dict[str, VisualizationArtifact]
    combined: Dict[str, VisualizationArtifact]
    indexed:  Dict[str, VisualizationArtifact]


# --- top-level run-result struct (returned by `main.run_main_process`) -----------------

class RunResult(TypedDict, total=False):
    """Dict returned by `main.run_main_process` (post-simulation summary).

    `components` is the cfg dict that was actually consumed (matches `SimConfig` shape
    after `parse_value` normalization), so consumers can re-feed it back into JaxGuard
    for re-runs without going through Guard / SMManager again.
    """

    components:        Dict[str, object]               # the consumed cfg dict (≈ SimConfig)
    jax_finished:      bool
    visualization_dir: Optional[PathStr]
    visualizations:    Optional[VisualizationBundle]


__all__ = [
    "ControllerBundle",      # re-exported from .controller for ergonomic single-import
    "EngineState",
    "Manifest",
    "RunResult",
    "RuntimeConfig",
    "VisualizationArtifact",
    "VisualizationBundle",
]
