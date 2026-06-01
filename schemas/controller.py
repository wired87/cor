"""Controller schemas — exact shape of `output/ctlr/{db,model}_ctlr.json`.

Prompt (2026-05-30): "Create a types dir which includes exact detailed types for all
objects inside of the project (analyze under condition of data processes and infer
based on that the correct schema)." — types here mirror `JaxGuard._build_ctlr_for_export`
1:1 so that any json.load() of those files type-checks against `DBController` /
`ModelController`. All ndarray-shaped fields are emitted as nested-list-of-int through
`_to_json_serializable`; we encode that as `list[int]` / `list[list[int]]` accordingly.
"""
from __future__ import annotations

from typing import List, Optional, TypedDict


class DBController(TypedDict):
    """Persistent metadata describing the DB layout for one run.

    Source: `JaxGuard._build_ctlr_for_export()["db"]`. Field order and shape match the
    arrays that `DBLayer` exposes after `build_db()` completes — every entry is JSON
    via `_to_json_serializable` (ndarray → nested list, complex → {real, imag}).
    """

    id: Optional[str]                            # `os.getenv("ENV_ID")` — may be None
    OUT_SHAPES: List[List[int]]                  # per-equation output shapes
    SCALED_PARAMS: List[int]                     # length P; scaled param sizes
    METHOD_TO_DB: List[List[int]]                # ragged 4-int rows
    AMOUNT_PARAMS_PER_FIELD: List[int]           # length U
    DB_PARAM_CONTROLLER: List[int]               # length P
    DB_KEYS: Optional[List[str]]                 # length P, may be absent in cfg
    FIELD_KEYS: Optional[List[str]]              # length F, may be absent in cfg
    MODULES: List[int]                           # length M
    FIELDS: List[int]                            # length F


class ModelController(TypedDict):
    """Variation / model controller metadata. Source: `_build_ctlr_for_export()["model"]`."""

    id: Optional[str]                            # `os.getenv("ENV_ID")` — may be None
    VARIATION_KEYS: Optional[List[str]]          # human-readable variation keys


class ControllerBundle(TypedDict):
    """Tuple-as-dict bundle returned by `_build_ctlr_for_export()`."""

    db: DBController
    model: ModelController
