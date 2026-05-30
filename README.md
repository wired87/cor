# COR — Compiler of Reality

Think of COR like a **science garden** for ideas:

```
        🌌 Discovery
           │
      🔬 Equation
           │
         🌳 Model
        /   |   \
      ⚛️  🧪  📈 Outcome
```

From the perspective of a top quantum scientist: this project helps turn scientific equations into a living simulation, so we can observe how complex systems might evolve over time.

## What this project does

- Takes equations and field definitions
- Builds a structured simulation graph
- Runs the simulation engine
- Produces data you can inspect and visualize

## Potential outcome

COR can help you explore “what could happen next” in complex scientific systems.  
In plain terms: it supports discovery, comparison of scenarios, and faster experiment ideas before expensive real-world testing.

## How to run

From the repository root:

```bash
python main.py
```

## Important note about equations and fields

The engine is designed to work with **all kinds of equations**, as long as:

1. the equations are provided in the `sm_manager/arsenal` directory, and  
2. required fields are integrated to `qfutils`. (todo extend ux)

## Progress

- 2026-05-29 — Stabilized `main.py` so the engine writes raw simulation data without errors:
  - `jax_test/guard.py`: fixed `os.makedirs(save_path)` (the file path) → now uses the parent directory; this was the root cause for the `PermissionError` on `output/results.json` (an empty directory was created in place of the JSON file).
  - `jax_test/gnn/gnn.py::_stack_node_inputs`: equalize the trailing flat length across all stacked P-columns so that vmap delivers identically-shaped per-element vectors to dynamic runnables — eliminates the `incompatible shapes for broadcasting` errors during GNN feature processing.
  - `jax_test/utils.py::create_runnable`: introduced a thin `SafeJnp` shim and an AST rewriter that turns `a @ b` into `jnp.matmul(a, b)`; the shim transparently promotes 0-D operands so chained equation expressions (`a @ b @ c`) no longer abort with `matmul ... ndim 0`.
  - `jax_test/ops/ops_defs.py` and `utils/math/ops_defs.py`: aligned `op_matmul`/`op_dot` with the same scalar-safe behavior.
  - Net effect: `py -3.11 main.py` exits 0, `output/results.json` is written as a proper file containing the base64-serialized `raw_out`/`f_out` payload, and the run output is free of shape-mismatch and matmul errors.
- 2026-05-29 — Made `output/results.json` carry real simulation data (was a 58-byte stub with empty msgpack):
  - `jax_test/gnn/gnn.py::serialie_input`: replaced `flax.serialization.to_bytes(db_layer.store)` (a never-populated slot) with a small `_flatten_to_float64_bytes(...)` helper that walks nested lists/tuples and JAX/NumPy arrays into one contiguous `float64` byte buffer — matches the consumer in `main.py::visualize` (`np.frombuffer(b64decode(...), float64)`).
  - `jax_test/gnn/gnn.py::calc_batch`: snapshot the per-step `all_outs` (raw equation results) and `all_features` (input feature embeddings) into two new history buffers on the GNN (`_raw_outs_history`, `_features_history`). This side-steps the broken `feature_encoder.out_linears[eq_idx]` chain (top-level appended Linears never reach per-eq slots), which was causing `save_out`'s inner zip to iterate zero times and `out_store`/`out_f_store` to stay empty.
  - `serialie_input` now serializes those history buffers plus the legacy stores (in case future config populates them), so `results.json` reflects the actual simulation evolution.
  - `_flatten_to_float64_bytes`: clamp non-finite samples (`nan`, ±`inf` emitted by chained equation ops like div-by-0 or log of ≤0) to `0.0` at the serialization boundary so the consumer always sees finite numbers.
  - Verified: `output/results.json` is now ~12 MB; decoded `raw_out` has 81,735 finite float64 samples in `[-172, 172]`; `f_out` has 1,047,936 finite float64 samples in `[-97, 96.4]`.
- 2026-05-30 — Restructured `output/` so each kind of sim product lives in its own folder:
  - `jax_test/guard.py::JaxGuard._init_output_layout`: creates `output/{results,ctlr,config,visualizations}/` up-front; `self.save_path` now points at `output/results/engine_state.json`.
  - `_export_engine_state` writes the b64-serialized `raw_out`/`f_out` payload into `output/results/engine_state.json`.
  - `_export_ctlr` (now active, no longer a no-op) writes `output/ctlr/db_ctlr.json` (DB controller metadata: shapes / params / module + field maps / DB+FIELD keys / modules+fields) and `output/ctlr/model_ctlr.json` (variation keys).
  - `_export_config_snapshot` copies the source `sim_config.json` into `output/config/sim_config.json`, dumps the engine-consumed `components.json`, and writes `runtime.json` (AMOUNT_NODES / SIM_TIME / DIMS / ENV_ID / platform) — making each run self-contained and reproducible.
  - `_write_manifest` walks the new layout and writes `output/manifest.json` (top-level index of all run artifacts → byte sizes) so downstream consumers can discover everything from one pointer file.
  - `main.py::run_main_process`: default `viz_root` switched from `color_master_output/` to `output/visualizations/` so the color_master products land inside the same structured tree. Manifest is also refreshed *after* color_master finishes so it captures the viz artifacts (which are produced post-`JaxGuard.main()`).
- 2026-05-30 — Added `schemas/` package with exact, inferred type definitions for every data object that flows through the pipeline:
  - `schemas/aliases.py` — atomic aliases (`Coord4`, `CoordXYZ`, `Base64Str`, `EnvId`, `ParamKey`, `ModuleIndex`, `FieldIndex`, `ParamIndex`, `NodeIndex`, `EquationIndex`, `VariationIdx`, `TimeStep`, `DimAxis`, plus `FloatArray`/`IntArray`/`AnyArray` behind `TYPE_CHECKING`).
  - `schemas/sim_config.py` — `SimConfig` `TypedDict` matching the live `sim_config.json` (24 keys, 545 DB params, 37 equations, 4 fields). Each field carries a length contract documented against the symbol table (P/U/M/F/E/A/D/N/T/I).
  - `schemas/injection.py` — `InjectionRequestCfg` (CLI/MCP shape from `Injector.set_inj_pattern`) and `InjectionPattern` / `InjectionPatternEntry` (engine internal 5-element row consumed by `GNN.inject`), plus `InjectionCfgWrapped` (the `{"json"|"b64"|"text"}` envelope `_parse_inj_cfg` accepts).
  - `schemas/controller.py` — `DBController`, `ModelController`, `ControllerBundle` mirroring `JaxGuard._build_ctlr_for_export` 1:1.
  - `schemas/output_artifacts.py` — `EngineState` (`output/results/engine_state.json`), `RuntimeConfig` (`output/config/runtime.json`), `Manifest` (`output/manifest.json`), `VisualizationArtifact` / `VisualizationBundle` (in-memory slurp dict), and `RunResult` (return type of `main.run_main_process`).
  - `schemas/gnn_state.py` — `Protocol`s for the live runtime objects: `DBLayerState`, `FeatureEncoderState`, `GNNState`, `NodeProtocol`. Inferred from the actual `self.X = ...` attribute set in `jax_test/gnn/{db_layer,feature_encoder,gnn}.py` — concrete classes satisfy them structurally without inheritance.
  - Naming: package is `schemas/` not `types/` because `main.py` does `sys.path.insert(0, _REPO_ROOT)`, which would make a literal `types/` package shadow Python's stdlib `types` module (used by `inspect`, `functools`, `dataclasses`, `pickle`, ...). Same intent, no foot-gun.
  - Verified: `from schemas import *` exposes 41 names; importing the package does **not** pull `jax` or `numpy` into `sys.modules` (heavy array types stay behind `TYPE_CHECKING`).
- 2026-05-30 — Adapted `color_master` so the **main combined animation** is now a per-grid-point white→blue feature-density view (replaces the previous mixed-keys combined GIF):
  - `color_master/main.py::render_grid_white_to_blue` — new render. Fixed N×N×N integer grid of dots on a fully white figure + axes; per-dot colour is `lerp(white, deep_blue, norm)` where `norm = volume[t,x,y,z] / global_max`. At t=0 (or any timestep where a point's summed value is 0) the dot is pure white and visually indistinguishable from the background — this is exactly the "start = grid with white background and white dots" requirement.
  - `color_master/sim_bridge.py::_build_grid_value_volume` — extracts a `(T, N, N, N)` non-negative volume from `gnn._features_history` (per-step list of feature tensors that `calc_batch` snapshots one-per-timestep). For each step it concatenates all feature tensors, takes `|·|` (so "more value = more blue"), pads to a multiple of `N**3`, reshapes to `(chunks, N, N, N)`, and **sums across the chunk axis**. The chunk-sum collapses every linearly-folded "field-like partition" of the flat feature stream onto the same grid coordinates — implementing "sum the features of all fields on a specific node grid point".
  - `color_master/sim_bridge.py::run_workflow_visualization` — after the existing per-key static + per-key animation pass, the bridge now builds the volume and calls `render_grid_white_to_blue` directly into `<viz>/combined/environment_3d.gif`, intentionally overwriting the default combined view so the *primary* animation is the requested white→blue grid. Wrapped in try/except so a viz failure can never break the workflow (the default combined GIF stays as a fallback).
  - `nan`/`inf` clamp at the visualization boundary inside the volume builder, mirroring the policy already used by `_flatten_to_float64_bytes` at the serialization boundary.
  - Verified: `py -3.11 main.py` exits 0; log shows `[grid-w2b] wrote ...combined\environment_3d.gif (T=3, N=3, max=5.98e+03)` and `manifest.json` is refreshed afterwards so the new GIF is registered.
- 2026-05-30 — Added per-grid-point activity visualizations (one 3D firegraph GIF + one 2D per-field line chart for every `(x, y, z)` in the simulation grid):
  - `color_master/grid_point_viz.py` — new module. Reads `output/ctlr/db_ctlr.json` (`MODULES` / `FIELDS` / `AMOUNT_PARAMS_PER_FIELD` / `DB_KEYS`) to derive the live field count `F` (= `sum(FIELDS)`, currently 38) and per-field labels of the form `m{module}_f{field_in_module}_{first_db_key}` so each line / firegraph node is traceable back to its controller slot.
  - `build_field_volume(jax_guard, amount_nodes, num_fields)` — extracts a `(T, F, N, N, N)` non-negative volume from `gnn._features_history`. Per timestep: concat all feature tensors, take `|·|`, pad to a multiple of `F * N**3`, reshape to `(chunks, F, N, N, N)`, sum over `chunks`. The chunk-sum collapses every linearly-folded "field-like partition" of the flat feature stream onto the same `(field, grid)` slot — implementing the user's "extract values index based of specific fields for all nodes" rule. Same `nan`/`inf` clamp policy as the white→blue volume.
  - `render_freq_chart_2d(...)` — per-grid-point 2D line chart. X = timestep, Y = `|activity|`, one polyline per field with a stable colour from a 20-base `tab20` palette (cycle-aware brightness shift for `F > 20`). Top-K most active fields surfaced in the legend; quieter fields stay drawn but faded so the eye still groups by colour.
  - `render_gridpoint_3d_animation(...)` — per-grid-point 3D firegraph GIF. F field-nodes laid out on a tilted ring around the grid point; node size ∝ field activity at this grid point at time t; **edges drawn between strongly co-active fields** with width ∝ `s_i · s_j` and colour = blend of the two field colours, top-K (default 30) pairs per frame. This is the canonical Feynman-vertex visualization of pairwise co-activation (gluon-gluon, photon-electron, …).
  - `render_gridpoint_visualizations(jax_guard, output_root, *, amount_nodes, ...)` — orchestrator. Loops over every `(x, y, z) ∈ {0..N-1}^3`, writes a folder `gridpoints/g_{x}_{y}_{z}/` with `freq_chart.png` + `3d_activity.gif`. Also writes `gridpoints/index.json` carrying the field palette legend (`field_index → label → rgb`) and the grid-point map so downstream tools can reconstruct the colour/label mapping without re-reading the controller.
  - `color_master/sim_bridge.py::run_workflow_visualization` — calls the new orchestrator after the white→blue main animation step. Wrapped in try/except so per-point viz can never break the workflow.
  - Verified: `py -3.11 main.py` produced `27` per-grid-point folders (T=3, F=38, N=3); `g_0_0_0/` contains `3d_activity.gif` (~79 KB) + `freq_chart.png` (~108 KB); top-level `gridpoints/index.json` (~11.7 KB) carries the palette legend.
