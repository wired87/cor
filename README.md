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

### Local engine control GUI

Modular control terminal (does not modify the core pipeline):

```bash
pip install -r gui/requirements.txt
python -m gui
```

Opens `http://127.0.0.1:8765` — set sim params, run the engine in a background thread, stream live terminal output, and inspect flat `output/` artifacts.

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
- 2026-06-07 — Added modular `gui/` local engine control terminal (NiceGUI):
  - `gui/engine_bridge.py` — lazy-imports `main.run_main_process`, runs in background thread with stdout tee.
  - `gui/panels/{control,terminal,dashboard}.py` — sim params, live log stream, flat `output/` dashboard + GIF preview.
  - Entry: `python -m gui` (port 8765); zero edits to core pipeline modules.
- 2026-06-07 — Moved `runtime.json` export from `JaxGuard` to root `Guard`:
  - `guard.py::Guard._export_runtime`: writes flat `output/runtime.json` from constructor args (`amount_nodes`, `sim_time`, `dims`) + `main(env_id)` — no `os.environ` lookups.
  - `jax_test/guard.py::_export_config_snapshot`: no longer writes `runtime.json` (only `sim_config.json` + `components.json`).
- 2026-06-07 — Added architecture plans for upcoming blueprint + adaptive-sim workflows:
  - `future_perspectives_01_image_blueprint_injection.md` — image → centered 3D line blueprint → `InjectionRequestCfg` → post-`output/` adaptation.
  - `future_perspectives_02_sim_analysis_adaptive_loop.md` — JAX pathfinder loop on `jax_test/gnn/pathfinder.py` for unknown-objective runs.
- 2026-06-07 — Refactored `color_master` into modular layout (removed unused MCP/Docker/test artifacts):
  - `workflow.py` — `run_workflow_visualization`; `series_collect.py` — JaxGuard data extraction; `path_viz.py` — indexed offline viz; `render/` — matplotlib renderers; `types.py` / `engine_payload.py` / `viz_config.py` / `grid_point.py` — shared I/O.
  - Repo `main.py` imports `color_master.workflow` directly; `sim_bridge.py` kept as backward-compat shim.
- 2026-06-07 — Debugged main pipe for fast runtime + flat `output/` layout:
  - Gated hot-loop stdout behind `COR_VERBOSE=1` in `guard.py`, `jax_test/gnn/{gnn,gnutils,feature_encoder}.py`, and removed vmap-path prints in `jax_test/mod.py::Node.core` — prior runs were I/O-bound on thousands of per-equation log lines and appeared hung during JAX sim.
  - `jax_test/guard.py::_init_output_layout`: all engine JSON artifacts now write flat into `output/` (`engine_state.json`, `db_ctlr.json`, `model_ctlr.json`, `sim_config.json`, `components.json`, `runtime.json`, `manifest.json`).
  - `main.py`: default `output_dir=output/`; skips base64 viz slurp unless `COR_SLURP_VIZ=1`.
  - `color_master/sim_bridge.py`: light preset skips per-key 3D build; main GIF → `output/environment_3d.gif`; per-grid-point viz off unless `COR_GRIDPOINT_VIZ=1`.
  - `jax_test/gnn/db_layer.py` + `gnn.py`: controller index arrays use `int32` (matches default JAX on Windows, silences int64 truncation warnings).
  - Verified: `py -3.11 main.py` exits 0; `output/engine_state.json` ~12 MB; `output/environment_3d.gif` written flat.
- 2026-05-30 — Added `qfu/gluon_neighbors.py` gluon 3D mesh neighbor resolver:
  - `build_gluon_neighbor_map(amount_nodes)` → `dict[px_id, dict[gluon_k, list[(neighbor_px, neighbor_gluon_j)]]]`. Each pixel is `px_{x}_{y}_{z}`; gluons are `gluon_0..7` (sub-cube corners). Uses `FieldUtils.shift_dirs` for the same 26-offset stencil as `qf_utils.npm` / `all_px_neighbors`. Sub-cube face matching picks the partner gluon at the shared interface (corner gluons own one octant ≈ 7 directions each; all 8 gluons cover the full 26-direction stencil per pixel). Verified: `py -3.11 qfu/gluon_neighbors.py` with `AMOUNT_NODES=3`.

## Theory (from here on)

Each **grid point** represents one **3D cube** (pixel / QFN cell). Inside that cube sit the **8 gauge sub-points** (`gluon_0` … `gluon_7`) as corners of the unit cell. **Gauge interaction** (gluon–gluon, gluon–quark, …) defines what happens **inside** the cube and at its faces to neighboring cubes; **fermion / quark** fields carry the internal state whose evolution is driven by those gauge couplings. The simulation stack should treat **topology** (who couples to whom across the 26-neighbor stencil) and **tensor layout** (flat DB slices with identical length per method) as separate, coordinated concerns — not as millions of blind graph edges.

## TODO — gauge mesh & uniform DB shape

- **Problem:** Per 3D gauge sub-point, interaction partners are **offset** (face-matched gluon indices differ by direction). A naive static edge `g1 → g2` does not capture the full stencil; the correct pattern looks more like `g1 → [g2, g6, g2, g6, …]` per direction slot — same **total shape length** for every method param, but **partner index varies** by mesh direction.
- **Idea:** Shape length can still be held **constant** through a **logical mix of grids** (search for improvements): encode the 26-direction neighbor gluon indices as a **fixed-width partner list** per source gluon, not as one global `g1→g2` pair. Result: all method inputs keep the same flat length; only the **index map** changes per slot.
- **`Guard.create_db`:** extend so **`db_to_method`** builds **`split_grid` including coords directly** (grid index + sub-gluon index + direction baked into the slice metadata, not inferred later in JAX).
- **`method_to_db` / `set_edge_method_to_db`:** consume the same **`split_grid` → unified format** so `METHOD_TO_DB` rows and `DBLayer.extract_flattened_grid` agree on layout.
- **Reference:** `qfu/gluon_neighbors.py` (`paired_neighbor_gluon_index`, `build_gluon_neighbor_map`) for the static partner-index rules; export compact mesh arrays in `components` (CSR / stencil) rather than materializing full nx edge lists.
- **Open:** evaluate stencil / roll-based JAX ops vs. explicit partner lists for `gg_coupling` and related gauge equations once `split_grid` is unified.
