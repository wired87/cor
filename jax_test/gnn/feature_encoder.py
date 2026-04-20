from typing import Any, List

import jax
from jax import vmap
import jax.numpy as jnp
import equinox as eqx

class FeatureEncoder(eqx.Module):
    # represents singe eq

    # todo do not get linears from db because difeent variatons (e.g. ff interaction) need diferent linears
    # tod build controller with each timestep ( track variations per eq dynamically )

    # Equinox registers PyTree fields from class annotations; custom __init__ may only assign these names.
    rngs: Any
    d_model: int
    amount_variations: int
    db_layer: Any
    AXIS: Any
    in_store: List[Any]
    out_store: List[Any]
    out_skeleton: List[Any]
    in_skeleton: List[Any]
    in_ts: List[Any]
    in_linears: List[Any]
    out_linears: List[Any]
    result_blur: float
    feature_controller: List[Any]
    # gien: eqx.Module is frozen after __init__ — keep step in a static list we mutate in-place.
    _current_step_ref: Any = eqx.field(static=True)

    def __init__(
            self,
            AXIS,
            db_layer,
            amount_variations:int,
            d_model: int = 64,
    ):
        # Wir speichern für jede Eingabe eine Projektion auf 64 Dimensionen
        self.rngs = jax.random.PRNGKey(42)
        self.d_model = d_model
        self.amount_variations = amount_variations
        self.db_layer = db_layer

        self.AXIS=AXIS

        # keep list and not flatten since
        self.in_store = []
        self.out_store = []
        self.out_skeleton = []
        self.in_skeleton = []
        self.in_ts = []  # CHANGED: gnn.py serialize() expects feature_encoder.in_ts

        self.in_linears = []
        self.out_linears = []

        self.result_blur = .9

        self.feature_controller = []
        object.__setattr__(self, "_current_step_ref", [0])

    @property
    def ctlr(self):
        """Short alias for docs / callers: same object as feature_controller."""
        return self.feature_controller

    # --- CTLR helpers ---------------------------------------------------------
    def begin_step(self, step: int):
        """
        Mark start of a simulation step for controller tracking.
        GNN.simulate(step) should call this once per time step.
        """
        print("begin step", )
        ref = self._current_step_ref
        try:
            step = int(step)
        except Exception as e:
            print(f"Err core.jax_test.gnn.feature_encoder::FeatureEncoder.begin_step | handler_line=77 | {type(e).__name__}: {e}")
            print(f"[exception] core.jax_test.gnn.feature_encoder.FeatureEncoder.begin_step: {e}")
            print("Err begin step:", e)
            step = int(ref[0])
        ref[0] = step

        # ensure outer list has slot for this step
        while len(self.feature_controller) <= step:
            self.feature_controller.append({"step": len(self.feature_controller), "eq": {}})

    def _get_ctlr_entry(self, eq_idx: int) -> dict:
        """
        Get (and lazily create) controller dict for current step + equation.
        """
        step = int(self._current_step_ref[0])
        # ensure step slot exists
        while len(self.feature_controller) <= step:
            self.feature_controller.append({"step": len(self.feature_controller), "eq": {}})

        step_entry = self.feature_controller[step]
        if "eq" not in step_entry or step_entry["eq"] is None:
            step_entry["eq"] = {}
        if eq_idx not in step_entry["eq"]:
            step_entry["eq"][eq_idx] = {}
        return step_entry["eq"][eq_idx]


    def get_linear_row(self, eq_idx, row_idx):
        print("get_linear_row...")
        return jax.tree_util.tree_map(
            lambda x: jnp.take(x, row_idx, axis=0),
            jnp.take(jnp.array(self.in_linears), eq_idx),
        )


    def gen_feature(
            self,
            param,
            linear,
    ):
        # embed linear:nnx.Linear
        print("gen_feature...")
        try:
            embedding = jax.nn.gelu(linear(param))
            #print("embedding", embedding)
            return embedding
        except Exception as e:
            print(f"Err core.jax_test.gnn.feature_encoder::FeatureEncoder.gen_feature | handler_line=123 | {type(e).__name__}: {e}")
            print(f"[exception] core.jax_test.gnn.feature_encoder.FeatureEncoder.gen_feature: {e}")
            print("Err gen_feature", e)

    def gen_in_feature_single_method_variation(
        self,
        param_grid, # current param entries
        linear_batch, # history param entries
    ):
        """
        GENERATE FEATURES WITH GIVEN LINEARS
        # todo use segment for pre segementation of
        # values instead of flatten them again
        """
        print("gen_feature_single_variation...")

        feature_block_single_param_grid = []


        # CREATE EMBEDDING



        try:
            for i, (param, linear) in enumerate(
                    zip(param_grid, linear_batch)
            ):
                if linear:
                    try:
                        #
                        embedding = jax.nn.gelu(linear(jnp.array(param)))

                        #
                        feature_block_single_param_grid.append(embedding)

                    except Exception as e:
                        print(f"Err _work_param at index {i}: {e}")

            # Am Ende alles zu einem JAX-Array zusammenfügen
            features = jnp.array(feature_block_single_param_grid)

        except Exception as e:
            print("Err gen_in_feature_single_method_variation", e)
            return jnp.array([])

        print("gen_feature_single_variation... done")
        return features


    def gen_out_feature_single_variation(
        self,
        param_grid,
        out_linears
    ):
        """
        Generate out-features: one embedding per (segment, linear) pair.
        --- FIX: Accept list of segments (one per out_linear) and loop; vmap fails when
        axis 0 sizes differ (e.g. flatten_param 36 vs idx 37). ---
        """
        print("gen_feature_single_variation...")
        #print("param_grid", type(param_grid))
        #print("out_linears", type(out_linears))

        if not out_linears:
            pg = jnp.atleast_1d(jnp.asarray(param_grid))
            if isinstance(param_grid, list):
                n = len(param_grid)
            else:
                n = pg.shape[0]
            return jnp.zeros((n, self.d_model), dtype=jnp.float32)

        # --- FIX: param_grid is list of segments (one per linear); loop to avoid vmap size mismatch ---
        if isinstance(param_grid, (list, tuple)):
            out_list = []
            for i, (seg, linear) in enumerate(zip(param_grid, out_linears)):
                seg = jnp.asarray(seg).ravel()
                in_len = int(getattr(linear, "in_features", seg.size))
                if seg.size < in_len:
                    seg = jnp.concatenate([seg, jnp.zeros(in_len - seg.size, dtype=seg.dtype)])
                elif seg.size > in_len:
                    seg = seg[:in_len]
                out_list.append(jax.nn.gelu(linear(seg)))
            print("gen_feature_single_variation... done")
            return jnp.stack(out_list, axis=0)
        # Fallback: single array (legacy); if leading dim != len(out_linears), loop to avoid vmap mismatch
        try:
            param_grid = jnp.atleast_1d(jnp.asarray(param_grid))
            n_rows = param_grid.shape[0]
            if n_rows != len(out_linears):
                out_list = []
                for i in range(len(out_linears)):
                    in_len = int(jnp.asarray(getattr(out_linears[i], "in_features", 1)).ravel()[0])
                    seg = param_grid[i] if i < n_rows else jnp.zeros(in_len)
                    seg = jnp.asarray(seg).ravel()
                    if seg.size < in_len:
                        seg = jnp.concatenate([seg, jnp.zeros(in_len - seg.size, dtype=seg.dtype)])
                    else:
                        seg = seg[:in_len]
                    out_list.append(jax.nn.gelu(out_linears[i](seg)))
                return jnp.stack(out_list, axis=0)
            idx_map = jnp.arange(len(out_linears))
            def _work_param(flatten_param, idx: int):
                return jax.nn.gelu(out_linears[idx](flatten_param))
            features = vmap(_work_param, in_axes=(0, 0))(param_grid, idx_map)
        except Exception as e:
            print(f"Err core.jax_test.gnn.feature_encoder::FeatureEncoder.gen_out_feature_single_variation | handler_line=225 | {type(e).__name__}: {e}")
            print(f"[exception] core.jax_test.gnn.feature_encoder.FeatureEncoder.gen_out_feature_single_variation: {e}")
            print("Err gen_feature_single_variation", e)
            return jnp.zeros((len(out_linears), self.d_model), dtype=jnp.float32)
        print("gen_feature_single_variation... done")
        return features



    def _save_in_feature_method_grid(
            self,
            features_tree:jnp.array,
            eq_idx,
            item_idx
    ):
        try:
            if len(self.in_store) <= eq_idx:
                self.in_store.extend(
                    [[] for _ in range(eq_idx + 1 - len(self.in_store))]
                )
            if len(self.in_store[eq_idx]) <= item_idx:
                self.in_store[eq_idx].extend(
                    [[] for _ in range(item_idx + 1 - len(self.in_store[eq_idx]))]
                )

            self.in_store[eq_idx][item_idx].append(features_tree)
        except Exception as e:
            print("Err _save_in_feature_method_grid", e)


    def build_linears(self, eq_idx, unscaled_db_len):
        print(f"build_linears for {eq_idx}...")

        def build_single(ilen):
            # Equinox Linear uses keyword `key` (PRNGKey), not Flax-style `rngs`.
            linear = eqx.nn.Linear(
                in_features=ilen,
                out_features=self.d_model,
                key=self.rngs,
            )
            return linear

        linears = []
        for item in unscaled_db_len:
            #print("item", item)
            var_linears = []
            item = jnp.atleast_1d(jnp.asarray(item))
            for variation in item:
                #print("item", item)
                if variation > 0:
                    var_linears.append(
                        build_single(
                            ilen=variation
                        )
                    )
                else:
                    # fallback
                    var_linears.append(None)

            linears.append(var_linears)
        self.in_linears.append(linears)
        print(f"build_linears... done")


    def create_in_features(
            self,
            inputs,
            axis_def,
            eq_idx=0,
    ):
        """
        :param inputs: array stores for single param values of the entire grid.
        """
        print("create_in_features...")
        try:
            results = []

            # create embeddings for all scaled instances and immediately store them
            for item_idx, (single_param_input_grid, linear_instance, ax) in enumerate(
                zip(inputs, self.in_linears[eq_idx], axis_def)
            ):
                #
                f_in_results = self.gen_in_feature_single_method_variation(
                    single_param_input_grid,
                    linear_instance,
                )
                results.append(f_in_results)

                # append to in_store[eq_idx][item_idx][t]
                self._save_in_feature_method_grid(
                    f_in_results,
                    eq_idx,
                    item_idx,
                )

            """try:
                ctl = self._get_ctlr_entry(eq_idx)
                ctl["in"] = {
                    "axis_def": tuple(axis_def) if axis_def is not None else None,
                    "n_params": len(inputs) if inputs is not None else 0,
                    "in_shapes": [
                        list(getattr(r, "shape", ())) if hasattr(r, "shape") else None
                        for r in (results or [])
                    ],
                }
            except Exception as _e_ctlr_in:
                print("Err ctlr(in):", _e_ctlr_in)"""
            return results
        except Exception as e:
            print("Err create_in_features:", e)
        print("create_in_features... done")
        return None



    def blur_result_from_in_tree(
            self,
            eq_idx,
            high_score_elements,
            len_variations,
    ):
        """
        high_score_elements: list of 1D arrays (one per param), each shape (len_variations,).
        Returns a list of length len_variations: per-row blur value or None (must recompute).
        """
        if len_variations == 0 or not high_score_elements:
            print("blur_result_from_in_tree... done (no variations)")
            return jnp.full((max(1, len_variations), self.d_model), jnp.nan)

        def _to_len(s):
            a = jnp.asarray(s).ravel()
            n = a.shape[0]
            if n >= len_variations:
                return a[:len_variations]
            return jnp.concatenate([a, jnp.zeros(len_variations - n, dtype=a.dtype)])
        padded = [ _to_len(s) for s in high_score_elements ]
        stacked = jnp.stack(padded, axis=0)
        row_scores = jnp.sum(stacked, axis=0)

        # Sentinel for "must recompute": same shape (d_model,) so vmap gets uniform structure.
        recompute_sentinel = jnp.full((self.d_model,), jnp.nan)

        def _get_blur_val(row_idx):
            score = row_scores[row_idx]
            if score <= self.result_blur and len(self.out_store) > eq_idx and len(self.out_store[eq_idx]) > 0:
                last_out = self.out_store[eq_idx][-1]
                n_out = last_out.shape[0] if hasattr(last_out, "shape") else len(last_out)
                if row_idx < n_out:
                    return jnp.asarray(last_out[row_idx])
            return recompute_sentinel

        result = jnp.stack([_get_blur_val(r) for r in range(len_variations)], axis=0)
        print("blur_result_from_in_tree... done")

        # --- CTLR: track blur / reuse statistics for current step & equation ---
        try:
            ctl = self._get_ctlr_entry(eq_idx)
            reuse_mask = row_scores[:len_variations] <= self.result_blur
            ctl["blur"] = {
                "len_variations": int(len_variations),
                "row_scores": jnp.asarray(row_scores[:len_variations]),
                "reuse_mask": jnp.asarray(reuse_mask),
            }
        except Exception as _e_ctlr_blur:
            print("Err ctlr(blur):", _e_ctlr_blur)
        return result





    def create_out_features(
            self,
            output,
            eq_idx,
    ):
        # Track per-time-step out-features for each equation.
        print("FeatureEncoder.out_processor...")
        try:
            # --- FIX: Pass list of segments (one per out_linear) to avoid vmap axis size mismatch ---
            results = self.gen_out_feature_single_variation(
                output,
                out_linears=self.out_linears[eq_idx],
            )

            # lazily init store in case prep() did not run as expected
            if len(self.out_store) <= eq_idx:
                self.out_store.extend(
                    [[] for _ in range(eq_idx + 1 - len(self.out_store))]
                )

            # append current time-step embedding block
            out_block = jnp.array(results)
            self.out_store[eq_idx].append(out_block)

            # --- CTLR: track per-step, per-eq out-feature meta for later iteration ---
            try:
                ctl = self._get_ctlr_entry(eq_idx)
                ctl["out"] = {
                    "n_out": int(out_block.shape[0]) if hasattr(out_block, "shape") and out_block.ndim >= 1 else 0,
                    "d_model": int(out_block.shape[1]) if hasattr(out_block, "shape") and out_block.ndim >= 2 else self.d_model,
                }
            except Exception as _e_ctlr_out:
                print(f"Err core.jax_test.gnn.feature_encoder::FeatureEncoder.create_out_features | handler_line=441 | {type(_e_ctlr_out).__name__}: {_e_ctlr_out}")
                print(f"[exception] core.jax_test.gnn.feature_encoder.FeatureEncoder.create_out_features: {_e_ctlr_out}")
                print("Err ctlr(out):", _e_ctlr_out)
        except Exception as e:
            print(f"Err core.jax_test.gnn.feature_encoder::FeatureEncoder.create_out_features | handler_line=444 | {type(e).__name__}: {e}")
            print(f"[exception] core.jax_test.gnn.feature_encoder.FeatureEncoder.create_out_features: {e}")
            print("Err FeatureEncoder.out_processor:", e)
        print("FeatureEncoder.out_processor... done")


    def fill_blur_vals(self, feature_row, prev_params):
        # similarity search (ss) nearest neighbor filter blur
        try:
            print("fill_blur_vals...")
            embeddings = jnp.stack(prev_params, axis=0)

            # JUST SINGLE ENTRY?
            if embeddings.ndim == 1:
                embeddings = jnp.reshape(embeddings, (1, -1))

            ec = feature_row
            if jnp.ndim(ec) == 1:
                ec = jnp.reshape(ec, (1, -1))

            # L2-Distanz zu allen (axis=-1 works for 1D and 2D; axis=1 fails for 1D)
            diff = embeddings - ec

            losses = jnp.linalg.norm(diff, axis=-1)

            # min entry returns idx (everythign is order based (fixed f-len)
            idx = jnp.argmin(losses)
            min_loss = losses[idx]
            return min_loss
        except Exception as e:
            print("Err fill_blur_vals", e)


    def get_precomputed_results(
            self,
            in_features,
            eq_idx,
    ):
        print("get_precomputed_results...")

        def generate_high_score_single(param_embedding_item, pre_param_grid):
            return self.fill_blur_vals(
                param_embedding_item,
                pre_param_grid,
            )

        try:
            high_score_map = []
            # LOOP EACH GRID OF SAME PARAMS DIFERENT POINTS
            for param_idx, param_embedding_grid in enumerate(in_features):
                # if we have no history yet, fall back to zeros
                if param_idx >= len(self.in_store[eq_idx]) or len(self.in_store[eq_idx][param_idx]) == 0:
                    high_score_map.append(
                        jnp.zeros(param_embedding_grid.shape[0])
                    )
                    continue

                scores = vmap(
                    generate_high_score_single,
                    in_axes=(0, None)
                )(
                    param_embedding_grid,
                    self.in_store[eq_idx][param_idx],
                )
                high_score_map.append(scores)
        except Exception as e:
            print("Err get_precomputed_results", e)
            high_score_map = []
        print("get_precomputed_results... done")
        return high_score_map


    def convert_feature_to_rows(
            self,
            axis_def,
            in_features,
    ):
        # convert past feature steps o rows
        print("convert_feature_to_rows...")
        print("convert_feature_to_rows in_features", [len(i) for i in in_features], axis_def)

        def batch_padding():
            for i, item in enumerate(in_features):
                if len(item) == 0:
                    padding = jnp.array([
                        jnp.zeros(self.d_model)
                        for _ in range(len(in_features[0]))
                    ])
                    in_features[i] = padding
            return in_features

        def _process(*item):
            print("_process...")
            _arrays = jax.tree_util.tree_map(jnp.array, item)
            _arrays = jax.tree_util.tree_map(jnp.ravel, _arrays)

            return jnp.concatenate(
                arrays=jnp.array(_arrays),
                axis=0
            )

        try:
            in_features = batch_padding()

            kernel = jax.vmap(
                fun=_process,
                in_axes=axis_def,
            )

            feature_rows = kernel(
                *in_features
            )
            print("convert_feature_to_rows... done")
            return feature_rows
        except Exception as e:
            print(f"Err core.jax_test.gnn.feature_encoder::FeatureEncoder.convert_feature_to_rows | handler_line=562 | {type(e).__name__}: {e}")
            print(f"[exception] core.jax_test.gnn.feature_encoder.FeatureEncoder.convert_feature_to_rows: {e}")
            print("Err convert_feature_to_rows", e)


    def create_out_linears(
            self,
            unscaled_db_len,
            feature_len_per_out,
    ):
        print("create_out_linears...")

        def _flat_ints(x):
            a = jnp.ravel(jnp.asarray(x))
            if a.size == 0:
                return []
            return [int(a[i]) for i in range(int(a.size))]

        u_dims = _flat_ints(unscaled_db_len)
        f_counts = _flat_ints(feature_len_per_out)
        n = max(len(u_dims), len(f_counts), 1)
        if len(u_dims) < n:
            u_dims = u_dims + [1] * (n - len(u_dims))
        if len(f_counts) < n:
            f_counts = f_counts + [1] * (n - len(f_counts))

        linears= []
        for in_dim, amount_features in zip(u_dims, f_counts):
            in_dim = max(1, int(in_dim))
            amount_features = max(0, int(amount_features))
            linears.extend([
                eqx.nn.Linear(
                    in_features=in_dim,
                    out_features=self.d_model,
                    key=self.rngs
                )
                for _ in range(amount_features)
            ])

        # transform 1d
        self.out_linears.append(linears)
        print("create_out_linears... done")

