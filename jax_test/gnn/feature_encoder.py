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

    def __init__(
            self,
            METHODS,
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

        self.AXIS = AXIS

        # keep list and not flatten since
        self.in_store = []
        self.out_store = []

        #
        self.out_skeleton = []
        self.in_skeleton = []
        self.in_ts = []  # CHANGED: gnn.py serialize() expects feature_encoder.in_ts

        self.in_linears = []

        #
        self.out_linears = [] # get set in create_out_linears_process
        self.out_feature_store = []

        self.result_blur = .9

        self.feature_controller = []





    def create_out_features(
            self,
            output, # represent grid for each variation list[grid]
            eq_idx,
    ):
        """
        Stack results to out_f_store (123,123,123) apply later (existing) operation
        for (111,222,333) -> just append resutls in create_out_features, saves t in rt
        """
        print("FeatureEncoder.out_processor...")
        try:
            for grid in output:
                results = self.gen_out_feature_single_variation(
                    grid,
                    out_linear=self.out_linears[eq_idx],
                )
                self.out_feature_store[eq_idx].extend(results)
        except Exception as e:
            print("Err FeatureEncoder.out_processor:", e)
        print("FeatureEncoder.out_processor... done")

    @property
    def ctlr(self):
        """Short alias for docs / callers: same object as feature_controller."""
        return self.feature_controller



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
            return embedding
        except Exception as e:
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
        param_grid, # represent all params
        out_linear
    ):
        """For each result withi the param grid, create embeddings -> return array"""
        grid_embeds =  []
        for item in param_grid:
            embedding = out_linear(item)
            grid_embeds.append(embedding)
        print("gen_feature_single_variation... done")
        return grid_embeds



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

    def build_single_linear(self, ilen):
        print("build_single...")
        linear = eqx.nn.Linear(
            in_features=ilen,
            # dimensions for the model
            out_features=self.d_model,
            key=self.rngs,
        )
        print("build_single... done")
        self.in_linears.append(linear)


    def create_in_features(
            self,
            inputs,
            eq_idx=0,
    ):
        print("create_in_features for", eq_idx)
        try:
            results = []
            linear_instance = self.linears[eq_idx]
            for variation_input in inputs:
                feature = jax.nn.gelu(
                    linear_instance(
                        jnp.array(variation_input)
                    )
                )
                results.append(feature)
            return results
        except Exception as e:
            print("Err create_in_features:", e)
        print("create_in_features... done")
        return None

    def save_features(self, all_features):
        """SAVE ALL FEATURES FOR SINGLE ITER ALL EQ"""
        for i, feature in enumerate(all_features):
            self.features[i].append(feature)
        print("FEATURES SAVED SUCCESSFULLY...")

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
            print("Err convert_feature_to_rows", e)


    def create_out_linear(
            self,
            feature_example
    ):
        print("create_out_linears...")

        self.out_linears.append(
            eqx.nn.Linear(
                in_features=feature_example ,
                out_features=self.d_model,
                key=self.rngs
            )
        )
        self.out_feature_store.append([])
        print("create_out_linears... done")

"""
self._save_in_feature_method_grid(
    f_in_results,
    eq_idx,
    item_idx,
)


def build_linears(
        self,
        eq_idx,
        unscaled_db_len,
):
    print(f"build_linears for {eq_idx}...")
    for item in unscaled_db_len:
        item = jnp.atleast_1d(jnp.asarray(item))
        merged_item = jnp.ravel(item)
        self.build_single(
            ilen=merged_item
        )
    self.in_linears.append(linear)
    print(f"build_linears... done")



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

        
        self.out_linears.append(linears)
        self.out_feature_store.append([])
        print("create_out_linears... done")
"""