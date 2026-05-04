"""
calc_batch / variation pipeline: keep edge coords as (V, P, 4); align extract_flat_params with
`ax_rows` length V; stack P tensors with leading dim V for Node vmap (fixes arity / vmap batch mismatch).
"""
import jax
from jax import jit, vmap
import jax.numpy as jnp

from jax_test.gnn.db_layer import DBLayer
from jax_test.gnn.feature_encoder import FeatureEncoder
from jax_test.gnn.gnutils import GNUtils
from jax_test.gnn.injector import InjectorLayer
from jax_test.jax_utils.conv_flat_to_shape import bring_flat_to_shape
from jax_test.mod import Node
from jax_test.utils import create_runnable, SHIFT_DIRS


class GNN(GNUtils):

    def __init__(
            self,
            gpu,
            **cfg,
    ):
        GNUtils.__init__(self)

        for k, v in cfg.items():
            setattr(self, k, v)

        self.model_feature_dims = 64

        # Generate grid coordinates based on amount_nodes dimensionality
        self.schema_grid = [
            (i, i, i)
            for i in range(getattr(self, "AMOUNT_NODES"))
        ]
        self.len_params_per_methods = {}
        self.change_store = []

        self.gpu = gpu

        # todo use runnable (after first versionnis deplyoed - currently just shcmatic)
        self.db_layer = DBLayer(
            gpu=self.gpu,
            **cfg
        )
        _n_var = int(
            jnp.sum(self.db_layer.DB_CTL_VARIATION_LEN_PER_FIELD)
        )

        self.feature_encoder = FeatureEncoder(
            self.METHODS,
            self.db_layer.AXIS,
            self.db_layer,
            amount_variations=_n_var,
        )

        self.injector = InjectorLayer(
            db_layer=self.db_layer,
            **cfg
        )

        self.linears = []
        self.axs_all_eqs = []
        self.in_shapes_all_eqs = []

        print("Node initialized and build successfully")

        # CHANGED: LEN_FEATURES_PER_EQ is list of variable-length per-eq -> jnp.array(...) gives inhomogeneous ValueError. Use lengths then cumsum.
        _len_per_eq = jnp.array([len(x) for x in self.LEN_FEATURES_PER_EQ])
        self.LEN_FEATURES_PER_EQ_CUMSUM = jnp.concatenate([
            jnp.array([0]),
            jnp.cumsum(_len_per_eq)
        ])
        self.LEN_FEATURES_PER_EQ_CUMSUM_UNPADDED = jnp.cumsum(_len_per_eq)

        self.FEATURES_CUMSUM = jnp.concatenate([
            jnp.array([0]),
            jnp.cumsum(
                jnp.array([len(i)
                           for i in self.LEN_FEATURES_PER_EQ])
            ),
        ])

    def main(self):
        print("start process...")
        self.prepare()
        self.simulate()
        jax.debug.print("process finished.")
        return self.serialie_input()


    def serialie_input(self):
        serialized_raw_out = self.serialize(
            self.db_layer.store
        )
        serialized_f_out = self.serialize(
            self.db_layer.out_f_store
        )
        print("serialization... done")
        return serialized_raw_out, serialized_f_out


    def prepare(self):
        # DB
        self.db_layer.build_db(
            getattr(self, "AMOUNT_NODES"),
        )
        # layer to define all ax et
        self.prep()

    def simulate(self):
        try:
            for step in range(getattr(self, "SIM_TIME")):
                jax.debug.print(
                    "Sim step {s}/{n}",
                    s=step,
                    n=getattr(self, "SIM_TIME")
                )
                self.calc_batch()
        except Exception as e:
            jax.debug.print("Err simulate: {m}", m=str(e)[:300])

    @jit
    def inject(self, step, db_layer):
        """
        Applies injections based on self.inj_pattern and current step.
        Supports the SOA structure: [module, field, param, node_index, schedule]
        where schedule is list of [time, value].
        """
        all_indices = []
        all_values = []

        for item in self.injection_pattern:
            if isinstance(item, list) and len(item) == 5 and isinstance(item[4], list):
                mod_idx, field_idx, param_idx, node_idx, schedule = item

                # Check schedule for current time
                for time_point, value in schedule:
                    if time_point == step:
                        all_indices.append([mod_idx, field_idx, param_idx, node_idx])  # rm 0?
                        all_values.append(value)

        if not all_indices:
            return

        all_indices = jnp.array(all_indices)  # shape [N, 5]
        all_values = jnp.array(all_values)  # shape [N]

        # inject step
        db_layer.nodes.at[
            tuple(all_indices.T)
        ].add(all_values)


    def get_mod_idx(self, abs_eq_idx):
        # get mod_idx based on eq_idx
        cum = jnp.cumsum(jnp.array(self.MODULES))  # kumulative Summe
        # finde erstes i, bei dem cum >= abs_eq_idx
        idx = jnp.argmax(cum >= abs_eq_idx)
        return idx

    def reshape_variant_block(self, edges, total_amount_params_current_eq, amount_params_current_eq):
        # to split edges into n(len(variations)) parts
        result = edges.reshape(
            total_amount_params_current_eq // amount_params_current_eq,
            amount_params_current_eq,
            4, # blocks
        )
        return result

    def prep(self):
        print("prep...")
        # SET DEF DB IDX (faster in runtime)
        self.db_layer.create_out_idx_map()

        for eq_idx, eq in enumerate(self.METHODS):
            variations, amount_params_current_eq, total_amount_params_current_eq = self.extract_eq_variations(
                eq_idx,
            )

            # prepare shapes struct
            self.in_shapes_all_eqs.append([])

            # GET METHOD VARIATION
            variations = self.reshape_variant_block(
                variations,
                total_amount_params_current_eq,
                amount_params_current_eq
            )
            #example_var = variations[0]
            for i, item in enumerate(variations):
                # ! item = all in for entire grid
                axs, shapes = self.db_layer.get_axis_shape(
                    item,
                )

                # prepare linear space for shape related entries
                self.feature_encoder.in_linears[eq_idx].append([])
                self.in_shapes_all_eqs[eq_idx].append([])

                # COLLECT META EACH EQ
                self.axs_all_eqs.append(axs)
                self.in_shapes_all_eqs[eq_idx][i].append(shapes)

                # SAVE LINEAR, SINGLE, FOR EACH EQ
                """
                Each shape represents grid (spec param)
                loop shapes -> create linear each 
                """
                for shape in shapes:
                    # create placeholder
                    flat = jnp.zeros(int(jnp.prod(jnp.array(shape))), dtype=jnp.int64)

                    # create&save eq linears
                    self.feature_encoder.build_single_linear(
                        ilen=flat,
                        eq_idx=eq_idx,
                        variation_idx=i
                    )

            # Create in feature store dims
            self.feature_encoder.in_f_store[eq_idx] = [
                []
                for _ in range(len(variations))
            ]

            #
            self.create_out_linears_process()

            node = self.create_node(eq, eq_idx)
            self.METHODS[eq_idx] = node
        print("prep... done")


    # gien: `axs_all_eqs` is appended once per variation row globally — recover the slice for equation `eq_idx`
    def _variation_axis_rows_for_eq(self, eq_idx: int) -> list:
        _off = sum(len(self.in_shapes_all_eqs[k]) for k in range(eq_idx))
        _n = len(self.in_shapes_all_eqs[eq_idx])
        return self.axs_all_eqs[_off : _off + _n]

    def _stack_node_inputs(self, inputs_nested):
        # gien: Node expects P arguments, each batched on axis 0 with size V (not V*P flat tensors)
        if not inputs_nested:
            return [], tuple()
        vn = len(inputs_nested)
        pn = len(inputs_nested[0])
        stacked: list = []
        for p in range(pn):
            cols = [jnp.asarray(inputs_nested[v][p], dtype=jnp.float32) for v in range(vn)]
            sh0 = [tuple(c.shape) for c in cols]
            if len(set(sh0)) == 1:
                stacked.append(jnp.stack(cols, axis=0))
                continue
            # gien: ragged rows / cols — pad/truncate so vmap sees one shared shape
            mx0 = max(int(s[0]) for s in sh0)
            mx1 = max(int(s[1]) if len(s) > 1 else 1 for s in sh0)
            aligned: list = []
            for c in cols:
                c2 = c
                if c2.ndim == 1:
                    c2 = c2[:, None]
                if int(c2.shape[0]) < mx0:
                    c2 = jnp.pad(c2, [(0, mx0 - int(c2.shape[0]))] + [(0, 0)] * (c2.ndim - 1))
                elif int(c2.shape[0]) > mx0:
                    c2 = c2[:mx0]
                if c2.shape[-1] < mx1:
                    c2 = jnp.pad(c2, [(0, 0)] * (c2.ndim - 1) + [(0, mx1 - int(c2.shape[-1]))])
                elif c2.shape[-1] > mx1:
                    c2 = c2[..., :mx1]
                aligned.append(c2)
            stacked.append(jnp.stack(aligned, axis=0))
        axis_def = tuple(0 for _ in range(pn))
        return stacked, axis_def


    def create_out_linears_process(self):
        for shape in self.db_layer.OUT_SHAPES:
            in_features = int(jnp.prod(jnp.array(shape)))
            self.feature_encoder.create_out_linear(
                in_features,
            )





    def create_node(self, eq_struct, eq_idx) -> Node:
        runnable = create_runnable(eq_struct)
        # gien: Node.axis is only used on differential paths — first variation row is a stable default
        _ax_rows = self._variation_axis_rows_for_eq(eq_idx)
        # gien: axis row ints must be JAX-clean (no None) for vmap in_axes on differential path
        _ax0 = tuple(_ax_rows[0] if _ax_rows else self.axs_all_eqs[0])
        _clean_axis = tuple(0 if a is None else int(jnp.asarray(a).ravel()[0]) for a in _ax0)
        node = Node(
            axis=_clean_axis,
            runnable=runnable,
            amount_variations=len(self.LEN_FEATURES_PER_EQ[eq_idx]),
        )
        return node

    def calc_batch(self):
        jax.debug.print("calc_batch...")

        # calc all methods and apply result to new g
        all_features = []
        all_ins = []
        all_out_features = []
        all_outs = []

        # START LOOP
        node: Node

        for eq_idx, node in enumerate(self.METHODS):
            # gien: `ax_rows` has length V; pairs with variation index, not param index alone
            ax_rows = self._variation_axis_rows_for_eq(eq_idx)

            # get flatten params for all variations
            # # # #
            # a a
            # b b
            variations, param_len, scaled_amount_params_all_variatons = self.extract_eq_variations(
                eq_idx,
            )

            #
            variations = self.reshape_variant_block(
                variations,
                scaled_amount_params_all_variatons,
                param_len,
            )

            # gien: keep (V, P, 4); transposing to (P, V, 4) broke zip with `ax_rows` (length V) and Node arity
            transformed = self.short_transformed(
                ax_rows,
                variations,
            )

            #
            flatten_transformed = self.extract_flat_params(
                eq_idx,
                transformed,
            )

            # reshape flattened batch values (needed to get node batch size)
            inputs_nested = self.shape_input(
                eq_idx,
                flatten_transformed,
            )
            # gien: P stacked inputs for functools/vmap arity matching `runnable`
            inputs_flat, axis_def = self._stack_node_inputs(inputs_nested)

            # calc single equation
            if not inputs_flat:
                results = jnp.asarray(0.0, dtype=jnp.float32)
            else:
                results = node(
                    unprocessed_in=inputs_flat,
                    precomputed_grid=None,
                    in_axes_def=axis_def,
                    eq_idx=eq_idx,
                )

            # CREATE EMBED -> LIST[EQIDX, all_features_all_runs]
            features = self.feature_encoder.create_in_features(
                inputs_nested,
                eq_idx,
            )
            all_features.extend(features or [])

            # OUTPUT FEATURE
            out_features = self.feature_encoder.create_out_features(
                output=results,
                eq_idx=eq_idx,
            )
            all_out_features.extend(out_features or [])

            # IN RAW
            all_ins.extend(inputs_flat)

            # OUT RAW — gien: scalar ndarray is not iterable for list.extend
            if results is None:
                pass
            elif isinstance(results, jnp.ndarray) and results.ndim == 0:
                all_outs.append(results)
            elif isinstance(results, (list, tuple)):
                all_outs.extend(results)
            else:
                all_outs.append(results)

        self.db_layer.save_t_step(
            all_out_features,
            all_features,
            all_outs,
        )

        self.feature_encoder.save_features(all_features)

        jax.debug.print("calc_batch... done")





    def get_flatten_value(self, *inputs):
        # print("self.SCALED_PARAMS_CUMSUM", self.SCALED_PARAMS_CUMSUM)
        # get unscaled abs param idx
        def xtract_single(mod_idx, field_idx, rel_param_idx):
            abs_unscaled_param_idx = self.db_layer.get_rel_db_index(
                mod_idx,
                field_idx,
                rel_param_idx
            )
            return abs_unscaled_param_idx

        return vmap(xtract_single, in_axes=0)(*inputs)

    def get_scaled_idx(self, abs_unscaled_param_idx_batch):
        # get batch scaled idx from unscaled batch
        def xtract_single(abs_unscaled_param_idx):
            abs_unscaled_param_idx_and_len = self.db_layer.get_db_index(
                abs_unscaled_param_idx
            )
            return abs_unscaled_param_idx_and_len

        return vmap(xtract_single, in_axes=0)(abs_unscaled_param_idx_batch)

    def extract_flat_params(self, eq_idx, transformed):
        # gien: `transformed` is (V, P, 4) after `short_transformed`; one DB slice per (variation, param)
        print("extract_flat_params...")
        variations = jnp.asarray(transformed)
        ax_rows = self._variation_axis_rows_for_eq(eq_idx)
        if variations.ndim != 3 or variations.shape[-1] != 4:
            print("extract_flat_params... done")
            return []
        v_cnt, p_cnt, _ = variations.shape
        if ax_rows and len(ax_rows) != v_cnt:
            print("Warn extract_flat_params: ax_rows vs V", len(ax_rows), v_cnt)
        flatten_transformed: list = []
        for v in range(v_cnt):
            single_param_grid: list = []
            for p in range(p_cnt):
                coord = variations[v, p, :]
                single_param_grid.append(
                    self.db_layer.extract_flattened_grid(coord)
                )
            flatten_transformed.append(single_param_grid)

        print("extract_flat_params... done")
        return flatten_transformed

    def batch_rel_idx(self, batch):
        batch = jnp.reshape(jnp.ravel(batch), (-1, 4))

        def _wrapper(item):
            return self.db_layer.get_rel_db_index(
                *item
            )

        return vmap(
            _wrapper,
            in_axes=0
        )(
            batch[:, 1:]
        )

    def get_rel_db_index_batch(self, eq_idx, transformed):
        #print("get_rel_db_index_batch...")

        def _extract_coord_batch(ax, coord_batch):
            if ax == 0:
                result = self.batch_rel_idx(
                    coord_batch
                )
            else:
                flat = jnp.ravel(jnp.asarray(coord_batch))
                result = self.db_layer.get_rel_db_index(
                    *flat[-3:]
                )
            # print(">result", result)
            return result

        variations = jnp.asarray(transformed)
        ax_rows = self._variation_axis_rows_for_eq(eq_idx)
        if variations.ndim != 3:
            return []
        v_cnt, p_cnt, _ = variations.shape
        rel_idx_map: list = []
        for v in range(v_cnt):
            row: list = []
            for p in range(p_cnt):
                item = jnp.expand_dims(variations[v, p, :], 0)
                ax = ax_rows[v][p] if v < len(ax_rows) and p < len(ax_rows[v]) else 0
                ax_i = 0 if ax is None else int(jnp.asarray(ax).ravel()[0])
                row.append(
                    _extract_coord_batch(ax_i, item)
                )
            rel_idx_map.append(row)
        return rel_idx_map

    def shape_input(self, eq_idx, flatten_transformed):
        print("shape_input...")
        inputs: list = []
        try:
            for variation_shapes_wrap, variation_grids in zip(
                self.in_shapes_all_eqs[eq_idx],
                flatten_transformed,
            ):
                # gien: prep stores `[shapes]` per variation — unwrap so each param aligns with one flat grid
                if not variation_shapes_wrap:
                    print("Warn shape_input: empty shapes wrap")
                    continue
                shapes_for_var = variation_shapes_wrap[0]
                if len(shapes_for_var) != len(variation_grids):
                    print(
                        "Warn shape_input: shapes vs grids",
                        len(shapes_for_var),
                        len(variation_grids),
                    )
                # gien: pad missing param grids so every expected shape slot gets a tensor (avoids short zip / empty inputs_flat)
                _grid_row = list(variation_grids)
                while len(_grid_row) < len(shapes_for_var):
                    _grid_row.append(jnp.array([0.0], dtype=jnp.float32))
                if len(_grid_row) > len(shapes_for_var):
                    _grid_row = _grid_row[: len(shapes_for_var)]
                per_shape_row: list = []
                for shape, raw_grid in zip(shapes_for_var, _grid_row):
                    try:
                        # gien: isolate bad (shape, grid) pairs — one bad cell no longer fails the whole variation
                        _st = tuple(max(1, int(s)) for s in (shape or (1,)))
                        arr = jnp.ravel(jnp.asarray(raw_grid))
                        res = bring_flat_to_shape(arr, _st)
                        # gien: empty blocks avoid (0, -1) reshape / modulo fallout in downstream stack
                        if res.size == 0:
                            _pd = int(jnp.prod(jnp.array(_st)))
                            _pd = max(1, _pd)
                            res = jnp.zeros((1, _pd), dtype=jnp.float32)
                        else:
                            _n0 = max(1, int(res.shape[0]))
                            res = jnp.reshape(res, (_n0, -1))
                        per_shape_row.append(res)
                    except Exception as _cell_e:
                        print("Warn shape_input cell:", _cell_e)
                        continue
                inputs.append(per_shape_row)
            print("shape_input... done")
            return inputs
        except Exception as e:
            print("Err shape_input", e)
            return []



    def extract_eq_variations(self, eq_idx):
        """
        infer method variation: linear layout — block i has length V[i]*P[i], concatenated.
        """
        #jax.debug.print("extract_eq_variations ")

        v = jnp.array(self.DB_CTL_VARIATION_LEN_PER_EQUATION)
        p = jnp.array(self.METHOD_PARAM_LEN_CTLR)
        per_block = v * p
        offset_starts = jnp.concatenate([jnp.array([0]), jnp.cumsum(per_block)])

        offset_start = int(offset_starts[eq_idx])
        offset_len = int(per_block[eq_idx])


        edges = jax.lax.dynamic_slice_in_dim(
            jnp.array(self.DB_TO_METHOD_EDGES),
            offset_start,
            offset_len,
        )
        amount_params = int(p[eq_idx])
        total_rows = offset_len
        return edges, amount_params, total_rows


    def set_shift(self, start_pos: list[tuple] = None):
        """
        Calculates neighboring node indices based on SHIFT_DIRS using pure-Python coordinate addition
        to avoid JAX type issues during initialization.
        """
        if start_pos is None:
            # If no starting positions are provided, we initialize for all nodes in the grid
            start_pos = self.schema_grid

        next_index_map = []
        for pos in start_pos:
            # SHIFT_DIRS[0] + SHIFT_DIRS[1] combines positive and negative directions
            for d in (SHIFT_DIRS[0] + SHIFT_DIRS[1]):
                # Pure python addition: tuple zip sum
                neighbor_pos = tuple(a + b for a, b in zip(pos, d))

                if neighbor_pos in self.schema_grid:
                    next_index_map.append(
                        self.schema_grid.index(neighbor_pos)
                    )
            # include the node itself
            next_index_map.append(
                self.schema_grid.index(pos)
            )
        return next_index_map

    def surrounding_check_with_node(self, node, values_flat, start_pos=None):
        if start_pos is None:
            start_pos = self.schema_grid
        all_dirs = SHIFT_DIRS[0] + SHIFT_DIRS[1]
        mat = []
        for pos in start_pos:
            center_idx = self.schema_grid.index(pos)
            row = []
            for d in all_dirs:
                neighbor_pos = tuple(a + b for a, b in zip(pos, d))
                idx = self.schema_grid.index(neighbor_pos) if neighbor_pos in self.schema_grid else center_idx
                row.append(idx)
            row.append(center_idx)
            mat.append(row)
        neighbor_index_map = jnp.array(mat)
        return node.surrounding_check(values_flat, neighbor_index_map)

    def get_index(self, pos: tuple) -> int:
        """Returns the integer index of a grid position."""
        return self.schema_grid.index(pos)


    def create_feature_rows(self, eq_idx, variations):
        #

        def _process(variation, row_idx):
            flatten_params = jax.tree_util.tree_map(
                lambda x: self.db_layer.extract_flattened_grid(x),
                variation
            )

            linears = self.feature_encoder.get_linear_row(
                eq_idx,
                row_idx,
            )

            feature_rows = jax.tree_util.tree_map(
                lambda p, l: self.feature_encoder.gen_feature(p, l),
                flatten_params,
                linears,
            )

            return feature_rows

        try:
            kernel = jax.vmap(
                _process,
                in_axes=(0, 0),
            )

            idx_map = jnp.arange(len(variations))

            feature_rows = kernel(
                variations,
                idx_map
            )

            # save features
            self.feature_encoder._save_in_feature_method_grid(
                feature_rows,
                eq_idx
            )

            print("create_feature_rows... done")
            return feature_rows
        except Exception as e:
            print("Err create_feature_rows", e)


"""

    def set_next_nodes(self, pos_module_field_node):
        for module in pos_module_field_node:
            for field_pos_list in module:
                field_pos_list = self.set_shift(field_pos_list)
        return pos_module_field_node


    def get_sdr_rcvr(self):
        receiver = []
        sender = []
        for i, item in enumerate(range(len(self.store))):
            for j, eitem in enumerate(self.edges):
                sender.extend([i for _ in range(len(self.edges))])
                receiver.extend(eitem)
        jax.debug.print("set direct interactions finsihed")
        return sender, receiver


    def sort_features(self, eq_idx, all_features):
        len_eq_variations = self.ITERATORS["eq_variations"][eq_idx]
        start_idx = jnp.sum(self.ITERATORS["eq_variations"])[:eq_idx]

        indices = jnp.array(start_idx+i for i in range(len_eq_variations))
        # FEATURE -> MODEL
        self.model_skeleton.at[
            tuple(indices.T)
        ].add(all_features)
def _workflow(self):
    model = [
        # INJECTIOIN -> to get directly inj pattern (todo switch throguh db mapping)
        # DB SCHEMA
        self.db_layer.db_pattern,

        # EDGE DB -> METHOD
        self.method_struct,

        # FEATURES
        self.model_skeleton,

        # FEATURE DB
        self.def_out_db
    ]

    return model


# generate model tstep (shapes must match [*inputs, outputs] → 20 items)
        def _shape_tree(x):
            if isinstance(x, (list, tuple)):
                return [_shape_tree(e) for e in x]
            return getattr(x, "shape", x)
        out_shapes = _shape_tree(all_results)

"""

"""
Ziel: 
    # backgorund: db_layer.out_shape include shapes for each
    # out variation per field
    # get feature starting point
    feature_out_shapes_start = self.LEN_FEATURES_PER_EQ_CUMSUM[eq_idx]
    feature_out_shapes_len = self.LEN_FEATURES_PER_EQ[eq_idx]

    # pick slice
    _out_shapes_slice = jax.lax.dynamic_slice_in_dim(
        self.db_layer.OUT_SHAPES,
        feature_out_shapes_start,
        feature_out_shapes_len,
    )


    amount_feature_blocks_eq = len(self.LEN_FEATURES_PER_EQ[eq_idx])

    # group extracted shape slice into LEN_FEATURES_PER_EQ item
    group_ids = jnp.repeat(
        jnp.arange(amount_feature_blocks_eq),
        self.LEN_FEATURES_PER_EQ[eq_idx]
    )

    #
    grouped_shapes = ops.segment_sum(_out_shapes_slice, group_ids)

    ## out = grouped features based on len_f_per_eq / block
    out_linears = []


    #
    for shapes_struct in grouped_shapes:
        for single_shape in shapes_struct:
            param_len = self.get_unscaled_param_len

        out_linears.extend(
            [ # woher erhalten wir len of param ->
                # PARAM_CTLR ->
                # wie ehralten rel db id?
                nnx.Linear(
                    in_features=,
                    out_features=self.d_model,
                    rngs=self.rngs
                )
            ]
        )


    def handle_in_features(self, grids):
        # gen in featrues

        def _handle_single_grid_features(grid, in_ax_def, t=0):
            in_features = self.feature_encoder.create_features(
                inputs=grid,
                axis_def=in_ax_def,
                time=t,
                param_idx=0
            )
            return in_features

        features = jax.tree_util.tree_map(
            _handle_single_grid_features,
            grids,
            self.axs_all_eqs
        )
"""

"""#
feature_rows = self.create_feature_rows(
    eq_idx,
    variations
)

# 
_blur = self.feature_encoder.get_precomputed_results(
    feature_rows,
    axis_def,
    eq_idx,



    def build_projections(self):
        self.projection_shapes = []

        # extend
        for i in range(len(self.in_shapes_all_eqs)):
            self.projection_shapes.append(
                [
                    *self.in_shapes_all_eqs[i],
                    self.db_layer.OUT_SHAPES[i]
                ]
            )

        return [
            [
                nnx.Linear(
                    in_features=shape,
                    out_features=self.model_feature_dims,
                    rngs=nnx.Rngs
                )
                for shape in shapes
            ]
           for shapes in self.projection_shapes
        ]


)"""


def extract_eq_variations(self, eq_idx):
    """
    infer metho variation start + len = end
    """
    #jax.debug.print("extract_eq_variations ")

    #
    offsets_variaitons_start = jnp.concatenate([
        jnp.array([0]),
        jnp.cumsum(jnp.array(self.DB_CTL_VARIATION_LEN_PER_EQUATION))
    ])[eq_idx]

    #
    offsets_param_start = jnp.concatenate([
        jnp.array([0]),
        jnp.cumsum(jnp.array(self.METHOD_PARAM_LEN_CTLR))
    ])[eq_idx]

    #
    offset_start = offsets_variaitons_start * offsets_param_start
    #print("offsets", offset_start)

    #
    offsets_variaitons_end = jnp.cumsum(jnp.array(self.DB_CTL_VARIATION_LEN_PER_EQUATION))[eq_idx]
    offsets_param_end = jnp.cumsum(jnp.array(self.METHOD_PARAM_LEN_CTLR))[eq_idx]
    offset_end = offsets_variaitons_end * offsets_param_end

    #
    offset_len = int(offset_end - offset_start)
    print("offset_len", offset_len)

    #
    edges = jax.lax.dynamic_slice_in_dim(
        jnp.array(self.DB_TO_METHOD_EDGES),
        offset_start,
        offset_len,
    )
    print("extract_eq_variations... done")
    return edges, offset_len


"""
high_score_elements = self.feature_encoder.get_precomputed_results(
    features_tree,
    axis_def,
    #eq_idx,
)

# Node vmap batch size = size of mapped axis; use max over batched (axis-0) inputs.
batch_size = 0
if inputs and axis_def is not None:
    for i, inp in enumerate(inputs):
        if hasattr(inp, "shape") and inp.shape and (i >= len(axis_def) or axis_def[i] == 0):
            batch_size = max(batch_size, int(inp.shape[0]))
if batch_size == 0 and inputs:
    batch_size = int(inputs[0].shape[0]) if hasattr(inputs[0], "shape") else 0

if not high_score_elements or batch_size == 0:
    _blur = jnp.full((max(1, batch_size), self.feature_encoder.d_model), jnp.nan)
else:
    _blur = self.feature_encoder.blur_result_from_in_tree(
        eq_idx,
        high_score_elements,
        batch_size,
    )
### ###
"""

"""
# pre-shorten grids along axes
transformed = self.short_transformed(
    axs,
    transformed,
)
"""


