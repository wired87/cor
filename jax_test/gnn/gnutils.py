"""GNUtils: variation block helpers for jax_test.gnn (short_transformed aligns (V,P,4) with axis rows)."""
import jax.numpy as jnp


class GNUtils:

    #
    def short_transformed(self, ax_rows, variations):
        # gien: `variations` is (V, P, 4); `ax_rows` has length V with one axis int per param p — no (P,V) transpose
        print("short_transformed... ")
        v_arr = jnp.asarray(variations)
        if v_arr.ndim != 3 or v_arr.shape[-1] != 4:
            print("short_transformed... done")
            return v_arr
        v_arr = jnp.array(v_arr)
        v_cnt, p_cnt, _ = v_arr.shape
        # gien: legacy rule — when param slot is not batch-axis, pin every variation to row-0 coords (shared DB slice)
        for p in range(p_cnt):
            if not ax_rows:
                continue
            # gien: first variation row defines broadcast rule for this param column
            _a0 = ax_rows[0][p] if p < len(ax_rows[0]) else 0
            ax_i = 0 if _a0 is None else int(jnp.asarray(_a0).ravel()[0])
            if ax_i != 0:
                ref = v_arr[0, p, :]
                for v in range(v_cnt):
                    v_arr = v_arr.at[v, p, :].set(ref)
        print("short_transformed... done")
        return v_arr

    #
    def serialize(self, data):
        import flax.serialization
        # Wandelt den State in einen Byte-String um
        print("serialize", type(data))
        if isinstance(data, list):
            for i, item in enumerate(data):
                print(f"item {i}", type(item))
                if isinstance(item, list):
                    for j, item2 in enumerate(item):
                        print(f"item {j}", type(item2))
        binary_data = flax.serialization.to_bytes(data)
        return binary_data