from sm_manager.sm.higgs.phi_utils import HiggsUtils

class HiggsCreator(HiggsUtils):
    """
    A minimalistic class for creating a simplified Higgs-like field node
    and connecting it within a graph structure.
    """

    def __init__(self, g):
        """
        Initialize the HiggsCreator instance state.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `g`.
        2. Delegates side effects or helper work through `HiggsUtils.__init__()`.
        3. Finishes by updating state, triggering side effects, or completing the workflow without a direct return value.

        Inputs:
        - `g`: Graph instance that the workflow reads from or mutates.

        Returns:
        - Returns `None`; the main effects happen through state updates, I/O, or delegated calls.
        """
        HiggsUtils.__init__(self)
        self.g=g

    def higgs_attrs(self, px_id, id=None) -> list[dict]:
        """
        Creates a Higgs field node (PHI) and connects it to a source node.
        """
        try:
            if id is  None:
                id = f"PHI__{px_id}"

            node_attrs = dict(
                id=id,
                tid=0,
                type="HIGGS",
                px=px_id,
                parent=["HIGGS"],
            )

            return [node_attrs]
        except Exception as e:
            print(f"Err core.qbrain.core.sm_manager.sm.higgs.higgs_creator::HiggsCreator.higgs_attrs | handler_line=44 | {type(e).__name__}: {e}")
            print(f"[exception] core.qbrain.core.sm_manager.sm.higgs.higgs_creator.HiggsCreator.higgs_attrs: {e}")
            print(f"Err higgs_attrs: {e}")


    def higgs_params_batch(self, dim, just_vals=False, just_k=False) -> dict or list:
        #phi = self.init_phi(h)
        """
        Higgs params batch for the HiggsCreator workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `dim`, `just_vals`, `just_k`.
        2. Builds intermediate state such as `field_value`, `field` before applying the main logic.
        3. Branches on validation or runtime state to choose the next workflow path.
        4. Delegates side effects or helper work through `self.field_value()`, `self.dmu()`, `list()`.
        5. Returns the assembled result to the caller.

        Inputs:
        - `dim`: Caller-supplied value used during processing.
        - `just_vals`: Caller-supplied value used during processing.
        - `just_k`: Caller-supplied value used during processing.

        Returns:
        - Returns `dict or list` to the caller.
        """
        try:
            field_value = self.field_value(dim=dim)
            field = {
                # --- Arrays scaled by N (Structure of Arrays - SOA) ---
                "phi": field_value,
                #"prev": np.zeros_like(phi_val),  # Previous timestep state
                "dmu_h": self.dmu(dim),
                "h": field_value,  # The physical scalar field component
                "dV_dh": field_value,  # Potential derivative
                "laplacian_h": field_value,
                "vev": [246.0],  # Vacuum Expectation Value (VEV)
                "energy": [0],  # Energy contribution per node
                "energy_density": field_value,

                # --- Scalars (Constants/System Aggregates) ---
                "potential_energy_H": field_value,
                "total_energy_H": field_value,  # Total system energy (System Scalar)
                "mass": [125.0],  # Higgs mass constant
                "lambda_H": [0.13] # Coupling constant
            }

            if just_vals:
                return list(field.values())
            elif just_k:
                return list(field.keys())
            else:
                return field

        except Exception as e:
            print(f"Err core.qbrain.core.sm_manager.sm.higgs.higgs_creator::HiggsCreator.higgs_params_batch | handler_line=97 | {type(e).__name__}: {e}")
            print(f"[exception] core.qbrain.core.sm_manager.sm.higgs.higgs_creator.HiggsCreator.higgs_params_batch: {e}")
            print("Err higgs_params_batchs", e)

    def create_higgs_field(self, px_id):
        """
        Create higgs field for the HiggsCreator workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `px_id`.
        2. Builds intermediate state such as `attrs` before applying the main logic.
        3. Delegates side effects or helper work through `self.higgs_attrs()`, `self.g.add_node()`, `self.g.add_edge()`.
        4. Finishes by updating state, triggering side effects, or completing the workflow without a direct return value.

        Inputs:
        - `px_id`: Identifier used to target the relevant entity.

        Returns:
        - Returns `None`; the main effects happen through state updates, I/O, or delegated calls.
        """
        attrs = self.higgs_attrs(
            px_id
        )

        self.g.add_node(attrs)

        # Connect PHI to PIXEL
        self.g.add_edge(
            px_id,
            attrs["id"],
            attrs=dict(
                rel=f"has_field",
                src_layer="PIXEL",
                trgt_layer="PHI",
            )
        )

