from data import GAUGE_FIELDS
from sm_manager.sm.gauge.gauge_utils import GaugeUtils

class GaugeCreator(GaugeUtils):
    def __init__(self, g_utils):
        """
        gu: Utility-Objekt mit init_G, init_fmunu, etc.
        graph: Dein Graph-Objekt mit add_node, add_edge
        layer: Aktueller Layer-Name
        gauge_fields: Dict der Gauge-Felder (GAUGE_FIELDS)
        """
        super().__init__()
        self.g = g_utils
        self.qfn_layer = "PIXEL"
        self.gluon_item_type = "GLUON"

    def get_gauge_params(
            self,
            ntype,
            pos,
            px_id: str,
            light=None,
            id=None,
    ) -> list:

        """
        Retrieve gauge params for the GaugeCreator workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `ntype`, `pos`, `px_id`, `light`.
        2. Builds intermediate state such as `field_key`, `attrs_struct`, `attrs` before applying the main logic.
        3. Branches on validation or runtime state to choose the next workflow path.
        4. Delegates side effects or helper work through `self._field_value()`, `ntype.lower()`, `range()`.
        5. Returns the assembled result to the caller.

        Inputs:
        - `ntype`: Caller-supplied value used during processing.
        - `pos`: Caller-supplied value used during processing.
        - `px_id`: Identifier used to target the relevant entity. Expected type: `str`.
        - `light`: Caller-supplied value used during processing.
        - `id`: Identifier used to target the relevant entity.

        Returns:
        - Returns `list` to the caller.
        """
        field_key = self._field_value(ntype)

        try:
            attrs_struct = []
            if "gluon" == ntype.lower():
                for item_index in range(8):
                    if nid is None:
                        nid = f"{ntype}__{px_id}__{item_index}"
                    else:
                        item_index = int(nid.split("_")[-1])

                    attrs = self.get_g_params_core(
                        pos,
                        nid,
                        ntype,
                        field_key,
                        item_index,
                        light
                    )

                    self.check_extend_attrs(
                        ntype,
                        attrs
                    )

                    attrs_struct.append(attrs)
                    nid = None
            else:
                if id is None:
                    id = f"{ntype}__{px_id}"

                attrs = self.get_g_params_core(
                    pos,
                    id,
                    ntype,
                    field_key,
                    item_index=None
                )

                self.check_extend_attrs(
                    ntype,
                    attrs,
                )

                attrs_struct.append(attrs)
            return attrs_struct
        except Exception as e:
            print(f"Err cor.qbrain.cor.sm_manager.sm.gauge.g_creator::GaugeCreator.get_gauge_params | handler_line=92 | {type(e).__name__}: {e}")
            print(f"[exception] cor.qbrain.cor.sm_manager.sm.gauge.g_creator.GaugeCreator.get_gauge_params: {e}")
            print(f"Err get_gauge_params: {e}")


    def check_extend_attrs(
            self,
            ntype,
            attrs,
    ):
        """
        Check extend attrs for the GaugeCreator workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `ntype`, `attrs`.
        2. Branches on validation or runtime state to choose the next workflow path.
        3. Delegates side effects or helper work through `ntype.lower()`, `attrs['parent'].append()`, `ntype.upper()`.
        4. Finishes by updating state, triggering side effects, or completing the workflow without a direct return value.

        Inputs:
        - `ntype`: Caller-supplied value used during processing.
        - `attrs`: Caller-supplied value used during processing.

        Returns:
        - Returns `None`; the main effects happen through state updates, I/O, or delegated calls.
        """
        if ntype.lower() == "gluon":
            attrs["parent"].append(ntype.upper())

    def get_g_params_core(
            self,
            pos,
            nid,
            ntype,
            field_key,
            item_index,
    ):
        """
        Retrieve g params cor for the GaugeCreator workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `pos`, `nid`, `ntype`, `field_key`.
        2. Builds intermediate state such as `attrs` before applying the main logic.
        3. Delegates side effects or helper work through `dict()`, `self.gfield()`.
        4. Returns the assembled result to the caller.

        Inputs:
        - `pos`: Caller-supplied value used during processing.
        - `nid`: Caller-supplied value used during processing.
        - `ntype`: Caller-supplied value used during processing.
        - `field_key`: Caller-supplied value used during processing.
        - `item_index`: Caller-supplied value used during processing.

        Returns:
        - Returns the computed result for the caller.
        """
        attrs = dict(
            id=nid,
            tid=0,
            parent=["GAUGE"],
            type=ntype,
            field_key=field_key,
            **self.gfield(pos, item_index),
        )
        return attrs

    def gfield(
            self,
            ntype,
            dim=1
    ):
        """
        Gfield workflow state for the GaugeCreator workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `ntype`, `dim`.
        2. Builds intermediate state such as `field_value`, `const`, `field` before applying the main logic.
        3. Delegates side effects or helper work through `self.field_value()`, `self.dmu()`, `self.fmunu()`.
        4. Returns the assembled result to the caller.

        Inputs:
        - `ntype`: Caller-supplied value used during processing.
        - `dim`: Caller-supplied value used during processing.

        Returns:
        - Returns the computed result for the caller.
        """
        field_value=self.field_value(dim=dim)

        const = {
            k: v
            for k, v in GAUGE_FIELDS[ntype].items()
        }
        # uniform format
        field = {
            "gg_coupling": field_value,
            "gf_coupling": field_value,
            "field_value": field_value,
            "prev_field_value": field_value,

            "j_nu": field_value,

            "dmuG": self.dmu(dim),
            "fmunu": self.fmunu(dim),
            "prev_fmunu": self.fmunu(dim),
            "dmu_fmunu": self.dmu_fmunu(dim),
            **{k:[v] for k, v in const.items()},
        }

        return field


    def get_gluon(self, pos):
        """
        Retrieve gluon for the GaugeCreator workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `pos`.
        2. Builds intermediate state such as `field_value` before applying the main logic.
        3. Delegates side effects or helper work through `self.gluon_fieldv()`, `dict()`, `self.dmu()`.
        4. Returns the assembled result to the caller.

        Inputs:
        - `pos`: Caller-supplied value used during processing.

        Returns:
        - Returns the computed result for the caller.
        """
        field_value=self.gluon_fieldv(pos)

        return dict(
            gg_coupling=0,
            gf_coupling=0,
            field_value=field_value,
            prev=0,
            j_nu=0,
            dmuG=self.dmu(pos),
            fmunu=self.fmunu(pos),
            prev_fmunu=self.fmunu(pos),
            dmu_fmunu=self.dmu_fmunu(pos),
            charge=0,
            mass=0.0,
            g=1.217,
            spin=1,
        )


    def create(self, src_qfn_id):
        """
        Create workflow state for the GaugeCreator workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `src_qfn_id`.
        2. Builds intermediate state such as `g_field` before applying the main logic.
        3. Branches on validation or runtime state to choose the next workflow path.
        4. Delegates side effects or helper work through `GAUGE_FIELDS.items()`, `g_field.upper()`, `print()`.
        5. Finishes by updating state, triggering side effects, or completing the workflow without a direct return value.

        Inputs:
        - `src_qfn_id`: Identifier used to target the relevant entity.

        Returns:
        - Returns `None`; the main effects happen through state updates, I/O, or delegated calls.
        """
        for g_field, gattrs in GAUGE_FIELDS.items():
            g_field = g_field.upper()

            if g_field.lower() == "gluon":
                self._create_gluon_items(
                    g_field,
                    gattrs,
                    src_qfn_id,
                )
            else:
                self._create_gauge(
                    g_field,
                    src_qfn_id,
                    gattrs
                )

            print(f"{g_field} for {src_qfn_id} created")


    def connect_intern_fields(self, pixel_id):
        """
        Connect intern fields for the GaugeCreator workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `pixel_id`.
        2. Branches on validation or runtime state to choose the next workflow path.
        3. Delegates side effects or helper work through `self.gauge_to_gauge_couplings.items()`, `print()`, `src_field.lower()`.
        4. Finishes by updating state, triggering side effects, or completing the workflow without a direct return value.

        Inputs:
        - `pixel_id`: Identifier used to target the relevant entity.

        Returns:
        - Returns `None`; the main effects happen through state updates, I/O, or delegated calls.
        """
        for src_field, trgt_fields in self.gauge_to_gauge_couplings.items():
            if src_field.lower() != "gluon": # -> alread connected in gluon process
                src_id, src_attrs = self.g.get_single_neighbor_nx(pixel_id, src_field.upper())

                for trgt_field in trgt_fields:
                    trgt_id, trgt_attrs = self.g.get_single_neighbor_nx(
                        pixel_id,
                        trgt_field.upper()
                    )
                    self.g.add_edge(
                        src=src_id,
                        trgt=trgt_id,
                        attrs=dict(
                            rel="intern_coupled",
                            src_layer=src_field.upper(),
                            trgt_layer=trgt_field.upper(),
                        )
                    )
        print("local gauges connected")



    def connect_gluons(self, nid):
        """
        Get all neighbors of a single gluon ->
        get their GLUON_ITEMs ->
        connect them all to nid
        """
        # Get all Gluon neighbors
        g_item = "GLUON"
        all_gluon_neighbor_items = self.get_gluon_neighbor_items(
            nid,
            trgt_type=g_item
        )

        # connect nid to all of them
        for ngluon_sub_id, snattrs in all_gluon_neighbor_items:
            self.g.add_edge(
                nid,
                ngluon_sub_id,
                attrs=dict(
                    rel="uses_param",
                    src_layer=g_item,
                    trgt_layer=g_item,
                )
            )



    def get_gluon_neighbor_items(self, nid, trgt_type):
        """
        Receive all possible gluon item cons for
        a single item
        intern & extern
        """
        all_gluon_neighbor_items = []
        # EXTERN
        gluon_neighbors = self.g.get_neighbor_list(
            nid,
            "GLUON"
        )

        for ngluon_id, _ in gluon_neighbors:
            # Get neighbors sub-gluon-fields
            gluon_neighbors_subs = self.g.get_neighbor_list(
                ngluon_id,
                trgt_type
            )
            all_gluon_neighbor_items.extend(gluon_neighbors_subs)

        return all_gluon_neighbor_items




    def _create_gluon_items(
        self,
        g_field,
        gattrs,
        src_qfn_id,
        pos,
        ntype,
    ):

        """
        Create gluon items for the GaugeCreator workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `g_field`, `gattrs`, `src_qfn_id`, `pos`.
        2. Builds intermediate state such as `all_gluon_ids`, `gauge_id`, `attrs` before applying the main logic.
        3. Branches on validation or runtime state to choose the next workflow path.
        4. Delegates side effects or helper work through `set()`, `range()`, `self.g.print_edges()`.
        5. Finishes by updating state, triggering side effects, or completing the workflow without a direct return value.

        Inputs:
        - `g_field`: Caller-supplied value used during processing.
        - `gattrs`: Caller-supplied value used during processing.
        - `src_qfn_id`: Identifier used to target the relevant entity.
        - `pos`: Caller-supplied value used during processing.
        - `ntype`: Caller-supplied value used during processing.

        Returns:
        - Returns `None`; the main effects happen through state updates, I/O, or delegated calls.
        """
        all_gluon_ids = set()
        for i in range(8):
            gauge_id = f"{g_field}_{i}_{src_qfn_id}"
            all_gluon_ids.add(gauge_id)

            attrs = self.get_gauge_params(
                pos=pos,
                id=gauge_id,
                px_id=gattrs[src_qfn_id],
                ntype=ntype
            )
            self.g.add_node(
                attrs=attrs
            )


        # Connect intern each gluon
        for gluon_id in all_gluon_ids:
            for trgt_gluon_id in all_gluon_ids:
                if gluon_id != trgt_gluon_id:
                    self.g.add_edge(
                        src=gluon_id,
                        trgt=trgt_gluon_id,
                        attrs=dict(
                            rel="intern_gluon",
                            src_layer=self.gluon_item_type,
                            trgt_layer=self.gluon_item_type,
                        )
                    )
                    print(f"Connect {gluon_id} -> {trgt_gluon_id}, {self.g.G.has_edge(gluon_id, trgt_gluon_id)}")

        self.g.print_edges("GLUON", "GLUON")
        print("Gluon items created")

    def _create_gauge(
        self,
        g_field,
        src_qfn_id,
        attrs
    ):

        """
        Create gauge for the GaugeCreator workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `g_field`, `src_qfn_id`, `attrs`.
        2. Builds intermediate state such as `gauge_id`, `parent_attrs` before applying the main logic.
        3. Delegates side effects or helper work through `self.get_gauge_params()`, `self.g.add_node()`.
        4. Returns the assembled result to the caller.

        Inputs:
        - `g_field`: Caller-supplied value used during processing.
        - `src_qfn_id`: Identifier used to target the relevant entity.
        - `attrs`: Caller-supplied value used during processing.

        Returns:
        - Returns the computed result for the caller.
        """
        gauge_id = f"{g_field}_{src_qfn_id}"

        parent_attrs = self.get_gauge_params(
            id=gauge_id,
            ntype=g_field,
            gluon_index=None,
            gattrs=attrs,
        )

        self.g.add_node(attrs=parent_attrs)

        return parent_attrs


    def _connect_2_qfn(self, src_qfn_id, gauge_id, g_field):
        # PIXEL -> Parent Edge
        """
        Connect 2 qfn for the GaugeCreator workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `src_qfn_id`, `gauge_id`, `g_field`.
        2. Delegates side effects or helper work through `self.g.add_edge()`, `dict()`.
        3. Finishes by updating state, triggering side effects, or completing the workflow without a direct return value.

        Inputs:
        - `src_qfn_id`: Identifier used to target the relevant entity.
        - `gauge_id`: Identifier used to target the relevant entity.
        - `g_field`: Caller-supplied value used during processing.

        Returns:
        - Returns `None`; the main effects happen through state updates, I/O, or delegated calls.
        """
        self.g.add_edge(
            src_qfn_id,
            gauge_id,
            attrs=dict(
                rel="has_field",
                src_layer=self.qfn_layer,
                trgt_layer=g_field,
            )
        )
