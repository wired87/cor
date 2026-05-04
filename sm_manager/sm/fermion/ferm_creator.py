from data import FERM_PARAMS
from sm_manager.sm.fermion.ferm_utils import FermUtils



class FermCreator(FermUtils):

    def __init__(self, g):
        """
        Initialize the FermCreator instance state.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `g`.
        2. Delegates side effects or helper work through `super().__init__()`, `super()`.
        3. Finishes by updating state, triggering side effects, or completing the workflow without a direct return value.

        Inputs:
        - `g`: Graph instance that the workflow reads from or mutates.

        Returns:
        - Returns `None`; the main effects happen through state updates, I/O, or delegated calls.
        """
        super().__init__()
        self.g=g
        self.layer = "PSI"
        self.parent=["FERMION", self.layer]

    def create_ferm_attrs(
            self,
            ntype,
            px_id,
            pos,
            light=False,
            id=None,
    ) -> list:
        """
        Create ferm attrs for the FermCreator workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `ntype`, `px_id`, `pos`, `light`.
        2. Builds intermediate state such as `attrs_struct`, `nid`, `fermid` before applying the main logic.
        3. Branches on validation or runtime state to choose the next workflow path.
        4. Delegates side effects or helper work through `ntype.lower()`, `range()`, `attrs_struct.append()`.
        5. Returns the assembled result to the caller.

        Inputs:
        - `ntype`: Caller-supplied value used during processing.
        - `px_id`: Identifier used to target the relevant entity.
        - `pos`: Caller-supplied value used during processing.
        - `light`: Caller-supplied value used during processing.
        - `id`: Identifier used to target the relevant entity.

        Returns:
        - Returns `list` to the caller.
        """
        attrs_struct = []
        try:
            if "quark" in ntype.lower():
                for item_index in range(3):
                    if nid is None:
                        nid = f"{ntype}__{px_id}__{item_index}"
                    else:
                        item_index = int(nid.split("_")[-1])

                    attrs_struct.append(
                        self.get_attrs_core(
                            px_id,
                            nid,
                            ntype,
                            pos,
                            item_index,
                        )
                    )
                    id=None
            else:
                fermid = f"{ntype}__{px_id}"
                attrs_struct.append(
                    self.get_attrs_core(
                        px_id,
                        fermid,
                        ntype,
                        pos,
                    )
                )
            nid = None
        except Exception as e:
            print(f"Err cor.qbrain.cor.sm_manager.sm.fermion.ferm_creator::FermCreator.create_ferm_attrs | handler_line=86 | {type(e).__name__}: {e}")
            print(f"[exception] cor.qbrain.cor.sm_manager.sm.fermion.ferm_creator.FermCreator.create_ferm_attrs: {e}")
            print(f"Err create_ferm_attrs: {e}")
        return attrs_struct


    def get_attrs_core(
            self,
            px_id,
            nid,
            ntype,
    ):
        """

        id=nid,
        tid=0,
        gterm=0,
        yterm=0,
        parent=self.parent,
        type=ntype,

        """
        # todo parallalize array creation for all F and px
        attr_struct = dict(
            id=nid,
            tid=0,
            type=ntype,
            parent=self.parent,
            px=px_id,
        )
        return attr_struct

    def create_f_core(self, pos, item, just_v=False, just_k=False):
        # just get shape of the structure
        """
        Create f cor for the FermCreator workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `pos`, `item`, `just_v`, `just_k`.
        2. Builds intermediate state such as `psi`, `field` before applying the main logic.
        3. Branches on validation or runtime state to choose the next workflow path.
        4. Delegates side effects or helper work through `self.field_value()`, `dict()`, `field.values()`.
        5. Returns the assembled result to the caller.

        Inputs:
        - `pos`: Caller-supplied value used during processing.
        - `item`: Caller-supplied value used during processing.
        - `just_v`: Caller-supplied value used during processing.
        - `just_k`: Caller-supplied value used during processing.

        Returns:
        - Returns the computed result for the caller.
        """
        psi = self.field_value(pos)
        field = dict(
            gterm=0,
            yterm=0,
            gf_coupling=psi,
            gg_coupling=psi,
            dmu_psi=self.dmu(pos),
            psi=psi,
            dirac=psi,
            psi_bar=psi,
            prev=psi,
            #quark_index=item_index,
            velocity=0.0,
            **item,
        )
        if just_v:
            return field.values()
        if just_k:
            return field.keys()
        return field

    def create_f_core_batch(
            self,
            ntype,
            dim:int = 1,
            just_v=False,
            just_k=False,
    ):
        """
        Create f cor batch for the FermCreator workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `ntype`, `dim`, `just_v`, `just_k`.
        2. Builds intermediate state such as `psi`, `item`, `field` before applying the main logic.
        3. Branches on validation or runtime state to choose the next workflow path.
        4. Delegates side effects or helper work through `self.field_value()`, `dict()`, `field.values()`.
        5. Returns the assembled result to the caller.

        Inputs:
        - `ntype`: Caller-supplied value used during processing.
        - `dim`: Caller-supplied value used during processing. Expected type: `int`.
        - `just_v`: Caller-supplied value used during processing.
        - `just_k`: Caller-supplied value used during processing.

        Returns:
        - Returns the computed result for the caller.
        """
        try:
            psi = self.field_value(dim=dim)
            item = FERM_PARAMS[
                ntype
            ]

            field = dict(
                gterm=psi,
                yterm=psi,
                gf_coupling=psi,
                gg_coupling=psi,
                dmu_psi=self.dmu(dim), # todo do not save diff
                psi=psi,
                dirac=psi,
                psi_bar=psi,
                prev=psi,
                velocity=psi,
                # wrap as list for later reshape to arr
                **{k:[v] for k,v in item.items()},
            )

            if just_v:
                return field.values()

            if just_k:
                return field.keys()

            return field
        except Exception as e:
            print(f"Err cor.qbrain.cor.sm_manager.sm.fermion.ferm_creator::FermCreator.create_f_core_batch | handler_line=214 | {type(e).__name__}: {e}")
            print(f"[exception] cor.qbrain.cor.sm_manager.sm.fermion.ferm_creator.FermCreator.create_f_core_batch: {e}")
            print(f"Err create_f_core_batch: {e}")






    def create_quark(self, pos, item, just_v=False):
        """
        Create quark for the FermCreator workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `pos`, `item`, `just_v`.
        2. Builds intermediate state such as `psi`, `field` before applying the main logic.
        3. Branches on validation or runtime state to choose the next workflow path.
        4. Delegates side effects or helper work through `self.quark_field()`, `dict()`, `field.values()`.
        5. Returns the assembled result to the caller.

        Inputs:
        - `pos`: Caller-supplied value used during processing.
        - `item`: Caller-supplied value used during processing.
        - `just_v`: Caller-supplied value used during processing.

        Returns:
        - Returns the computed result for the caller.
        """
        psi = self.quark_field(pos)
        field = dict(
            gterm=0,
            yterm=0,
            gf_coupling=psi,
            gg_coupling=psi,
            dmu_psi=self.dmu(pos),
            psi=psi,
            dirac=psi,
            psi_bar=psi,
            prev=psi,
            velocity=0.0,
            **item,
        )
        if just_v:
            return field.values()
        return field



    def create(self, src_qfn_id):
        # PSI
        """
        Create workflow state for the FermCreator workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `src_qfn_id`.
        2. Builds intermediate state such as `ferm_field` before applying the main logic.
        3. Iterates through the available items and applies the same workflow rules consistently.
        4. Delegates side effects or helper work through `FERM_PARAMS.items()`, `self.connect_quark_doublets()`, `print()`.
        5. Finishes by updating state, triggering side effects, or completing the workflow without a direct return value.

        Inputs:
        - `src_qfn_id`: Identifier used to target the relevant entity.

        Returns:
        - Returns `None`; the main effects happen through state updates, I/O, or delegated calls.
        """
        for ferm_field, attrs in FERM_PARAMS.items():
            print(f"Create {ferm_field} for {src_qfn_id}")
            ferm_field = ferm_field.upper()
            self._create_quark_parent(
                ferm_field,
                src_qfn_id,
                attrs
            )
        self.connect_quark_doublets(src_qfn_id)
        print("Fermions created and Quarks connected")

    def connect_quark_doublets(self, src_qfn_id):
        """
        # die drei quark-paare (up+down, charm+strange, top+bottom) sind eigenständige felder
        # → jedes paar bildet ein separates SU(2)_L-dublett (auch nach SSB)
        # → jedes dublett hat eigene komponenten und eigene masse

        #  immer nur EIN quark-doublet koppelt direkt an ein W⁺ oder W⁻ vertex (pro punkt im raum)
            aber es kann sein dass bottom up an w+ und cham strange an w- zur selben zeit koppelt!

        # W±-kopplung überlagert sie jedoch durch die CKM-matrix:
        # → z. B. W⁺ koppelt u → d, s, b (nicht nur d)
        # → realisiert über linearkombination: d' = V_ud * d + V_us * s + V_ub * b

        # welches dublett an ein W⁺ koppelt, wird durch die QUANTENZAHLEN UND DIE DYNAMIK des prozesses bestimmt

        #<<< kupplungs faktoren (was legt fest welches dubet an w+/- kuppelt?: >>>

        # 1. verfügbare teilchen im prozess
        #    → wenn z. B. ein top-quark erzeugt wurde, kann (t, b) koppeln

        # 2. energie des systems
        #    → schwere dubletts (z. B. (t, b)) benötigen mehr energie
        #    → bei niedriger energie sind nur (u, d), (c, s) relevant

        # 3. CKM-Matrix
        #    → legt fest, wie stark ein up-typ quark mit jedem down-typ koppelt
        #    → z. B. u → d,s,b mit gewichtung (V_ud, V_us, V_ub)

        # 4. erhaltungsgrößen
        #    → ladung, farbe, energie, impuls etc. müssen im vertex erhalten bleiben

        # zusammengefasst:
        # das universelle W⁺-Feld kann mit allen dubletts koppeln
        # welches tatsächlich koppelt, hängt von teilchenzustand und erlaubten wechselwirkungen ab
        """
        partner_map = {
            "up": "down",
            "down": "up",
            "charm": "strange",
            "strange": "charm",
            "top": "bottom",
            "bottom": "top",
        }


        # Get Partners and connect
        for p1, p2 in partner_map.items():
            src_layer = f"{p1}_quark".upper()
            trgt_layer = f"{p2}_quark".upper()

            src_id, src_attrs = self.g.get_single_neighbor_nx(src_qfn_id, src_layer)
            trgt_id, trgt_attrs = self.g.get_single_neighbor_nx(src_qfn_id, trgt_layer)

            # P1 -> P2
            self.g.add_edge(
                src_id,
                trgt_id,
                attrs=dict(
                    rel=f"doublet_partner",
                    src_layer=src_layer,
                    trgt_layer=trgt_layer,
                )
            )


    def _create_quark_parent(
            self,
            ferm_field,
            src_qfn_id,
            attrs,
    ):
        """
        Create quark parent for the FermCreator workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `ferm_field`, `src_qfn_id`, `attrs`.
        2. Builds intermediate state such as `fermid`, `parent_attrs` before applying the main logic.
        3. Delegates side effects or helper work through `print()`, `self.g.add_node()`, `self.g.add_edge()`.
        4. Returns the assembled result to the caller.

        Inputs:
        - `ferm_field`: Caller-supplied value used during processing.
        - `src_qfn_id`: Identifier used to target the relevant entity.
        - `attrs`: Caller-supplied value used during processing.

        Returns:
        - Returns the computed result for the caller.
        """
        print(f"Create parent FermField {ferm_field}")
        fermid = f"{ferm_field}_{src_qfn_id}"

        parent_attrs = None # todo

        self.g.add_node(attrs=parent_attrs)

       # PSI -> PIXEL
        self.g.add_edge(
            src_qfn_id,
            f"{fermid}",
            attrs=dict(
                rel=f"has_field",
                src_layer="PIXEL",
                trgt_layer=ferm_field,
            )
        )
        print(f"created {fermid}")
        return fermid



    def psi_x_bar(self, ferm_field):
        #  for general ferms and single quark item
        """
        Psi x bar for the FermCreator workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `ferm_field`.
        2. Builds intermediate state such as `psi`, `psi_bar` before applying the main logic.
        3. Delegates side effects or helper work through `self._init_psi()`.
        4. Returns the assembled result to the caller.

        Inputs:
        - `ferm_field`: Caller-supplied value used during processing.

        Returns:
        - Returns the computed result for the caller.
        """
        psi = self._init_psi(ntype=ferm_field)
        psi_bar = self._init_psi(ntype=ferm_field)
        return psi, psi_bar

