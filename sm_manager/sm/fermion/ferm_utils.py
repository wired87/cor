import numpy as np
from jax import lax

from qfu.field_utils import FieldUtils


class FermGCoupler(FieldUtils):
    """
    Calcs couplings between fermions and X
    Couplings Ferm -> G
    """

    def __init__(self):
        # save constants and terms both in edges
        # Coupling Constants between F and G
        """
        Initialize the FermGCoupler instance state.

        Workflow:
        1. Starts from the current object state and local workflow context.
        2. Delegates side effects or helper work through `super().__init__()`, `super()`.
        3. Finishes by updating state, triggering side effects, or completing the workflow without a direct return value.

        Inputs:
        - None.

        Returns:
        - Returns `None`; the main effects happen through state updates, I/O, or delegated calls.
        """
        super().__init__()
        self.coupling_constants = {
            "photon": self.photon_coupling,
            "z_boson": self.z_boson_coupling,
            "w_boson": self.w_boson_coupling,
            "gluon": self.gluon_coupling,
            "higgs": self.higgs_yukawa_coupling,
        }


    # zwischen fermion und gauge feeldern ist immer "g" als kupplungskonstante aktiv

    def photon_coupling(self, g: float, theta_W: float) -> float:
        # e = g * sin(theta_W)
        """
        Photon coupling for the FermGCoupler workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `g`, `theta_W`.
        2. Delegates side effects or helper work through `np.sin()`.
        3. Returns the assembled result to the caller.

        Inputs:
        - `g`: Graph instance that the workflow reads from or mutates. Expected type: `float`.
        - `theta_W`: Caller-supplied value used during processing. Expected type: `float`.

        Returns:
        - Returns `float` to the caller.
        """
        return g * np.sin(theta_W)

    def z_boson_coupling(self, g: float, theta_W: float) -> float:
        # g_Z = g / cos(theta_W)
        """
        Z boson coupling for the FermGCoupler workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `g`, `theta_W`.
        2. Delegates side effects or helper work through `np.cos()`.
        3. Returns the assembled result to the caller.

        Inputs:
        - `g`: Graph instance that the workflow reads from or mutates. Expected type: `float`.
        - `theta_W`: Caller-supplied value used during processing. Expected type: `float`.

        Returns:
        - Returns `float` to the caller.
        """
        return g / np.cos(theta_W)

    def w_boson_coupling(self, g: float) -> float:
        # W-Boson verwendet direkt g
        """
        W boson coupling for the FermGCoupler workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `g`.
        2. Returns the assembled result to the caller.

        Inputs:
        - `g`: Graph instance that the workflow reads from or mutates. Expected type: `float`.

        Returns:
        - Returns `float` to the caller.
        """
        return g

    def gluon_coupling(self, g_s: float) -> float:
        # QCD: Gluon-Feld verwendet direkte Kopplungskonstante
        """
        Gluon coupling for the FermGCoupler workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `g_s`.
        2. Returns the assembled result to the caller.

        Inputs:
        - `g_s`: Caller-supplied value used during processing. Expected type: `float`.

        Returns:
        - Returns `float` to the caller.
        """
        return g_s

    def higgs_yukawa_coupling(self, mass_fermion: float, vev: float) -> float:
        # Yukawa: Y_f = m_f / v (z.B. v = 246 GeV)
        """
        Higgs yukawa coupling for the FermGCoupler workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `mass_fermion`, `vev`.
        2. Returns the assembled result to the caller.

        Inputs:
        - `mass_fermion`: Caller-supplied value used during processing. Expected type: `float`.
        - `vev`: Caller-supplied value used during processing. Expected type: `float`.

        Returns:
        - Returns `float` to the caller.
        """
        return mass_fermion / vev



class FermUtils(
    FermGCoupler,
):

    def __init__(self):
        """
        Initialize the FermUtils instance state.

        Workflow:
        1. Starts from the current object state and local workflow context.
        2. Delegates side effects or helper work through `super().__init__()`, `super()`.
        3. Finishes by updating state, triggering side effects, or completing the workflow without a direct return value.

        Inputs:
        - None.

        Returns:
        - Returns `None`; the main effects happen through state updates, I/O, or delegated calls.
        """
        super().__init__()

    def _coupling_ferm_zboson(
            self,
            psi,
            g,  # kupplungskonstante für gauges
            theta_W,
            psi_bar,
            gauge  # feld
    ):


        """
        Coupling ferm zboson for the FermUtils workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `psi`, `g`, `theta_W`, `psi_bar`.
        2. Delegates side effects or helper work through `np.sum()`, `np.cos()`.
        3. Returns the assembled result to the caller.

        Inputs:
        - `psi`: Caller-supplied value used during processing.
        - `g`: Graph instance that the workflow reads from or mutates.
        - `theta_W`: Caller-supplied value used during processing.
        - `psi_bar`: Caller-supplied value used during processing.
        - `gauge`: Caller-supplied value used during processing.

        Returns:
        - Returns the computed result for the caller.
        """
        return g / np.cos(theta_W) * np.sum(psi_bar * np.sum(self.gamma * np.sum(gauge * psi)))

    def _get_quark_doublet(self):
        """
        Erstellt ein valides (3, 2, 4) Quark-Dublett ψ
        für W±-Kopplung basierend auf dem aktuellen Quarktyp (ntype)
        todo untersceide zwischen l/r -> w kuppelt nur an l
        """
        partner_map = {
            "up": "down",
            "charm": "strange",
            "top": "bottom",
        }

        if self.short_lower_type not in partner_map:
            raise ValueError(f"Kein gültiger Quarktyp für W-Kopplung: {self.short_lower_type}")

        # eigener zustand
        psi = getattr(self, "psi")  # erwartet shape (3, 4)
        #print("psi_self", psi_self)

        # Quark kupelt nur an linke seite (if links: rechts = 0 -> nimm kompletten spinor)
        # partner-nid ermitteln
        self_id= getattr(self, "id")
        #print("self_nid", self_nid)

        # alltimes 3,4
        total_down_psi_sum = np.zeros_like(psi, dtype=complex)

        ckm_struct = self.ckm[self.short_lower_type]  # e.g. {"d": 0.974, "s": 0.225, "b": 0.004}
        for quark_type, ckm_val in ckm_struct.items():
            # Extract Neighbor

            quark_type = f"{quark_type}_quark".upper()
            #print(f"get_quark_doublet from {quark_type}")
            neighbor_quark = self.all_subs["FERMION"][quark_type]  # dict: id:attrs

            # Extract Data tuple: id, attrs & args
            item_paare = list(neighbor_quark.items())[0]
            item_attrs = item_paare[1]

            neighbor_psi = item_attrs.get("psi")

            # Get & Check handedness
            n_handedness = item_attrs.get("handedness", None)

            if n_handedness and isinstance(n_handedness, str) and n_handedness == "left":
                # Sum CKM val and neighbor_quark_psi
                component = neighbor_psi * ckm_val

            else:
                # Default value für right handed Quarks
                component = 0

            # Add to total
            total_down_psi_sum += component

        #print(f"psi_self: {psi_self}-{psi_self.shape}")
        #print(f"total_down_psi_sum: {total_down_psi_sum}")
        doublet = np.stack([psi, total_down_psi_sum], axis=1)
        #print(f"Quark doublet extracted: {doublet, doublet.shape}")

        return doublet


    def _is_quark(self, type):
        """
        Is quark for the FermUtils workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `type`.
        2. Builds intermediate state such as `ntype` before applying the main logic.
        3. Delegates side effects or helper work through `type.upper()`, `k.upper()`.
        4. Returns the assembled result to the caller.

        Inputs:
        - `type`: Caller-supplied value used during processing.

        Returns:
        - Returns the computed result for the caller.
        """
        ntype = type.upper()
        return ntype in [k.upper() for k in self.quarks]

    def _init_psi(self, ntype, s=False, stim=True):
        """
        Initialize psi for the FermUtils workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `ntype`, `s`, `stim`.
        2. Builds intermediate state such as `random_stim`, `psi` before applying the main logic.
        3. Branches on validation or runtime state to choose the next workflow path.
        4. Delegates side effects or helper work through `ntype.lower()`, `np.array()`, `ValueError()`.
        5. Returns the assembled result to the caller.

        Inputs:
        - `ntype`: Caller-supplied value used during processing.
        - `s`: Caller-supplied value used during processing.
        - `stim`: Caller-supplied value used during processing.

        Returns:
        - Returns the computed result for the caller.
        """
        random_stim = 1 if stim else 0
        if ntype.lower() in self.leptons:
            psi = np.array([
                [0.0 + 0.0j],
                [random_stim + 0.0j],
                [0.0 + 0.0j],
                [random_stim + 0.0j]
            ], dtype=complex)

        elif ntype.lower() in self.quarks:
            # Beispiel: Dirac-Spinor für einen up-Quark (Farbraum + Spinor → 3x4 Matrix)
            psi = np.array([
                [1.0 + 0.0j, random_stim + 0.0j, random_stim + 0.0j, 0.0 + 0.0j],  # Rot
                [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, random_stim + 0.0j],  # Grün
                [random_stim + 0.0j, random_stim + 0.0j, 1.0 + 0.0j, random_stim + 0.0j],  # Blau
            ], dtype=complex)
        else:
            raise ValueError(f"Invalid type {ntype}")

        ###print("psi set", psi)
        return psi


    ############
    # COUPLINGS
    ############


    def _get_gauge_generator(
            self,
            ntype,
            gluon_index=None,
            psi=None,
            Q=1.0,
            theta_w=28.7,
    ):
        """
        Retrieve gauge generator for the FermUtils workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `ntype`, `gluon_index`, `psi`, `Q`.
        2. Builds intermediate state such as `ntype`, `T`, `T3` before applying the main logic.
        3. Branches on validation or runtime state to choose the next workflow path.
        4. Delegates side effects or helper work through `ntype.lower()`, `np.identity()`, `len()`.
        5. Returns the assembled result to the caller.

        Inputs:
        - `ntype`: Caller-supplied value used during processing.
        - `gluon_index`: Caller-supplied value used during processing.
        - `psi`: Caller-supplied value used during processing.
        - `Q`: Caller-supplied value used during processing.
        - `theta_w`: Caller-supplied value used during processing.

        Returns:
        - Returns the computed result for the caller.
        """
        ntype = ntype.lower()
        if ntype == "photon":
            T = Q * np.identity(len(psi))
            #print("p T", T)

        elif ntype == "z_boson":
            T3 = self.T[2]
            Y = 0.5 * np.identity(2, dtype=complex)
            T = np.cos(theta_w) * T3 - np.sin(theta_w) * Y

            #print(f"zT final: {T}")
        elif ntype == "gluon":
            T = self.su3_group_generators[gluon_index]
            #print("gluon T", T)

        elif ntype == "w_plus":
            T = np.array([[0, 1], [0, 0]], dtype=complex)
            #print("w+ T", T)

        elif ntype == "w_minus":
            T = np.array([[0, 0], [1, 0]], dtype=complex)
            #print("w- T", T)
        return T

    def _get_gauge_generator_kernel(
            self,
            attrs: dict[str, np.ndarray],  # Batched admin_data is unused, but required by vmap pattern
            static_data: dict,  # Holds constants and matrices
            gauge_index: int,  # STATIC argument (0:Photon, 1:Z, 2:Gluon, 3:W+, 4:W-)
    ) -> np.ndarray:
        """
        Calculates the gauge group generator T using JAX-compatible conditional logic.

        The function uses jax.lax.switch to avoid Python control flow.
        """

        # Unpack static constants (This happens at compile-time/trace-time)
        theta_w = static_data['theta_w']
        su2_generators = static_data['su2_generators']
        su3_generators = static_data['su3_generators']

        # Unpack batched attributes
        Q = attrs['Q']
        len_psi = attrs['psi'].shape[0]  # Get dimension from psi shape

        # --- Define JAX-compatible calculation cases ---

        # Case 0: Photon
        def photon_case(args):
            Q, len_psi = args
            return Q * np.identity(len_psi, dtype=np.complex64)

        # Case 1: Z_Boson
        def z_boson_case(args):
            theta_w, su2_generators = args
            T3 = su2_generators[2]  # Third Pauli matrix
            Y = 0.5 * np.identity(T3.shape[0], dtype=np.complex64)  # Identity matrix for weak hypercharge
            return np.cos(theta_w) * T3 - np.sin(theta_w) * Y

        # Case 2: Gluon
        def gluon_case(args):
            su3_generators, gluon_index = args
            return su3_generators[gluon_index]

        # Case 3: W_Plus
        def w_plus_case(args):
            return np.array([[0, 1], [0, 0]], dtype=np.complex64)

        # Case 4: W_Minus
        def w_minus_case(args):
            return np.array([[0, 0], [1, 0]], dtype=np.complex64)

        # --- Use jax.lax.switch to select the correct kernel based on the static index ---

        T = lax.switch(
            gauge_index,
            [photon_case, z_boson_case, gluon_case, w_plus_case, w_minus_case],
            (Q, len_psi) if gauge_index == 0 else (theta_w, su2_generators) if gauge_index == 1 else
            (su3_generators,
             attrs.get('gluon_index')) if gauge_index == 2 else  # gluon_index must be in attrs or static
            # For simple W+- cases, arguments are trivial or constant
            (None,)  # Pass a dummy arg for cases 3 and 4
        )

        return T


    def _extract_psi_lrm(self, psi, handedness, is_quark):
        """
        Extract psi lrm for the FermUtils workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `psi`, `handedness`, `is_quark`.
        2. Builds intermediate state such as `make` before applying the main logic.
        3. Delegates side effects or helper work through `print()`.
        4. Returns the assembled result to the caller.

        Inputs:
        - `psi`: Caller-supplied value used during processing.
        - `handedness`: Caller-supplied value used during processing.
        - `is_quark`: Caller-supplied value used during processing.

        Returns:
        - Returns the computed result for the caller.
        """
        make = {
            "left": psi[:2],
            "right": psi[2:],
        }
        try:
            return make[handedness]
        except Exception as e:
            print(f"Err core.qbrain.core.sm_manager.sm.fermion.ferm_utils::FermUtils._extract_psi_lrm | handler_line=451 | {type(e).__name__}: {e}")
            print(f"[exception] core.qbrain.core.sm_manager.sm.fermion.ferm_utils.FermUtils._extract_psi_lrm: {e}")
            print(f"Err extract_psi_lrm: {e}")
        return psi


    def _fermion_gauge_coupling(
            self,
            psi,
            field_value,
            g,
            T,

    ):
        """
        Fermion-Gauge-Kopplung nach SSB.
        Ja. W⁺/W⁻ koppeln ausschließlich an linkshändige Fermionen.
        """

        term = np.zeros_like(psi, dtype=complex)
        """
        T = self._get_gauge_generator_kernel(
            ntype,
            gluon_index,
            psi,
        ntype,
        gluon_index,
        )
        """
        ##printer(locals())
        for mu in range(4):
            term += -self.i * g * field_value[mu] * (T @ psi)
        return term

