"""
Injector: create a function that creates based on sim_time:int and amount_params and values form (-100 - 100), list items [elektron, photon] a powerset.

See ``Injector.make_inj_powerset_data`` and ``nonempty_powerset_indices``.
"""
import pprint
import random
from itertools import combinations, product

from firegraph.graph import GUtils
from qfu.all_subs import ALL_SUBS
from qfu.field_utils import FieldUtils
from qfu.qf_utils import QFUtils


def _inj_powerset_rng_seed(sim_time: int, particle: str) -> int:
    # Deterministic 32-bit seed: avoids Python's salted hash() (not stable across runs).
    acc = sim_time & 0xFFFFFFFF
    for ch in particle:
        acc = (acc * 1315423911 + ord(ch)) & 0xFFFFFFFF
    return acc


def nonempty_powerset_indices(amount_params: int):
    """
    Yield every non-empty subset of range(amount_params) as increasing index tuples.
    Order matches powerset by subset size, then lexicographic (combinations order).
    """
    if amount_params < 0:
        raise ValueError("amount_params must be non-negative")
    idxs = list(range(amount_params))
    for r in range(1, amount_params + 1):
        # return
        yield from combinations(idxs, r)


#@ray.remote
class Injector(
    #BaseActor,
    FieldUtils,
):

    """
    fetch and save ncfg
    Todo later come back to blcoks and phase. for now keep jus blocks
    """
    def __init__(
            self,
            g:GUtils,
            amount_nodes:int
    ):
        """
        Initialize the Injector instance state.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `world_cfg`.
        2. Delegates side effects or helper work through `BaseActor.__init__()`, `FieldUtils.__init__()`, `FBRTDBMgr()`.
        3. Finishes by updating state, triggering side effects, or completing the workflow without a direct return value.

        Inputs:
        - `world_cfg`: Caller-supplied value used during processing.

        Returns:
        - Returns `None`; the main effects happen through state updates, I/O, or delegated calls.
        """
        FieldUtils.__init__(self)
        self.ncfg = {}

        self.g=g
        self.dim=3
        self.cluster_schema = []
        self.amount_nodes=amount_nodes
        self.pos_schema = [
            (0 for _ in range(self.dim))
            for _ in range(self.amount_nodes)
        ]
        #self.injector_schema:dict[str, dict] = self.christmas_tree()

    def get_positions(self, amount, dim) -> list:
        # [(0,0,0), (0,0,1), ..., (1,1,1)]
        print("get_positions...")
        print("amount", amount)
        print("dim", dim)
        _range = list(product(range(amount), repeat=dim))
        print("range:", len(_range))
        return _range


    def rainbow(
        self,
        amount_nodes,
        sim_time: int = 10,
        fields=None,
        dims:int=3,
        value_low: int = -100,
        value_high: int = 100,
    ) -> dict[
            str,  # field
            list[
                tuple[
                    tuple[tuple[int],  # pos
                    list[list[int], list[int]]
                    ] # data
                ]
            ]
        ]:
        print("rainbow amount_nodes", amount_nodes)
        try:
            fields = list(fields)

            # VALIDATE
            if sim_time <= 0 or amount_nodes <= 0 or not fields:
                return {
                    field: []
                    for field in fields
                }

            #
            positions = self.get_positions(
                amount_nodes,
                dims,
            )

            #
            injections = {
                field: [
                    (
                        tuple(pos),
                        [
                            [n for n in range(sim_time)],
                            [random.randint(-100, 100) for _ in range(sim_time)]
                        ]
                    )
                    for pos in positions
                ]
                for field in fields
            }
            print("INJ STRUCT CREATED")
            return injections
        except Exception as e:
            print("Err create rainbow:", e)

    def christmas_tree(self):
        """
        Christmas tree for the Injector workflow.

        Workflow:
        1. Starts from the current object state and local workflow context.
        2. Delegates side effects or helper work through `list()`, `range()`, `random.choices()`.
        3. Returns the assembled result to the caller.

        Inputs:
        - None.

        Returns:
        - Returns the computed result for the caller.
        """
        return  {
            sub:{
                t: [
                    list(random.choices(self.pos_schema, 5)),
                    [
                        random.randint(0, 10)
                        for _ in range(5)
                    ]
                ]
                for t in range(self.sim_time)
            }
            for sub in ALL_SUBS # todo change dynamic get from G
        }

    def get_injector(self, t, ntype):
        """
        Retrieve injector for the Injector workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `t`, `ntype`.
        2. Returns the assembled result to the caller.

        Inputs:
        - `t`: Caller-supplied value used during processing.
        - `ntype`: Caller-supplied value used during processing.

        Returns:
        - Returns the computed result for the caller.
        """
        return self.injector_schema[ntype][t]


    def ncfg_by_time(self, time:int):

        """
        This method imlices all ndim param values are merged to a big list

        punkt : param : [[t], [v]]
        """

        applier_struct = []
        for ntype, struct in self.ncfg.items():
            # add ntype dim
            applier_struct.append([])

            for j, item in enumerate(struct):
                # item[j] represents parameter index
                time_series: list[int] = item[0]
                strenght_series: list[int] = item[1]

                if time in time_series or self.check_repeatment():
                    index = time_series.index(time)
                    strength = strenght_series[index]

    def set_inj_pattern(
        self,
        inj_struct: dict[
        str, # field
            list[tuple[tuple[int], # pos
            list[list[int], list[int]] # data
            ]]
        ]
    ) -> None:
        # ganzer cpu processing stuff wird relay frontend:
        # pi:t:e
        #
        # tod later make docker image for gpu and cpu!

        """
        Set inj pattern for the Injector workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `inj_struct`.
        2. Builds intermediate state such as `modules`, `mid`, `keys` before applying the main logic.
        3. Branches on validation or runtime state to choose the next workflow path.
        4. Delegates side effects or helper work through `self.g.get_nodes()`, `enumerate()`, `self.g.get_neighbor_list()`.
        5. Finishes by updating state, triggering side effects, or completing the workflow without a direct return value.

        Inputs:
        - `inj_struct`: Caller-supplied value used during processing. Expected type: `dict[str, list[tuple[tuple[int], list[list[int], list[int]]]]]`.

        Returns:
        - Returns `list[int, tuple[tuple, float]]` to the caller.
        """
        print("Injections compound to graph...")
        try:
            modules = self.g.get_nodes(
                filter_key="type",
                filter_value="MODULE"
            )
            print("modules count:", len(modules))
            self.module_pattern_collector = []

            for i, (mid, module) in enumerate(modules):
                print(f"set inj for mid {mid}")

                # get module fields
                fields = self.g.get_neighbor_list(
                    node=mid,
                    target_type="FIELD",
                )
                print("fields", len(fields))
                # ds unendlichekeiszeichen ist eigentlich gerade -> nur die daten bewegen sich
                for j, (fid, fattrs) in enumerate(fields.items()):
                    print(f"check {fid} in injection struct...")
                    if fid in inj_struct:
                        field_injection_struct:list[tuple] = inj_struct[fid]
                        print(f"{len(field_injection_struct)} for {fid}")
                        for pos_item_struct in field_injection_struct:
                            pos = pos_item_struct[0]
                            print("pos", pos)
                            data = pos_item_struct[1]
                            print("data", data)

                            self.g.add_node(
                                attrs=dict(
                                    id=f"{pos}__{fid}",
                                    type="INJECTION",
                                    frequency=data[0],
                                    amplitude=data[1],
                                )
                            )

                            self.g.add_edge(
                                src=fid,
                                trgt=f"{pos}__{fid}",
                                attrs=dict(
                                    rel="has_injection",
                                    pos=pos,
                                    fid=fid,
                                    data=data,
                                    src_layer="FIELD",
                                    trgt_layer="INJECTION",
                                )
                            )
                    else:
                        print(f"{fid} has no injections set...")
            print("Injections compound to graph... done")
        except Exception as e:
            print(f"Err set_inj_pattern: {e}")

    def get_update_rcv_ncfg(
            self,
            attr_struct:list[dict]
    ):

        """
        Retrieve update rcv ncfg for the Injector workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `attr_struct`.
        2. Builds intermediate state such as `updated_attr_struct` before applying the main logic.
        3. Branches on validation or runtime state to choose the next workflow path.
        4. Delegates side effects or helper work through `GUtils()`, `QFUtils()`, `self.apply_stim_attr_struct()`.
        5. Returns the assembled result to the caller.

        Inputs:
        - `attr_struct`: Caller-supplied value used during processing. Expected type: `list[dict]`.

        Returns:
        - Returns the computed result for the caller.
        """
        self.g = GUtils(
            G=self.get_G(),
        )

        self.qfu = QFUtils(self.g)

        if not len(list(self.ncfg.keys())):
            self.get_ncfg()

        # prod
        updated_attr_struct = self.apply_stim_attr_struct(
            attr_struct
        )
        print("Finished injection process")
        return updated_attr_struct


    def apply_stim_default(
            self,
            attrs_struct:list[dict]
    ):
        """
        Apply stim default for the Injector workflow.

        Workflow:
        1. Reads and normalizes the incoming inputs, including `attrs_struct`.
        2. Builds intermediate state such as `current_iter` before applying the main logic.
        3. Branches on validation or runtime state to choose the next workflow path.
        4. Delegates side effects or helper work through `len()`, `print()`, `attrs.get()`.
        5. Returns the assembled result to the caller.

        Inputs:
        - `attrs_struct`: Caller-supplied value used during processing. Expected type: `list[dict]`.

        Returns:
        - Returns the computed result for the caller.
        """
        if len(attrs_struct):
            current_iter = attrs_struct[0]["tid"]
            if current_iter // self.world_cfg["phase"] == 0:
                for attrs in attrs_struct:
                    if attrs.get("type") == self.world_cfg["particle"]:
                        attrs["energy"] = self.world_cfg["energy"]
                    #print("Stim applied")
            else:
                print("Skipping stim at iter", current_iter)
            return attrs_struct
        else:
            print("apply_stim_default has no len attrs_struct")

    def apply_stim_attr_struct(
            self,
            attr_struct:list[dict]
    ):
        """

        Checks each node of a list for current stim phase and applies energy to it

        """
        # gnn kannit ur einem node alles ancheinander prozessieren
        print("Applying stim to attr struct")

        for attrs in attr_struct: # loop each field attrs
            try:
                nid = attrs["id"]
                tid = attrs["tid"]
                if nid in self.ncfg: # do we have a ncfg for this node?
                    total_iters = self.ncfg[nid].get("total_iters", 0)
                    print("total_iters", total_iters)
                    if total_iters == 0:
                        blocks:list = self.ncfg[nid]["blocks"]
                        print("blocks", blocks)
                        for block in blocks: # loop blocks
                            for phase in block: # loop single phase
                                total_iters += int(phase["iters"])
                                print("new total_iters", total_iters)

                    # calculate the rest of t_i / tid
                    rest_val: int = total_iters // tid
                    print("rest_val", rest_val)

                    # get current phase
                    total_phase_iters = 0
                    current_phase = None
                    for block in self.ncfg[nid]["blocks"]:
                        print("block", block)
                        for phase in block:
                            print("phase", phase)

                            total_phase_iters += int(phase["iters"])
                            if total_phase_iters > rest_val:
                                # Right phase found
                                current_phase=phase
                                print("new current_phase", current_phase)
                                break

                    if current_phase is not None:
                        attrs["energy"] = current_phase["energy"]
                        print("Applied stim to", nid, attrs["energy"])
                    else:
                        print("Err: couldnt identify aphase to apply stim to")
            except:
                print("Err cor.qbrain.cor.injector::Injector.apply_stim_attr_struct | handler_line=388 | Exception handler triggered")
                print("[exception] cor.qbrain.cor.injector.Injector.apply_stim_attr_struct: caught Exception")
                print(f"Err: couldnt identify aphase: {nid}")
        return attr_struct

if __name__ == "__main__":
    cfg = Injector.make_inj_powerset_data(sim_time=100, amount_params=5, particle_items=["ELECTRON", "PHOTON"])
    print("cfg", cfg)
    print("len", len(cfg))
    print("keys", cfg.keys())
    print("values", cfg.values())
    print("items", cfg.items())



"""



        for field_index, field in enumerate(fields):
            for pos_index, pos in enumerate(positions):
                amplitudes = [
                    amplitude_cycle[(time_step + pos_index + field_index) % len(amplitude_cycle)]
                    for time_step in time_steps
                ]
                data = [amplitudes, list(time_steps)]
                inj_struct[field].append((pos, data))

                if self is not None and getattr(self, "g", None) is not None:
                    inj_id = f"{pos}__{field}"
                    self.g.add_node(
                        attrs=dict(
                            id=inj_id,
                            type="INJECTION",
                            frequency=data[1],
                            amplitude=data[0],
                        )
                    )
                    self.g.add_edge(
                        src=field,
                        trgt=inj_id,
                        attrs=dict(
                            rel="has_injection",
                            pos=pos,
                            fid=field,
                            data=data,
                            src_layer="FIELD",
                            trgt_layer="INJECTION",
                        )
                    )


"""