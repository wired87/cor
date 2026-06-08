"""
Gluon neighbor mesh for the 3D pixel grid.

User prompt — "Each grid point is a pixel; 8 gluons (gluon_0..gluon_7) are sub-points
of a 3D cube within that pixel. Each gluon interacts with the environment across 26 spatial
neighbors, forming a 3D mesh. Build an algorithm that, for every pixel id and every gluon id,
resolves the gluon interaction partners as:
  dict[px_id, dict[px_gluon_id, list[tuple(neighbor_px_id, neighbor_gluon_id)]]]
Reuse existing 26-neighbor direction logic (FieldUtils.shift_dirs / npm pm_axes)."

Each gluon index i encodes a sub-voxel corner of the unit cube inside one pixel:
  lx = i & 1,  ly = (i >> 1) & 1,  lz = (i >> 2) & 1
When stepping to a neighbor pixel along direction (dx, dy, dz) the paired gluon index j
is the corner that meets on the shared face / edge / corner (standard staggered sub-grid).
Interior pixels yield up to 26 partners per gluon; boundary pixels yield fewer (no self-fallback).
"""

from __future__ import annotations

import os
import sys

# CHAR: repo root on sys.path so `from qfu.*` works when run as script or imported from main
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from typing import Dict, List, Optional, Tuple

from qfu.all_subs import GLON_MAP
from qfu.field_utils import FieldUtils

# CHAR: type aliases for the requested return structure
PxId = str
GluonId = str
GluonNeighborMap = Dict[PxId, Dict[GluonId, List[Tuple[PxId, GluonId]]]]

# CHAR: 8 gluon sub-indices per pixel — same count as GLON_MAP / g_creator range(8)
NUM_GLUONS = len(GLON_MAP)  # 8


def _gluon_local_coords(gluon_index: int) -> Tuple[int, int, int]:
    """Decode gluon_0..7 into local sub-cube corner (lx, ly, lz) ∈ {0,1}^3."""
    if not (0 <= gluon_index < NUM_GLUONS):
        raise ValueError(f"gluon_index must be in 0..{NUM_GLUONS - 1}, got {gluon_index}")
    # CHAR: bit layout matches g_creator item_index 0..7 as corners of the pixel cube
    lx = gluon_index & 1
    ly = (gluon_index >> 1) & 1
    lz = (gluon_index >> 2) & 1
    return lx, ly, lz


def _gluon_index_from_local(lx: int, ly: int, lz: int) -> int:
    """Encode local corner (lx, ly, lz) back to gluon index 0..7."""
    return int(lx) | (int(ly) << 1) | (int(lz) << 2)


def _px_id_from_coords(x: int, y: int, z: int) -> PxId:
    """Pixel id string — matches `gluon__px_{x}_{y}_{z}__{i}` prefix used in qf_utils.get_neighbor_ids."""
    return f"px_{x}_{y}_{z}"


def _gluon_id(gluon_index: int) -> GluonId:
    """Human-readable gluon key inside one pixel (gluon_0 .. gluon_7)."""
    return f"gluon_{gluon_index}"


def iter_26_directions(shift_dirs: Dict[str, Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    """Return the 26 neighbor offset vectors — same geometry as FieldUtils.get_dirs / npm pm_axes.

    For each entry in shift_dirs (13 axis / diagonal directions) we emit +offset and -offset.
    This mirrors `get_dirs()` which builds plus/minus lists from the same shift_dirs table.
    """
    directions: List[Tuple[int, int, int]] = []
    for _name, plus_vec in shift_dirs.items():
        plus = tuple(int(v) for v in plus_vec)
        # CHAR: minus side uses per-component sign flip — identical to FieldUtils.get_dirs()
        minus = tuple(-int(v) if v != 0 else 0 for v in plus)
        directions.append(plus)
        if minus != plus:
            directions.append(minus)
    return directions


def paired_neighbor_gluon_index(
    source_gluon_index: int,
    direction: Tuple[int, int, int],
) -> Optional[int]:
    """Return neighbor gluon index for one step along `direction`, or None if no face match.

    Sub-cube exit rule (3D mesh):
      - axis with dx=+1 : source must have lx=1, neighbor gets lx'=0
      - axis with dx=-1 : source must have lx=0, neighbor gets lx'=1
      - axis with dx=0  : lx' = lx  (same sub-column along that axis)
    Same for y/z. If any required face is missing, this gluon does not couple in that direction.
    """
    lx, ly, lz = _gluon_local_coords(source_gluon_index)
    dx, dy, dz = (int(direction[0]), int(direction[1]), int(direction[2]))

    # CHAR: x-axis face matching
    if dx > 0:
        if lx != 1:
            return None
        lx_n = 0
    elif dx < 0:
        if lx != 0:
            return None
        lx_n = 1
    else:
        lx_n = lx

    # CHAR: y-axis face matching
    if dy > 0:
        if ly != 1:
            return None
        ly_n = 0
    elif dy < 0:
        if ly != 0:
            return None
        ly_n = 1
    else:
        ly_n = ly

    # CHAR: z-axis face matching
    if dz > 0:
        if lz != 1:
            return None
        lz_n = 0
    elif dz < 0:
        if lz != 0:
            return None
        lz_n = 1
    else:
        lz_n = lz

    return _gluon_index_from_local(lx_n, ly_n, lz_n)


def build_pixel_coord_index(amount_nodes: int) -> Dict[PxId, Tuple[int, int, int]]:
    """Map every pixel id → integer grid coordinate (x, y, z) for an N×N×N lattice."""
    index: Dict[PxId, Tuple[int, int, int]] = {}
    for x in range(amount_nodes):
        for y in range(amount_nodes):
            for z in range(amount_nodes):
                index[_px_id_from_coords(x, y, z)] = (x, y, z)
    return index


def build_gluon_neighbor_map(amount_nodes: int) -> GluonNeighborMap:
    """Build full gluon neighbor mesh for all pixels and all 8 gluons per pixel.

    Returns:
        dict[px_id, dict[gluon_k, list[(neighbor_px_id, neighbor_gluon_j)]]]

    Uses FieldUtils.shift_dirs for the same 26-direction stencil as qf_utils.npm / all_px_neighbors.
    """
    if amount_nodes < 1:
        raise ValueError("amount_nodes must be >= 1")

    # CHAR: reuse existing direction table from FieldUtils (13 dirs → 26 signed offsets)
    shift_dirs = FieldUtils().shift_dirs
    directions = iter_26_directions(shift_dirs)
    coord_by_px = build_pixel_coord_index(amount_nodes)

    result: GluonNeighborMap = {}

    for px_id, (x, y, z) in coord_by_px.items():
        per_gluon: Dict[GluonId, List[Tuple[PxId, GluonId]]] = {}

        for gi in range(NUM_GLUONS):
            gid = _gluon_id(gi)
            partners: List[Tuple[PxId, GluonId]] = []

            for direction in directions:
                dx, dy, dz = direction
                nx, ny, nz = x + dx, y + dy, z + dz

                # CHAR: boundary — skip missing neighbor pixels (no npm-style self fallback)
                if not (0 <= nx < amount_nodes and 0 <= ny < amount_nodes and 0 <= nz < amount_nodes):
                    continue

                nj = paired_neighbor_gluon_index(gi, direction)
                if nj is None:
                    continue

                n_px = _px_id_from_coords(nx, ny, nz)
                partners.append((n_px, _gluon_id(nj)))

            per_gluon[gid] = partners

        result[px_id] = per_gluon

    return result


def gluon_neighbors_for_pixel(
    neighbor_map: GluonNeighborMap,
    px_id: PxId,
    gluon_id: GluonId,
) -> List[Tuple[PxId, GluonId]]:
    """Convenience lookup: one gluon at one pixel → list of (neighbor_px, neighbor_gluon)."""
    return list(neighbor_map.get(px_id, {}).get(gluon_id, []))


def full_gluon_node_id(px_id: PxId, gluon_index: int) -> str:
    """Canonical graph node id — matches qf_utils.get_neighbor_ids('gluon', px_suffix)."""
    # CHAR: px_id is already 'px_x_y_z'; get_neighbor_ids expects suffix without duplicate 'px_'
    suffix = px_id[3:] if px_id.startswith("px_") else px_id
    return f"gluon__px_{suffix}__{gluon_index}"


def _print_sample(map_: GluonNeighborMap, amount_nodes: int) -> None:
    """CHAR: readable sanity output for __main__ — interior vs corner partner counts."""
    interior_px = _px_id_from_coords(amount_nodes // 2, amount_nodes // 2, amount_nodes // 2)
    corner_px = _px_id_from_coords(0, 0, 0)

    print(f"[gluon_neighbors] grid N={amount_nodes}  pixels={amount_nodes ** 3}  gluons/pixel={NUM_GLUONS}")
    print(f"[gluon_neighbors] 26-direction stencil from shift_dirs: {len(iter_26_directions(FieldUtils().shift_dirs))} offsets")

    for label, px in [("corner", corner_px), ("interior", interior_px)]:
        print(f"\n--- {label} pixel {px} ---")
        for gi in range(NUM_GLUONS):
            gid = _gluon_id(gi)
            partners = map_[px][gid]
            lx, ly, lz = _gluon_local_coords(gi)
            print(f"  {gid} local=({lx},{ly},{lz})  partners={len(partners)}")
            # CHAR: show first 3 pairs so mesh wiring is visible without flooding stdout
            for n_px, n_g in partners[:3]:
                print(f"      -> ({n_px}, {n_g})")
            if len(partners) > 3:
                print(f"      ... +{len(partners) - 3} more")


if __name__ == "__main__":
    # CHAR: prompt test — default N=3 matches output/config/sim_config.json AMOUNT_NODES
    import os

    n = int(os.environ.get("AMOUNT_NODES", "3"))
    print(f"[gluon_neighbors] building map for AMOUNT_NODES={n}")

    mesh = build_gluon_neighbor_map(amount_nodes=n)

    center = _px_id_from_coords(n // 2, n // 2, n // 2)
    # CHAR: interior corner gluon owns one octant → 7 of 26 directions (8 gluons × ~7 = 26 pixel-level links)
    center_g1 = mesh[center]["gluon_1"]
    print(f"\n[gluon_neighbors] interior {center} gluon_1 partner count = {len(center_g1)} (7 = one octant of 26-dir stencil)")

    _print_sample(mesh, n)

    # CHAR: example full node id used elsewhere in the pipeline
    example_node = full_gluon_node_id(center, 1)
    print(f"\n[gluon_neighbors] example graph node id: {example_node}")
