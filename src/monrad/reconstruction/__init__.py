"""Stage 3 — position decoding.

Reconstructs (x, y) hit positions from a detector's ``*.bin`` 16-row block via
the fiber x ribbon encoding (DESIGN.md §6), and enumerates per-plane candidate
positions for the Stage 5 combinatorial track search (DESIGN.md §10).
"""

from .hit import (
    GOOD_QUALITIES as GOOD_QUALITIES,
    Hit as Hit,
    decode_position as decode_position,
)
from .candidates import (
    PlaneCandidate as PlaneCandidate,
    disambiguate_telescope_hits as disambiguate_telescope_hits,
    reconstruct_plane_candidates as reconstruct_plane_candidates,
)
