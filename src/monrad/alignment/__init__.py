"""Stage 4 — telescope alignment.

Fits per-plane offsets, rotations and out-of-plane tilts from the telescope's
own straight-through tracks (DESIGN.md §7).
"""

from .accumulator import (
    AlignmentAccumulator as AlignmentAccumulator,
    AlignmentCorrection as AlignmentCorrection,
    PlaneCorrection as PlaneCorrection,
    fit_telescope_alignment as fit_telescope_alignment,
)
from .io import (
    load_alignment as load_alignment,
    save_alignment as save_alignment,
)
