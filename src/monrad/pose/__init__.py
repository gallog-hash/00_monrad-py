"""Stage 5 — probe pose fit.

Decodes telescope-probe coincidences and fits the probe pose (t_x, t_y, θ, z_p)
with covariance via the four-step optimizer (DESIGN.md §8).
"""

from .types import (
    Coincidence as Coincidence,
    DecodeReport as DecodeReport,
    GATE_ORDER as GATE_ORDER,
    PoseResult as PoseResult,
)
from .optimize import (
    fit_probe_pose as fit_probe_pose,
    filter_rigidity as filter_rigidity,
    filter_off_probe as filter_off_probe,
    _linear_solve_fixed_theta as _linear_solve_fixed_theta,
    _sigma_tel_at_z as _sigma_tel_at_z,
    _tel_line_fit as _tel_line_fit,
)
from .fitter import (
    PoseFitter as PoseFitter,
    _CHI2_TRACK as _CHI2_TRACK,
)
