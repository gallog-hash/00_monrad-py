#!/usr/bin/env python3
"""
Generate synthetic telescope + probe files for pipeline testing.

Ground truth: t_x=50 mm, t_y=-30 mm, theta=17°, z_p=300 mm.
Output goes to tests/data/synth/{telescope,probe}/.
"""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from monrad.synthetic import generate  # noqa: E402

OUT = ROOT / "tests" / "data" / "synth"

T_X_MM = 50.0
T_Y_MM = -30.0
THETA_RAD = math.radians(17.0)
Z_P_MM = 300.0

if __name__ == "__main__":
    result = generate(
        out_dir=OUT,
        t_x=T_X_MM,
        t_y=T_Y_MM,
        theta=THETA_RAD,
        z_p=Z_P_MM,
    )
    n = result["n_coincidences"]
    print("Tracks generated : 1000")
    print(f"Probe coincidences: {n}  ({n / 10:.1f}%)")
    print(f"Telescope files  : {result['tel_dir']}")
    print(f"Probe files      : {result['probe_dir']}")
    for subdir in (result["tel_dir"], result["probe_dir"]):
        for f in sorted(subdir.iterdir()):
            kb = f.stat().st_size / 1024
            print(f"  {f.name:<40} {kb:7.1f} kB")
