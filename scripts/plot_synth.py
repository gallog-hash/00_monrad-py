#!/usr/bin/env python3
"""
Visualise the synthetic dataset.

Calls generate() with the same seed so no files need to be
pre-generated; the output is saved alongside this script as
synth_plot.png.
"""

import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from monrad.synth import (    # noqa: E402
    generate, Z_TEL, STRIP_MM, N_TEL,
)

T_X = 50.0
T_Y = -30.0
THETA = math.radians(17.0)
Z_P = 300.0
N_PRB = 30

result = generate(
    out_dir=ROOT / 'tests' / 'data' / 'synth',
    t_x=T_X, t_y=T_Y, theta=THETA, z_p=Z_P,
    n_probe_ch=N_PRB,
)
tracks = result['tracks']
probe_hits = result['probe_hits']
n_coinc = result['n_coincidences']
coinc_set = set(probe_hits)

ax_arr = np.array([t[0] for t in tracks])
bx_arr = np.array([t[1] for t in tracks])
ay_arr = np.array([t[2] for t in tracks])
by_arr = np.array([t[3] for t in tracks])

# Positions at z_p (telescope frame)
xp = ax_arr + bx_arr * Z_P
yp = ay_arr + by_arr * Z_P

# Probe hits in probe frame
cos_t, sin_t = np.cos(THETA), np.sin(THETA)
u_arr = np.array([
    (xp[i] - T_X) * cos_t + (yp[i] - T_Y) * sin_t
    for i in sorted(coinc_set)
])
v_arr = np.array([
    -(xp[i] - T_X) * sin_t + (yp[i] - T_Y) * cos_t
    for i in sorted(coinc_set)
])

mask = np.array([i in coinc_set for i in range(len(tracks))])
zen = np.degrees(np.arctan(np.hypot(bx_arr, by_arr)))
phi = np.degrees(np.arctan2(by_arr, bx_arr))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    f'Synthetic dataset — ground truth: '
    f't_x={T_X:.0f} mm, t_y={T_Y:.0f} mm, '
    f'θ=17°, z_p={Z_P:.0f} mm  '
    f'({n_coinc} coincidences / 1000 tracks)',
    fontsize=11,
)

# ── panel 1: entry positions at z = 0 ──────────────────────────
ax = axes[0]
ax.scatter(
    ax_arr[~mask], ay_arr[~mask],
    s=2, c='steelblue', alpha=0.25, rasterized=True,
)
ax.scatter(
    ax_arr[mask], ay_arr[mask],
    s=8, c='crimson', alpha=0.8,
)
lim = N_TEL * STRIP_MM
ax.set_xlim(-15, lim + 15)
ax.set_ylim(-15, lim + 15)
ax.set_xlabel('x at z = 0  [mm]')
ax.set_ylabel('y at z = 0  [mm]')
ax.set_title('Track entry positions (top telescope plane)')
ax.set_aspect('equal')
ax.legend(
    handles=[
        mpatches.Patch(color='steelblue', label='telescope only'),
        mpatches.Patch(color='crimson',   label='coincident'),
    ],
    fontsize=8,
)

# ── panel 2: probe hits in probe frame ─────────────────────────
ax = axes[1]
prb_lim = N_PRB * STRIP_MM
ax.scatter(u_arr, v_arr, s=10, c='crimson', alpha=0.8)
for val in (0, prb_lim):
    ax.axhline(val, color='k', lw=0.8, ls='--')
    ax.axvline(val, color='k', lw=0.8, ls='--')
ax.set_xlim(-15, prb_lim + 15)
ax.set_ylim(-15, prb_lim + 15)
ax.set_xlabel('u  [mm]  (probe frame)')
ax.set_ylabel('v  [mm]  (probe frame)')
ax.set_title(f'Probe hits ({n_coinc} events)')
ax.set_aspect('equal')

# ── panel 3: angular distribution ──────────────────────────────
ax = axes[2]
h = ax.hist2d(
    phi, zen, bins=30,
    cmap='Blues', rasterized=True,
)
ax.scatter(
    phi[mask], zen[mask],
    s=6, c='crimson', alpha=0.6, label='coincident',
)
fig.colorbar(h[3], ax=ax, label='all tracks')
ax.set_xlabel('azimuth φ  [°]')
ax.set_ylabel('zenith θ  [°]')
ax.set_title('Track angular distribution')
ax.legend(fontsize=8)

plt.tight_layout()
out = Path(__file__).with_name('synth_plot.png')
plt.savefig(out, dpi=130, bbox_inches='tight')
print(f'Saved → {out}')
plt.show()
