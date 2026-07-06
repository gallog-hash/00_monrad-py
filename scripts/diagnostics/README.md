# Diagnostics — testLab 20210723 wide-angle anomaly

One-off inspectors written while tracing the testLab 20210723 17:08–18:26 UTC
`z_p` excursion to its root cause. They are **investigation scripts, not
pipeline code** — no tests, not wired into the CLI. The full diagnosis and all
numeric results are in the handoff
[`docs/handoffs/2026-07-06-wide-b-chi2-discriminant-feasibility.md`](../../docs/handoffs/2026-07-06-wide-b-chi2-discriminant-feasibility.md)
and project memory `testlab-20210723-anomaly-no-raw-telescope-signature`.

Run from the repo root (each does `sys.path.insert(0, "src")`). The ones that
persist an `.npz` write to `$MONRAD_DIAG_OUT` (default: current dir); run
`coinc_dt.py` before `wide_block_inspect.py` (which reads `coinc_dt.npz`).

| script | what it measures |
|---|---|
| `tel_raw_inspect.py` | raw telescope 3-plane self-track slope \|b\| / residual per `.bin` file |
| `tel_plane_inspect.py` | raw telescope per-plane occupancy / multiplicity / quality |
| `tel_time_inspect.py` | telescope event temporal burstiness (interarrival cv, Fano) from `_GPS.bin` |
| `probe_raw_inspect.py` | raw Probe_0 rate/occupancy/multiplicity/quality/position, UTC-bucketed |
| `coinc_b_dist.py` | coincidence-level telescope-track \|b\|=hypot(b_x,b_y) distribution, UTC-bucketed |
| `coinc_dt.py` | per-coincidence Δt = t_tel − t_prb vs \|b\| (timing / mis-pairing test) |
| `wide_block_inspect.py` | candidate multiplicity of the telescope blocks under wide vs narrow coincidences |

Dataset paths are hard-coded to `data/0_testLab_20210723/{Base,Probe_0}` with
`--z-tel 0 -1340 -670`; edit inline to point at another acquisition.
