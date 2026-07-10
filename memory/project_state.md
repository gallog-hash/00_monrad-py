---
name: Pipeline implementation status
description: Which stages are implemented, streaming redesign status
type: project
---

Stage 1 (time reconstruction) is implemented in `src/monrad/stage1.py`.
The streaming redesign from DESIGN_UPDATE.md §2 is complete.

**Why:** DESIGN_UPDATE.md replaces the batch `reconstruct()` with a
streaming `reconstruct_stream()` generator to bound RAM usage.

**How to apply:** New pipeline code should consume `reconstruct_stream()`.
The deprecated `reconstruct()` wrapper still exists for backward compat.

Key symbols in stage1.py:
- `reconstruct_stream()` — generator yielding `(TimedEvent, PosRef)` pairs
- `_build_next_interval(c0, c1, n0, f0, tau)` — single-pair interval builder
- `_iter_gps_records(path)` — yields `(tick, gen, is_pps)` in acquisition order
- `reconstruct()` — deprecated thin wrapper, emits DeprecationWarning

All five stages are implemented.

`synth.generate()` now accepts `plane_offsets: dict[int, tuple[float,float]]`
to simulate per-plane translational misalignments.

Tests: 146 total, all pass.
- `tests/test_stage1.py`: 45 tests
- `tests/test_stage2.py`: 6 tests
- `tests/test_stage3.py`: 5 tests
- `tests/test_stage4.py`: 29 tests (identity, accumulator filtering, offset recovery,
  middle-plane-by-z selection — see below)
- `tests/test_stage5.py`: 16 tests (helpers, integration, 3σ parameter recovery)

Stage 4 key notes:
- `fit_telescope_alignment` selects the tiltable middle plane by z-order
  (`mid = int(np.argsort(z)[1])`), not hardcoded file-column 1. Lets columns be
  stored out of z order (e.g. `--z-tel 0 -1340 -670`). Merged to `main` via
  PR #8 (merge commit 3805500); covered by `TestMiddlePlaneByZOrder`. The same
  PR added a "Run configuration" block (data dirs + telescope z) to the top of
  `run_pipeline.py`'s summary.txt. See [[testlab-20210723-plane-z-order]].

Stage 5 key notes:
- `PoseFitter.MIN_FIT = 30` (not 50) — with n_tracks=1000, only ~48 coincidences
  survive the chi2<4.0 telescope line-fit cut after quantization effects.
- 4-fold θ degeneracy: square probe gives equivalent solutions at θ+k·π/2.
  Tests handle this with `_theta_err_mod90()` and `_nearest_k90()` helpers.
