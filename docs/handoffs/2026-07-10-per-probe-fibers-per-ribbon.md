# Per-probe fibers-per-ribbon-channel (N) — plan

## Context

New fact from the user, not yet reflected anywhere in the code: **the number
of fibres wired per ribbon channel can differ between two probes**, and it
affects position reconstruction. Physically, a ribbon channel's 10-bit fiber
mask (bits 10–19 of the 20-bit X/Y field, DESIGN.md §2.3) always has 10 bit
positions in hardware (fixed by the readout ASIC), but a probe may only wire
fibres to the first `N` of those 10 positions — the rest never fire. `N` is
exactly DESIGN.md §2.4's combine factor:

```
ch = N · ribbon_bit + fiber_bit ,   N = 10   (currently hardcoded everywhere)
```

DESIGN.md §2.4 and `CLAUDE.md`'s "Fiber × ribbon encoding" invariant both
currently state `N = 10` as a fixed fact ("comfortably covers... any
practical probe"). That's now known wrong for at least one real deployment:
`N` must become a **per-detector parameter**, not a constant.

**Scope, confirmed with the user:**
- Only `N` (the combine factor) varies per probe. The physical strip pitch
  (`STRIP_MM = 10.0`, DESIGN.md §6.5) stays global — unaffected.
- No concrete probe numbers yet — this is general plumbing, sized the same
  way `--n-probe-ch` already is (single value broadcast, or one per probe).
- Telescope `N` is assumed fixed at 10 (not stated as varying) — out of
  scope to parameterize the telescope decode path.

**Confirmed already correct, no change needed:** `combine_channel`/
`split_channel` (`src/monrad/decoders/position.py:47-54`) and
`BinDecoder._reconstruct_coord` (`position.py:145-159`) already take an `n`/
`N` parameter distinct from `POS_HALF_BITS` (the fixed 10-bit field width).
The primitives already anticipated this; only callers hardcode `N=10`.

**Bug found during investigation (must fix, not just refactor):**
`src/monrad/reconstruction/hit.py` sets a single module constant
`_N = POS_HALF_BITS` (`hit.py:29`) and reuses it for two *logically distinct*
things that happen to both equal 10 today:
1. the combine factor `N` passed to `combine_channel`/`split_channel`/
   `_reconstruct_coord` (this is the thing that must vary per probe), and
2. slicing the 20-element per-bit TOT `counts` array into ribbon/fiber halves
   — `ribbon_counts = counts[:_N]` / `fiber_counts = counts[_N:]`
   (`hit.py:211-212`, and the equivalent at `hit.py:281-282`).

(2) is **always** a slice at the raw hardware bit width (10), because
`_bit_counts` (`hit.py:92-110`) always returns ribbon bits at indices 0-9 and
fiber bits at 10-19, regardless of how many of those fiber bits are actually
wired. If a probe's `N` becomes e.g. 5 and this slice still uses `_N`, TOT
weighting would silently read the wrong half of the counts array. Fixing this
requires **two separate constants** going forward, not one.

## Design decisions

1. **Name:** `n_fibers_per_ribbon`, mirroring `n_probe_ch`'s naming
   (parameter name, CLI flag `--fibers-per-ribbon`). Default `10`
   (`POS_HALF_BITS`) everywhere, so every existing call site and test is
   behavior-unchanged unless it opts in.
2. **Two constants, not one, in `hit.py`:** keep `POS_HALF_BITS` (imported,
   fixed at 10) for the raw-bit-width slice in `_bit_counts`-derived arrays
   and `split_half`; thread a new `n: int = POS_HALF_BITS` parameter through
   the combine-factor call sites only. This is the fix for the bug above,
   done as part of adding the parameter (not a separate pass).
3. **Only the probe-side decode path needs a variable `N`.** Tracing actual
   callers:
   - `reconstruct_plane_candidates` is called *only* for the telescope
     (`pose/fitter.py:170`, `n_cols=3`) — telescope `N` stays default, no
     caller change needed there, but the function still gains the parameter
     (default `POS_HALF_BITS`) for API symmetry and so a future
     telescope-varies case isn't a second migration.
   - `decode_position` is called for the telescope in `fit_alignment`
     (`monitor/io.py:105-116`, always default `N`) and for the probe in
     `PoseFitter._decode_cluster` (`pose/fitter.py:244-250`) and directly in
     `stream_coincidences` isn't a caller — `PoseFitter` is. The probe call
     in `fitter.py:244` is the one real call site that needs a non-default
     `N`.
4. **`PoseFitter` gains one constructor parameter**, `prb_fibers_per_ribbon:
   int = POS_HALF_BITS`, used only in its own probe-side `decode_position`
   call. Telescope decode inside the same fitter is untouched.
5. **CLI plumbing mirrors `--n-probe-ch` exactly:** `monrad-monitor` gets a
   scalar `--fibers-per-ribbon` (default 10); `monrad-multiprobe` gets a
   repeatable `--fibers-per-ribbon` (`nargs="+"`, default `[10]`, broadcast-
   or-match-`--probe`-count validation, same pattern as
   `multiprobe.py:97-105` / `:299-304`).
6. **Synthetic generator needs the same parameter** to produce test data for
   a non-default-`N` probe: `_ch_to_u64`'s hardcoded `c // 10, c % 10` split
   (`synthetic/generate.py:181-182`) becomes `c // n, c % n`, and `generate()`
   gains `n_probe_fibers_per_ribbon: int = 10` threaded to the probe encode
   call (`generate.py:476`). Telescope encode call (`generate.py:450`) keeps
   the default. `_cluster_fiber_mask`'s width clamp
   (`generate.py:105-119`, capped at 10) is **unaffected** — it bounds
   cluster width within the raw 10-bit fiber field, unrelated to `N`;
   generating multi-bit *cluster* hits for a non-default-`N` probe is out of
   scope (see Deferred) — only golden (width=1) hits need to exercise a
   non-default `N` for this change's tests.

## Changes

### `src/monrad/reconstruction/hit.py`
- Remove the conflated `_N = POS_HALF_BITS` module constant.
- `decode_position(..., n_fibers_per_ribbon: int = POS_HALF_BITS)` — new
  parameter, threaded to `_decode_axis` calls (currently `hit.py:354-355`)
  and the `_axis_candidates` calls in the unresolved-axis branch
  (`hit.py:365`, `370`).
- `_decode_axis(field_or, bit_counts=None, n: int = POS_HALF_BITS)` — pass
  `n` to `BinDecoder._reconstruct_coord(fcs, rcs, n)` (`hit.py:274`) and to
  the TOT-weighting branch (`hit.py:280-283`), where `bit_counts[:n]` /
  `bit_counts[n:]` must become `bit_counts[:POS_HALF_BITS]` /
  `bit_counts[POS_HALF_BITS:]` (the bug fix — always the raw width, never
  `n`) while `_tot_weighted_centroid`'s own `n` param stays the combine
  factor.
- `_axis_candidates(field_or, n: int = POS_HALF_BITS)` — `combine_channel(r,
  f, n)` (`hit.py:180`) uses the parameter.
- `_axis_candidates_with_tot(field_or, counts, tot_weights=False, n: int =
  POS_HALF_BITS)` — same bug-fix split as `_decode_axis`:
  `ribbon_counts = counts[:POS_HALF_BITS]` / `fiber_counts =
  counts[POS_HALF_BITS:]` (fixed width, `hit.py:211-212`), while
  `combine_channel(r, f, n)` (`hit.py:240`) and `split_channel(c, n)`
  (`hit.py:220`) use the combine-factor parameter.
- `_tot_weighted_centroid(candidates, ribbon_counts, fiber_counts, n: int =
  POS_HALF_BITS)` — `split_channel(ch, n)` (`hit.py:153`) uses the parameter.
- **Gate:** every one of these functions must default to `POS_HALF_BITS` so
  no existing caller changes behavior.

### `src/monrad/reconstruction/candidates.py`
- `reconstruct_plane_candidates(..., n_fibers_per_ribbon: int =
  POS_HALF_BITS)` — threaded to the two `_axis_candidates_with_tot` calls
  (`candidates.py:94-95`). Import `POS_HALF_BITS` from `..decoders.position`
  (currently only `_STRIP_MM` is imported from `.hit`, `candidates.py:19-25`).

### `src/monrad/pose/fitter.py`
- `PoseFitter.__init__` gains `prb_fibers_per_ribbon: int = POS_HALF_BITS`
  (import `POS_HALF_BITS` from `..decoders.position`), stored as
  `self.prb_fibers_per_ribbon`.
- `_decode_cluster`'s probe `decode_position` call (`fitter.py:244-250`)
  passes `n_fibers_per_ribbon=self.prb_fibers_per_ribbon`. The telescope
  `reconstruct_plane_candidates` call (`fitter.py:170-177`) is left at its
  default — telescope `N` is out of scope.

### `src/monrad/monitor/io.py`
- `stream_coincidences(..., fibers_per_ribbon: int = POS_HALF_BITS)` —
  passed to `PoseFitter(prb_fibers_per_ribbon=fibers_per_ribbon, ...)`
  (`io.py:168-178`).
- `fit_alignment` (telescope-only) is untouched.

### `src/monrad/monitor/timeseries.py`
- `monitor_probe(..., fibers_per_ribbon: int = 10)` — passed through to the
  `stream_coincidences(...)` call at `timeseries.py:596`.
- CLI (`_build_parser`/`main`): add `--fibers-per-ribbon` (scalar `type=int,
  default=10`), same run-configuration logging pattern as `--n-probe-ch`
  (`timeseries.py:895`, `:905`).

### `src/monrad/monitor/multiprobe.py`
- `monitor_probes(..., fibers_per_ribbon: list[int] | None = None)` — same
  broadcast-or-match-`prb_dirs`-length validation as `n_probe_ch`
  (`multiprobe.py:97-105`), passed per-probe into each
  `PoseFitter(prb_fibers_per_ribbon=fibers_per_ribbon[k], ...)`
  (`multiprobe.py:117-129`).
- CLI: `--fibers-per-ribbon` (`nargs="+"`, `type=int`, `default=[10]`), same
  parse-time length validation as `--n-probe-ch`
  (`multiprobe.py:210-220`, `:299-304`), same run-configuration logging
  (`multiprobe.py:330`).

### `src/monrad/synthetic/generate.py`
- `_ch_to_u64(..., n: int = 10)` — `r_y, f_y = c_y // n, c_y % n` / `r_x, f_x
  = c_x // n, c_x % n` (`generate.py:181-182`), used by both the plain and
  folded encode paths (fold reads `r_y`/`f_y` etc. computed from this
  division).
- `generate(..., n_probe_fibers_per_ribbon: int = 10)` — passed as `n=` to
  the probe's `_ch_to_u64` call (`generate.py:476`). The telescope's
  `_ch_to_u64` call (`generate.py:450`) keeps the default.

### `DESIGN.md`
- §2.4: change `ch = N · ribbon_bit + fiber_bit, N = 10` to state `N` is a
  per-detector parameter (hardware wiring choice — how many of a ribbon's 10
  fiber positions are actually connected), defaulting to 10, with the
  telescope currently assumed fixed at 10 and probes potentially differing.
- §6.5: note (if not already implied) that the channel→mm formula's `ch` is
  computed with that detector's own `N`.

### `CLAUDE.md`
- Update the "Fiber × ribbon encoding" bullet under "Key invariants to
  preserve": `ch = 10 × ribbon_bit + fiber_bit` is the *default*, not a fixed
  fact — `N` (fibers wired per ribbon channel) is configurable per probe via
  `n_fibers_per_ribbon`/`--fibers-per-ribbon`, defaulting to 10.

### Tests
- `tests/test_stage3.py`: new cases (near `TestDecodePosition` /
  `TestPlaneCandidates`) decoding a golden hit encoded with `n=5` (or similar
  non-default value) via `decode_position(..., n_fibers_per_ribbon=5)` and
  `reconstruct_plane_candidates(..., n_fibers_per_ribbon=5)`, asserting the
  recovered channel/mm matches the `n=5` encoding rather than the `n=10`
  default — this is the test that would have caught the bit_counts-slicing
  bug if `_N` had silently diverged from `POS_HALF_BITS` before this fix.
- `tests/test_stage5.py` or `tests/test_monitor_multiprobe.py`: one
  end-to-end case with two synthetic probes generated at different
  `n_probe_fibers_per_ribbon` (e.g. 10 and 5) sharing one telescope
  acquisition, run through `monitor_probes(..., fibers_per_ribbon=[10, 5])`,
  asserting each probe's pose still recovers its truth — this is the
  regression test for the actual motivating scenario ("two probes differ").
- `tests/test_monitor_multiprobe.py`: `--fibers-per-ribbon` broadcast vs.
  per-probe list vs. parse-error cases, mirroring the existing `--n-probe-ch`
  tests exactly.
- Full suite must stay green with zero behavior change for every existing
  test (all defaults are 10 = current hardcoded behavior).

## Verification

```bash
uv run ruff check . && uv run ruff format --check .
uv run pytest tests/test_stage3.py tests/test_stage5.py -q
uv run pytest tests/test_monitor_timeseries.py tests/test_monitor_multiprobe.py -q
uv run pytest -q   # full suite, must be unchanged
```

Confirm: a probe encoded with a non-default `N` decodes correctly only when
`n_fibers_per_ribbon` is passed explicitly (and decodes *wrong* if the
default is used against non-default-`N` data — proving the parameter is
actually load-bearing, not a no-op); two probes with different `N` sharing
one `monrad-multiprobe` run each recover their own truth.

## Critical files

- **Core fix (bug + parameterization):** `src/monrad/reconstruction/hit.py`
  (remove `_N`, split into fixed `POS_HALF_BITS` slicing vs. variable `n`
  combine factor — this is where the latent bug lives today).
- **Threading:** `src/monrad/reconstruction/candidates.py`,
  `src/monrad/pose/fitter.py`, `src/monrad/monitor/io.py`,
  `src/monrad/monitor/timeseries.py`, `src/monrad/monitor/multiprobe.py`.
- **Test data generation:** `src/monrad/synthetic/generate.py`.
- **Docs to update:** `DESIGN.md` §2.4/§6.5, `CLAUDE.md`'s fiber×ribbon
  invariant bullet.
- **Unaffected (verified during investigation):**
  `src/monrad/decoders/position.py` (`combine_channel`/`split_channel`/
  `BinDecoder._reconstruct_coord` already parameterized), `STRIP_MM`/physical
  pitch (stays global per user's confirmed scope), telescope decode paths
  (`fit_alignment`, the `reconstruct_plane_candidates` call in
  `PoseFitter._decode_cluster`), `n_probe_ch`/footprint-gate math
  (orthogonal axis — channel *count*, not fibers-per-ribbon).

## Deferred (out of scope for this pass)

- Telescope-side variable `N` (not stated as varying; API left symmetric but
  unexercised).
- Synthetic generation of multi-bit *cluster*-width hits for a non-default-`N`
  probe (`_cluster_fiber_mask`'s clamp stays hardcoded at the raw 10-bit
  field) — only golden (width=1) hits are needed to test this change.
- Per-probe `STRIP_MM`/strip pitch (explicitly ruled out by the user for this
  pass).
