# Code review findings to fix — feat/probe-monitoring

Written 2026-07-08. Branch `feat/probe-monitoring`. Output of `/code-review
high --comment` run against `git diff main...HEAD` scoped to code files
(`src/`, `scripts/`, `tests/`, `pyproject.toml`). No PR existed for this
branch at review time, so `--comment` had nothing to post to; findings were
printed instead. Method: 8 independent finder angles (line-by-line scan,
removed-behavior audit, cross-file tracer, reuse, simplification, efficiency,
altitude, CLAUDE.md conventions), each run as its own agent, then every
surviving candidate re-checked by a separate one-vote verifier agent
(CONFIRMED / PLAUSIBLE / REFUTED). One candidate — rigidity/footprint gates
not being wired into `PoseFitter`/`scripts/run_pipeline.py` — was REFUTED:
`docs/handoffs/2026-07-07-off-probe-track-gate-strategy.md` and
`docs/handoffs/2026-07-07-rigidity-footprint-gates-validated.md` both
document this as an intentional scope decision ("the monitor is the
deliverable"), not an oversight, so it's omitted below.

## How to use this

Intended to be fixed one finding at a time, each in its own fresh session.
Each item below is self-contained: file/line, what's wrong, and a concrete
scenario that breaks. Ranked most-severe first (correctness bugs in
production code paths first, then diagnostic-script bugs, then cleanup).

## Findings

### 1. [FIXED] Alignment fit dilution via disambiguated hits
**File:** `src/monrad/alignment/accumulator.py:248`
`AlignmentAccumulator.add()` now calls `disambiguate_telescope_hits()` (new,
in `src/monrad/reconstruction/candidates.py`) *before* the golden/cluster
quality gate. Disambiguation picks the recovered candidate by minimizing
distance to the same two-plane straight-line predictor that
`fit_telescope_alignment` later uses to estimate that plane's offset/tilt
(same `(j1,j2)` pairing, same formula — see `accumulator.py:151-159` vs.
`candidates.py:161-206`).
**Failure scenario:** DESIGN.md documents ~83% of real telescope hits as
unresolved, so this is the majority path, not an edge case. For a plane with
a genuine small physical offset, disambiguation systematically manufactures
near-zero residuals against the no-misalignment null hypothesis, diluting
the alignment sample and pulling the fitted offset toward zero — potentially
flipping `AlignmentCorrection.needs_correction` from `True` to `False` and
silently leaving real telescope misalignment uncorrected. No test exercises
`AlignmentAccumulator` with unresolved-then-disambiguated hits alongside an
injected misalignment (`tests/test_stage4.py`'s statistical basis assumes
all-golden hits; `TestDisambiguateHits` in `tests/test_stage3.py` only tests
the standalone function in isolation).
**Likely fix direction:** either exclude disambiguated hits from the
alignment fit's residual sample (only naturally-resolved hits should vote on
misalignment), or weight/flag them distinctly so `fit_telescope_alignment`
can down-weight or ignore them.
**Fix applied (2026-07-08):** `AlignmentAccumulator.add()` now records,
per event, which of the 3 planes were replaced by
`disambiguate_telescope_hits()` (identity check against the pre-disambiguation
hits) and passes that mask through to `fit_telescope_alignment(hits, z_tel,
disambiguated)`. For each plane k, events where plane k itself was
disambiguated are excluded from that plane's own two-plane residual sample
(dx/dy, rotation, and — for the middle plane — delta_z/tilt), since those
positions were manufactured from the exact same `(j1, j2)` predictor used to
compute the residual. Disambiguated events remain usable as `j1`/`j2` input
for other planes' fits. Added `TestDisambiguationExclusion` in
`tests/test_stage4.py`, which builds a 1000-event batch (100 naturally-golden
+ 900 disambiguated, mirroring DESIGN.md's ~83% unresolved rate) with a 5/3mm
injected middle-plane offset and confirms the fitted offset still recovers
the true value; verified this test fails against the pre-fix code (diluted
estimate ≈0.5/0.27mm) and passes with the fix.

### 2. [FIXED] Unvalidated CLI floors crash on `fit_probe_pose`'s hard minimum
**Files:** `src/monrad/monitor/timeseries.py:498` (`--min-fit`),
`src/monrad/monitor/resolution.py:1146` (`--n`)
Neither CLI argument validates a floor against `fit_probe_pose`'s hard
`len(coincs) >= 3` requirement (`src/monrad/pose/optimize.py:368-369`,
`ValueError` otherwise), and neither call site wraps the fit in
try/except.
**Failure scenario:** `monrad-monitor --min-fit 2 ...` crashes with an
uncaught `ValueError` on the first window. `monrad-resolution --n 1 2 300
--z-p 300 ...` crashes partway through a sweep — after already generating
synthetic data for that geometry — and since `write_sweep_csv` only runs
once after the *entire* geometry loop, all previously-computed sweep results
for the run are lost.
**Likely fix direction:** clamp/validate both CLI args to `>= 3` (or
`_MIN_COINCS`, `src/monrad/pose/optimize.py:20`) at parse time with a clear
error message, rather than letting the exception surface from deep inside
the fit.
**Fix applied (2026-07-08):** Exported `_MIN_COINCS` from `monrad.pose`
(`src/monrad/pose/__init__.py`). `timeseries._parse_args` now calls
`parser.error(...)` if `--min-fit < _MIN_COINCS`; `resolution._parse_args`
does the same if any `--n` value is below the floor — both fail fast at
parse time with a clear message, before any data generation or streaming
starts. Added `test_cli_min_fit_below_floor_rejected`
(`tests/test_monitor_timeseries.py`) and `test_cli_n_below_floor_rejected`
(`tests/test_monitor_resolution.py`) asserting `SystemExit`.

### 3. [FIXED] Trailing batch at end-of-stream dropped with no log
**File:** `src/monrad/monitor/timeseries.py:369` (end of `monitor_probe`'s
`for co in stream:` loop)
When the coincidence stream ends with a non-empty, non-passing trailing
`batch`, the loop just ends — no `logger.warning` fires, unlike the
`raw_cap` overflow branch a few lines up (`timeseries.py:355`). The module
docstring (`timeseries.py:26-28`) claims the trailing remainder is "dropped
the same way" as the logged raw_cap path.
**Failure scenario:** An acquisition whose final window never reaches
`min_fit` survivors before the stream ends has that data silently dropped
with zero diagnostic trace — an operator watching `monrad-monitor` output
can't tell "last few minutes were dropped" from "still accumulating."
**Likely fix direction:** add a `logger.warning` after the loop for a
non-empty leftover `batch`, mirroring the raw_cap branch's message.
**Fix applied (2026-07-08):** Added a `logger.warning` right after the
`for co in stream:` loop in `monitor_probe`, firing whenever a non-empty
`batch` remains once the stream is exhausted. It reports the trailing
window's `utc_start`/`utc_end` (from `win_start_ns` and the last buffered
coincidence's `t_ns`), the raw coincidence count, and the configured
`min_fit`, mirroring the wording of the existing raw_cap-abandoned-window
warning. Added `test_trailing_batch_logs_warning`
(`tests/test_monitor_timeseries.py`), which drives `monitor_probe` with
`min_fit=10_000_000` (guaranteeing every coincidence ends up in one
never-closing trailing batch) and asserts a "trailing window" warning is
captured via `caplog`.

### 4. [FIXED] `monitor_probe` retains full `PoseResult` per closed window, unbounded RAM growth
**File:** `src/monrad/monitor/timeseries.py:295` (append into `results`),
`:224` (`results` init), `:388` (return); `WindowResult` fields at
`:103-120` (esp. `pose: PoseResult` at `:120`); `PoseResult` definition at
`src/monrad/pose/types.py:91-119`.
`monitor_probe` is a plain function returning `list[WindowResult]`, not a
generator. Every closed window's full `PoseResult` — including
`inliers`/`outliers` (`list[Coincidence]`, essentially the whole raw batch
for that fit) and a `(360,2)` `chi2_curve` array — is kept for the life of
the call, contradicting the docstring's claim that "only the open batch is
ever buffered in RAM."
**Failure scenario:** a long monitoring session with many windows grows
memory linearly and unboundedly with run length, contrary to the documented
bounded-RAM design and the project's streaming-architecture invariant.
**Likely fix direction:** either don't store the full `PoseResult` on
`WindowResult` (extract only the scalar fields already duplicated there:
t_x, t_y, z_p, theta, sigmas, resid_rms) and drop `inliers`/`outliers`
before appending, or turn `monitor_probe` into a generator so callers can
choose whether to retain history.
**Fix applied (2026-07-08):** Removed the `pose: PoseResult` field from
`WindowResult` (`src/monrad/monitor/timeseries.py`) — nothing outside its own
construction ever read it, since all consumers (CSV writer, plotter, CLI
summary, tests) already used the scalar fields duplicated alongside it.
`_fit_and_record` no longer passes `pose=pose` into the stored result, so a
window's `PoseResult` (and its `inliers`/`outliers` coincidence lists plus
the `(360,2)` `chi2_curve`) is now garbage-collected once that window's
scalars are extracted, leaving `results: list[WindowResult]` growing only
with bounded per-window scalars. Added
`test_window_result_holds_only_scalars` (`tests/test_monitor_timeseries.py`),
which asserts no `WindowResult` field is typed `PoseResult` and every stored
value is a plain scalar or datetime.

### 5. [CONFIRMED] Cold-start rigidity gate reruns full O(n²)/O(n³) fit on every appended coincidence
**File:** `src/monrad/monitor/timeseries.py:251` (`_run_gates`, cold-start
branch), docstring at `:196-200` (self-acknowledges the issue); `_run_gates`
called from the batching loop at `:339-368`; `filter_rigidity` at
`src/monrad/pose/optimize.py:253-333` (O(n²) pairwise matrix, rebuilt from
scratch every call, `:283-292`).
While `prev_pose is None` (before the first window ever closes), every
single newly-appended coincidence triggers a full `fit_probe_pose` call
(coarse+fine θ scan, LM polish, Mahalanobis refit) just to get a `z_ref`
anchor for the rigidity gate, plus a full O(n²) pairwise-distance rebuild in
`filter_rigidity`. No caching or "every K coincidences" throttle exists.
**Failure scenario:** a contaminated first window that never clears
`min_fit` survivors reruns both at every batch size from `min_fit` up to
`raw_cap` (`RAW_CAP_MULTIPLIER * min_fit`, default 5x) — roughly O(raw_cap³)
total work for one window. The code's own docstring already admits this is
"the one place this can get noticeably slower than before."
**Likely fix direction:** cache the bootstrap `z_ref` across growth steps
within the same cold-start window (only recompute periodically, or use a
cheap closed-form estimator instead of the full nonlinear fit for this
throwaway anchor value).

### 6. [CONFIRMED] `rng.choice` crash on small population (diagnostic script)
**File:** `scripts/diagnostics/wide_block_inspect.py:30`
`rng.choice(narrow_all, 400, replace=False)` hardcodes a sample size of 400
with no bounds check against `len(narrow_all)`.
**Failure scenario:** any dataset or re-run acquisition producing fewer than
400 narrow (`|b|<=0.5`) coincidences crashes the script with numpy's "Cannot
take a larger sample than population when replace is False" instead of
degrading gracefully.
**Likely fix direction:** `min(400, len(narrow_all))`, or an early
assertion with a clear message.

### 7. [PLAUSIBLE] Silent negative-gap filtering in timing diagnostic (diagnostic script)
**File:** `scripts/diagnostics/tel_time_inspect.py:43`
`gaps = gaps[gaps >= 0]` silently discards non-monotonic CLK gaps with no
count or warning, even though the comment at `:39` ("guard against wrap /
non-monotonic") shows the code anticipates this exact failure mode.
**Failure scenario:** a non-monotonic CLK sequence — itself a real
data-quality signal for a script whose whole purpose is timing inspection —
is thrown away silently instead of surfaced. (A related claim that
truncated `_GPS.bin` files would silently read garbage was checked and
refuted during verification: `np.frombuffer(..., count=n)` raises a hard
`ValueError` on truncation instead, so that half of the original candidate
does not hold.)
**Likely fix direction:** print/log the count of dropped negative gaps
rather than silently filtering.

### 8. [CONFIRMED] `Z_TEL` hardcoded identically in three diagnostic scripts
**Files:** `scripts/diagnostics/coinc_b_dist.py:23`,
`scripts/diagnostics/coinc_dt.py:25`,
`scripts/diagnostics/tel_raw_inspect.py:27`
`Z_TEL = np.array([0.0, -1340.0, -670.0])` is a bare module constant in all
three, instead of a parameter — unlike `scripts/run_pipeline.py`'s
`--z-tel` flag and `monrad.monitor.io.fit_alignment(z_tel=...)`.
**Failure scenario:** one script's own comment notes this z-order is
dataset-specific and non-obvious. If this dataset's column-to-z mapping is
ever corrected, all three files need the identical edit; missing one
silently reintroduces the exact telescope z-order/tilt mixup this project's
alignment work exists to catch (see [[telescope-tilt-not-zrotation]]).
**Likely fix direction:** factor `Z_TEL` into a shared constant/CLI arg
these scripts import, or at minimum a single shared diagnostics config
module.

### 9. [CONFIRMED] Diagnostic scripts reimplement canonical decoders instead of importing them
**Files:** `scripts/diagnostics/tel_raw_inspect.py:47`,
`scripts/diagnostics/tel_plane_inspect.py` (same pattern, ~line 28) — both
hand-roll `.bin` header parsing/reshaping instead of calling
`BinDecoder.read()` (`src/monrad/decoders/position.py:63-87`) despite
already importing `BinDecoder` for its constants.
`scripts/diagnostics/tel_time_inspect.py` (~lines 20-32) redefines
`GPS_CLK_MASK`/`GPS_GEN_SHIFT`/`GPS_FLAG_SHIFT` and reimplements bit
unpacking instead of importing from `src/monrad/decoders/gps.py:18-30`.
**Failure scenario:** if the on-disk position-file or GPS-word format ever
changes, the canonical decoders get updated but these scripts keep decoding
with the stale hardcoded layout/masks, silently producing wrong values with
nothing importing the canonical modules to catch the drift.
**Likely fix direction:** replace the hand-rolled parsing with
`BinDecoder(path).read()` and imports from `monrad.decoders.gps`.

### 10. [CONFIRMED] Stale module docstring in `pose/optimize.py`
**File:** `src/monrad/pose/optimize.py:5`
Still describes `fit_probe_pose` as including "an opt-in absolute-mm
residual cut layered after the Mahalanobis refit." That cut was removed by
an earlier commit on this branch
(`docs/handoffs/2026-07-07-abs-resid-cut-removed.md`) and no such machinery
remains in the function body — the only remaining `max_resid_mm` logic is
`filter_rigidity`, a separate *pre-fit* gate applied before `fit_probe_pose`
even runs, not an in-fit cut.
**Failure scenario:** a reader relying on the docstring to understand the
fit pipeline will look for code that isn't there.
**Likely fix direction:** revert to the pre-diff wording ("...the
Mahalanobis outlier cut and stratified-half consistency test.") or describe
where the gating logic actually lives now (`monitor/timeseries.py`'s
`_run_gates`).

## Not included (refuted during verification)

- **Rigidity/footprint gates not wired into `PoseFitter`/`run_pipeline.py`**
  — checked against `docs/handoffs/2026-07-07-off-probe-track-gate-strategy.md`
  and `docs/handoffs/2026-07-07-rigidity-footprint-gates-validated.md`, both
  of which explicitly document this as intentional ("the monitor is the
  deliverable; do NOT change `PoseFitter`'s default behaviour"). Not a bug.

## Suggested skills

- `verify` — for confirming each fix behaves as intended before moving to
  the next finding.
- `astral:ruff` — lint/format after each edit.
