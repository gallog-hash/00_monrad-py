# Code review findings to fix — PR #14 (feat/multiprobe-monitoring)

Written 2026-07-10. PR: https://github.com/gallog-hash/00_monrad-py/pull/14
(`feat/multiprobe-monitoring` → `main`, currently at commit `23c826d`).
Bundles 4 commits: multi-probe pose monitoring (Step 3), the `--n-probe-ch`
footprint-exceeded warning, making the fiber×ribbon combine factor `N` a
per-detector parameter (`n_fibers_per_ribbon`/`--fibers-per-ribbon`, default
10 — see `docs/handoffs/2026-07-10-per-probe-fibers-per-ribbon.md` for the
original design), and a README doc fix.

Method: `/code-review` at `high` effort against `git diff main...HEAD` for
PR #14. 8 independent finder angles (line-by-line scan, removed-behavior
audit, cross-file tracer, reuse, simplification, efficiency, altitude,
CLAUDE.md conventions), each run as its own agent. Candidates deduped, then
every survivor re-checked by a separate one-vote verifier agent — all 10
that reached verification came back **CONFIRMED** (none REFUTED). Full
finder/verifier transcripts are in this conversation, not reproduced here.

## How to use this

Fix one finding at a time, each in its own commit (or its own session, per
[[separate-plan-and-execute-sessions]] if the fix is non-trivial). Ranked
most-severe first: silent-corruption/crash bugs in production code paths
first, then diagnostic-tool inconsistency, then test-coverage gaps, then
cleanup/efficiency. After each fix, run the full suite
(`uv run pytest -q`, ~251 tests, takes ~8 min) plus
`uv run ruff check . && uv run ruff format --check .` before moving on.

## Findings

### 1. Synthetic generator can corrupt bits when `n_probe_ch` doesn't fit `10×N` — FIXED (`aa8ef32`)
**File:** `src/monrad/synthetic/generate.py:187` (`_ch_to_u64`)
`r_y, f_y = c_y // n, c_y % n` has no bound check that `c_y < 10 * n`. With
`generate(prb_dir, n_probe_ch=100, n_probe_fibers_per_ribbon=3)`, a hit
quantized to channel 99 gives `r_y = 99 // 3 = 33`, so `y_rib = 1 << 33`
lands inside the X-ribbon field (bits 32-41) once OR'd into the u64 word,
silently corrupting the X coordinate of synthetic test/fixture data instead
of raising an error.
**Failure scenario:** any future test or fixture generated with a
`n_probe_ch`/`n_probe_fibers_per_ribbon` combination where
`n_probe_ch > 10 * n_probe_fibers_per_ribbon` produces garbage ground truth
with no error — hard to debug since the corruption is silent.
**Likely fix direction:** `generate()` (or `_ch_to_u64` itself) should
`raise ValueError` if `n_probe_ch > 10 * n_probe_fibers_per_ribbon`
(the true max channel range for combine factor `n`).

### 2. No cross-validation between a probe's `n_probe_ch` and `fibers_per_ribbon` — FIXED (`a656569`)
**File:** `src/monrad/monitor/multiprobe.py:112` (`monitor_probes`)
Nothing ties a probe's `n_probe_ch` to its `fibers_per_ribbon`, even though
the real channel range is bounded by `10 * N`.
**Failure scenario:** `--n-probe-ch 40 --fibers-per-ribbon 10` for a probe
actually wired at N=4: `decode_position` silently aliases distinct physical
channels for hits with `ribbon > 0`. Some land outside the footprint and
trigger the existing overflow warning (misdiagnosed as "channel count too
small" — see finding 9), but others alias into in-range-but-wrong channels
with **zero** warning, silently biasing the fitted pose.
**Likely fix direction:** in `monitor_probes()` (and/or `monitor_probe`
singular), validate `n_probe_ch[k] <= 10 * fibers_per_ribbon[k]` per probe at
call time and raise a clear `ValueError` — this can't be fully solved (the
user could still supply a *wrong-but-plausible* N), but it catches the
class of error where the two flags are inconsistent with each other.

### 3. Unguarded `divmod` on `--fibers-per-ribbon 0` crashes deep in decode — FIXED (`15f8dcb`)
**File:** `src/monrad/decoders/position.py:52` (`split_channel`)
`split_channel(channel, n)` does `divmod(channel, n)` with no lower-bound
check on `n`. Neither `timeseries.py`'s nor `multiprobe.py`'s argparse setup
validates `--fibers-per-ribbon >= 1` (unlike the existing `--min-fit` floor
check against `_MIN_COINCS`).
**Failure scenario:** `monrad-monitor --fibers-per-ribbon 0 --tot-weights`
against data with any cluster-quality (width>1) probe hit:
`_tot_weighted_centroid` calls `split_channel(ch, 0)` → `divmod(ch, 0)` →
`ZeroDivisionError` deep in decode, instead of a clean argparse error at
parse time.
**Likely fix direction:** add a floor validation (`>= 1`, arguably `<= 10`
too since a probe can't wire more than the raw 10 fiber positions) in both
`timeseries.py::_parse_args` and `multiprobe.py::_parse_args`, mirroring the
`--min-fit` pattern.

### 4. `scripts/run_pipeline.py` never got the new parameter at all — FIXED
**File:** `scripts/run_pipeline.py:593` (`PoseFitter(...)` construction)
No `--fibers-per-ribbon` (or `--n-probe-ch`) CLI flag exists in this script,
and its `PoseFitter(...)` call omits `prb_fibers_per_ribbon`.
**Failure scenario:** a user follows CLAUDE.md's documented
`python scripts/run_pipeline.py --telescope ... --probe ...` entry point
against a probe wired with N≠10 (the exact real-world case this PR targets):
`decode_position` silently uses the N=10 default inside `PoseFitter`,
systematically mis-mapping probe channels with no error or warning, while
the sibling `monrad-monitor`/`monrad-multiprobe` CLIs added in this same PR
decode the same data correctly.
**Likely fix direction:** add `--fibers-per-ribbon` (scalar, default 10) to
`run_pipeline.py`'s parser and pass it as `prb_fibers_per_ribbon=` into the
`PoseFitter` construction.

### 5. `monrad-decode-bin --or` debug tool hardcodes N=10, now inconsistent — FIXED
**File:** `src/monrad/decoders/position.py:294` (`BinDecoder.or_visual`)
The debug/inspection tool hardcodes `N = POS_HALF_BITS` with no way to pass
a custom N via the `--or` CLI flag (that flag's `N` argument means "limit
output to N GEN groups," unrelated to fibers-per-ribbon).
**Failure scenario:** `monrad-decode-bin data/.../probe.bin --or 5` against
a probe with real N=4 (e.g. the testLab 40-channel probe — see
[[testlab-20210723-probe-size]]) reports a different golden/cluster channel
than the actual pipeline decode would for the identical raw word — misleading
for exactly the probes this PR was built to support.
**Likely fix direction:** add an optional `n_fibers_per_ribbon` parameter to
`or_visual`/`analyze` and a corresponding CLI flag on `monrad-decode-bin`
(name it something that doesn't collide with the existing `--or` group-limit
flag).

### 6. Footprint-overflow warning blames only `n_probe_ch`, never mentions `fibers_per_ribbon` — FIXED
**File:** `src/monrad/monitor/timeseries.py:346` (push-time warning) and
`:446-448` (finalize-time summary, suggests a specific `--n-probe-ch` value)
Both messages attribute any out-of-bounds decoded probe hit solely to
`n_probe_ch` being too small, never mentioning that a wrong
`--fibers-per-ribbon` value (channel aliasing) can produce the identical
symptom.
**Failure scenario:** a user with a correct `--n-probe-ch` but wrong
`--fibers-per-ribbon` sees the overflow warning, follows its suggestion to
raise `--n-probe-ch`, and papers over a combine-factor misconfiguration
instead of fixing the actual root cause.
**Likely fix direction:** reword both messages to name both possible causes
(footprint too small *or* wrong fibers-per-ribbon), rather than only the
former.

### 7. New tests only cover golden hits, missing the actual regression case — FIXED (`d1435b7`)
**File:** `tests/test_stage3.py:133` (`TestFibersPerRibbon`)
All three new tests build a golden (width=1) hit at non-default N. No test
combines a cluster-width (width>1) hit with non-default N — the only path
that exercises the `bit_counts[:POS_HALF_BITS]` vs `bit_counts[n:]` split in
`_decode_axis`/`_axis_candidates_with_tot`, which is the actual latent bug
this PR fixed (see `docs/handoffs/2026-07-10-per-probe-fibers-per-ribbon.md`,
which explicitly flags this exact test as the one that "would have caught
the bit_counts-slicing bug").
**Failure scenario:** a future refactor that re-conflates `POS_HALF_BITS`
and `n` inside the TOT-weighted-centroid branch would pass the full test
suite untouched, silently corrupting TOT-weighted centroids for any real
non-default-N probe with cluster hits.
**Likely fix direction:** add a test encoding a width>1 cluster hit at
non-default N (e.g. N=5) with `tot_weights=True`, asserting the recovered
centroid matches the N=5 encoding and would differ under the N=10 default —
mirroring the existing `test_tot_weights_shifts_cluster_centroid` pattern but
with a non-default combine factor.

### 8. `monrad-resolution` can't exercise the new non-default-N path — FIXED (`11834c1`)
**File:** `src/monrad/monitor/resolution.py:153` (`generate()` /
`stream_coincidences()` calls)
`fibers_per_ribbon`/`n_probe_fibers_per_ribbon` is never threaded through
here, and there's no `--fibers-per-ribbon` CLI flag (nor, actually, a
`--n-probe-ch` flag — `n_probe_ch` is a bare function-default, not wired to
argparse at all).
**Failure scenario:** self-consistent today (both sides implicitly default
to N=10, so no wrong results) — but it's a coverage gap: the σ(N)
characterization sweep this PR's sibling drivers gained has no way to
characterize a probe wired at non-default N.
**Likely fix direction:** thread `fibers_per_ribbon` through
`resolution.py`'s `generate()`/`stream_coincidences()` calls; add a CLI flag
if/when `n_probe_ch` itself gets one.

### 9. Multi-probe monitoring now actually exercises an already-documented inefficiency
**File:** `src/monrad/pose/fitter.py:119` (`PoseFitter._decode_cluster`)
`_decode_cluster`'s combinatorial telescope-track search
(`reconstruct_plane_candidates` + up to 16³ candidate triples via
`itertools.product` + `_fit_triple` chi² minimization) has no memoization
keyed on the shared telescope entry. `multiprobe.py`'s `monitor_probes()` is
the first caller to run N independent `PoseFitter`s against one shared
cluster stream in production — its own module docstring already documents
this exact cost as a "known, deferred inefficiency," but this PR is what
makes it real (previously only a theoretical concern).
**Failure scenario:** for any cluster in coincidence with ≥2 probes at once
(explicitly a real case per the module docstring), the identical
telescope-only search now runs N times per cluster instead of once.
**Likely fix direction:** memoize `(best_fit, best_chi2, best_cands,
tel_quality)` once per cluster (keyed on the telescope `PosRef`/identity),
shared across the per-probe loop in `multiprobe.py` — decode only the
probe-specific part per fitter. This was already proposed (and deferred) in
the module docstring; may be a larger change worth its own session.

### 10. Broadcast/validate-length logic duplicated 4× in `multiprobe.py`
**File:** `src/monrad/monitor/multiprobe.py:102` (`monitor_probes`, for both
`n_probe_ch` and `fibers_per_ribbon`) and `:322-336` (`_parse_args`, same
pattern again via `parser.error` instead of `ValueError`)
The "broadcast singleton to N, else validate length matches `n_probes`, else
raise" pattern is written out 4 times in one file — twice in
`monitor_probes()`, twice in `_parse_args()`. This PR authored all 4 copies
(both `n_probe_ch`'s pair and `fibers_per_ribbon`'s pair are new to this
diff — `multiprobe.py` doesn't exist on `main`).
**Failure scenario:** a future third per-probe override means copy-pasting a
5th/6th near-identical block; a fix to the broadcast rule made in one copy is
easy to miss in the other three.
**Likely fix direction:** extract a shared
`_broadcast_per_probe(name, values, default, n_probes) -> list` helper
(raising plain `ValueError`), called from `monitor_probes()` for both
params, and from `_parse_args()` wrapped in `try/except ValueError as e:
parser.error(str(e))`.

## Suggested skills

- `verify` — for confirming each fix behaves as intended (drive the actual
  CLI/decode path, not just tests) before moving to the next finding.
- `astral:ruff` — lint/format after each edit.
- `/code-review` (low or medium effort) on the final diff before opening a
  follow-up PR, to catch anything the fixes themselves introduce.
