# Plan — finding 9: memoize telescope-track search across probes

Written 2026-07-13. Prior work: PR #14
(`feat/multiprobe-monitoring` → `main`) merged (squash) as `main@61d60b5`;
both local and remote `feat/multiprobe-monitoring` branches are deleted.
Findings 1–8 and 10 from
`docs/handoffs/2026-07-10-fibers-per-ribbon-pr-review-findings.md` are fixed
on `main`. Finding 9 (below) was deliberately deferred out of that PR as a
non-blocking efficiency issue and is the sole open item from that review.

This doc is a starting plan for a fresh session, not a final design — the
"recommended approach" section reflects a simplification found by reading
the current code (see below), not what finding 9's original writeup assumed.
Confirm it still holds before implementing.

## Problem (finding 9, unchanged from the original review)

`PoseFitter._decode_cluster` (`src/monrad/pose/fitter.py:119-286`) runs a
combinatorial telescope-track search (`reconstruct_plane_candidates` +
up to 16³ candidate triples via `itertools.product` + `_fit_triple` χ²
minimization, lines 173-234) for every cluster. `multiprobe.py`'s
`monitor_probes()` (`src/monrad/monitor/multiprobe.py:181-185`) is the first
production caller that runs N independent `PoseFitter`s against one shared
cluster stream:

```python
for cluster in build_cluster_stream(tel, probes):
    for fitter, acc in zip(fitters, accumulators):
        co = fitter.decode_cluster(cluster)
        ...
```

For any cluster in coincidence with ≥2 probes at once, the identical
telescope-only search runs N times instead of once. Already flagged as a
"known, deferred inefficiency" in the `multiprobe.py` module docstring
(lines 24-30) before this PR made it real.

## What's actually shared across fitters (read from current code)

In `monitor_probes()` (`multiprobe.py:143-163`), every `PoseFitter` is built
from:
- the same `tel_z`, `alignment` (one `fit_alignment(...)` call, shared
  object), `tel_id=0`, `tel_pos_paths=tel.pos_paths`
- the same `tot_thresh`, `tot_weights`, `min_anchor_planes` — these are
  single scalars applied to every fitter (module docstring: "Gate
  thresholds ... apply identically to every probe"), *not* per-probe lists
  like `n_probe_ch`/`fibers_per_ribbon`

That means the telescope-side portion of `_decode_cluster` — everything
from `tel_entries`/candidate reconstruction (line 173) through
`tel_quality` (line 244) — is provably identical across all fitters for a
given cluster, given today's `monitor_probes` call pattern. Only the
probe-side portion (`prb_refs` extraction, `decode_position` on
`prb_ref` with the *per-probe* `prb_fibers_per_ribbon`, quality gate,
`Coincidence` construction) genuinely differs per fitter.

## Two caveats to check before implementing

1. **`alignment` is mutable per-fitter.** `PoseFitter.update_alignment()`
   (`fitter.py:81-82`) lets a caller replace one fitter's alignment
   independently. `multiprobe.py` never calls it today (confirmed by grep),
   but any shared-computation design must either assert/require that all
   fitters' `alignment` stay identical objects, or key the shared cache on
   `id(alignment)` (or an equality check) rather than assuming it forever.
2. **`on_decode` reporting is per-fitter.** `_decode_cluster`'s internal
   `_report()` closure calls `self.on_decode` for every accept/reject
   reason (`ambiguous_cluster`, `zero_candidate_plane`, `no_anchor_plane`,
   `chi2_track_cut`, `accepted`). `multiprobe.py` never sets `on_decode`
   today, but a shared-computation design must still fire each fitter's own
   `on_decode` (if set) with the shared telescope-side reason/counts —
   don't silently drop reporting for fitters 2..N just because fitter 1
   computed the result.

## Recommended approach

The module docstring's "memoize keyed on cluster identity" framing implies
a cache data structure (dict, LRU, etc.), but `monitor_probes()` already
visits one cluster at a time in its outer loop — there is no need to
persist anything across iterations. The simpler shape:

1. Extract the telescope-only portion of `_decode_cluster` (lines 156-244:
   `tel_entries`/`prb_refs` length-1 gate on the *telescope* side only,
   `reconstruct_plane_candidates`, zero-candidate gate, anchor gate,
   triple search, `tel_quality`) into a standalone method, e.g.
   `PoseFitter._decode_telescope_track(cluster) -> TelescopeTrackResult |
   RejectionReport`, parameterized only by what the telescope side needs
   (`self.tel_id`, `self.tel_z`, `self.tel_pos_paths`, `self.alignment`,
   `self.tot_thresh`, `self.tot_weights`, `self.min_anchor_planes`).
2. Add a `PoseFitter` method that accepts a precomputed telescope result
   and does only the probe-side work (extract `prb_refs`, decode with
   `self.prb_fibers_per_ribbon`, quality gate, build `Coincidence`, fire
   `self.on_decode` with the combined reason). `_decode_cluster` itself
   keeps calling both steps in sequence — so every *existing* single-fitter
   caller (`timeseries.py::monitor_probe`, `resolution.py`,
   `scripts/run_pipeline.py`) is unaffected.
3. In `monitor_probes()`'s inner loop, compute the telescope result once
   per cluster (e.g. via `fitters[0]`, after asserting all fitters share
   the same `tel_id`/`alignment`/etc. — or just require it as a
   precondition of the new shared-path helper), then pass it to every
   fitter's probe-side method instead of calling `decode_cluster` N times.

This keeps `_decode_cluster`'s single-fitter contract and tests untouched
and adds an opt-in shared path used only by `multiprobe.py`.

## Test plan

- Unit test comparing the shared-computation path against N independent
  `decode_cluster()` calls on the same synthetic cluster (2+ probes),
  asserting identical `Coincidence` values and identical `on_decode`
  reports per fitter — regression guard that hoisting the computation
  doesn't change any observable behavior.
- A call-count assertion (e.g. monkeypatch/count calls to
  `reconstruct_plane_candidates` or `_fit_triple`) showing N probes now
  cost 1 telescope search per cluster instead of N.
- Full suite (`uv run pytest -q`) — should stay at the current pass count
  with additions, no regressions in `tests/test_monitor_multiprobe.py` or
  `tests/test_stage5.py`.
- `uv run ruff check . && uv run ruff format --check .`

## Suggested skills

- `verify` — after implementing, drive `monrad-multiprobe` against
  synthetic multi-probe data and confirm per-probe results are unchanged
  from before the optimization (not just unit tests).
- `astral:ruff` — lint/format after each edit.
- `/code-review` (medium or high effort) on the resulting diff before
  opening a PR — this touches a shared streaming hot path, worth an
  independent pass.
- [[separate-plan-and-execute-sessions]] — this doc is the plan; execute
  it in a fresh session rather than continuing in the one that wrote it.
