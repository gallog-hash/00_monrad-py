# Session handoff — 2026-06-11 (RESOLVED)

## Goal

Fix double-counting in the stage 2 → stage 5 pipeline that inflated
`n_inliers` and over-tightened the pose-fit covariance.

## Root cause (corrected understanding)

`stage2.coincidence_stream` emitted the *entire* sliding deque on every
heap pop that left ≥ 2 detectors in the window. This produced
overlapping, growing clusters, so the same event was reported in many
clusters. `PoseFitter._decode_cluster` then did a last-wins scan, so one
probe hit was paired with several telescope tracks across successive
yields and accumulated multiple times.

The original handoff proposed an `id()`-based dedup in stage 5. That was
**rejected** for two reasons found this session:

1. It doesn't fix the described bug — last-wins picks a *different*
   `tel_ref` on each growing yield, so `(id(tel_ref), id(prb_ref))` keys
   differ and dedup never fires.
2. `id()` is unsafe — `PosRef` objects *are* GC'd once evicted from the
   deque, so addresses get reused.

DESIGN.md §5.1 is authoritative and already specifies the correct
behaviour: **emit each transitive-closure cluster exactly once, when it
closes.** Stage 4 does *not* consume `coincidence_stream` (it uses
`reconstruct_stream` directly), so the handoff's worry about collateral
damage to stage 4 was unfounded. The fix belongs in stage 2.

## What was changed

1. `src/monrad/stage2.py` — `coincidence_stream` now builds disjoint
   transitive-closure clusters and yields each once, on close (gap to the
   next popped event > `window_ns`), plus a final flush. Every event
   appears in at most one cluster.

2. `src/monrad/stage5.py` — `_decode_cluster` now requires **exactly one**
   telescope event and **exactly one** event from *this* probe; ambiguous
   clusters are rejected (return `None`) instead of last-wins. Events from
   other probe detectors are ignored, so one telescope event can still be
   in coincidence with several distinct probes (one `PoseFitter` each).
   No `_seen` dedup was added — the stage-2 fix removes the root cause.

3. Tests added:
   - `tests/test_stage2.py::TestTransitiveClosure` — in-memory streams that
     trigger the high-rate growing-window scenario (the file-based synth
     data never does, since tracks are ~10 ms apart). Asserts disjoint,
     once-only clusters and window-edge behaviour.
   - `tests/test_stage5.py::TestDecodeClusterDisambiguation` — the
     one-tel/one-probe guard short-circuits before file decode.

## State

All 120 tests pass; ruff clean. Done.

---

# Session handoff — 2026-06-11 (corner-probe edge-case audit)

## Goal

Lab-testing scenario: a single 30 cm probe sits on a *corner* of the
telescope's top plane (z_p ≈ 0, mounted nearly flat).  Question: how does
the pipeline handle a probe event that coincides with *two* telescope
tracks (one being the muon that actually crossed the probe), plus the
other edge cases real sea-level muon flux produces?

## What was added

`tests/test_corner_probe_edge_cases.py` — builds a synthetic corner-probe
dataset (reusing the byte-level writers in `monrad.synth`) that injects
every plausible edge case as a *labelled* event, runs the real stage 1-5
pipeline, and asserts the handling of each.  12 tests; full suite now 132.

## How "two telescope tracks in one coincidence" is handled

The telescope is one detector sampled in 80 ns (16-row) windows, so this
arrives two ways and is rejected at two different points — the pipeline
**never tries to pick which track went through the probe**:

- **Same 80 ns window** → one telescope event whose per-plane OR-mask
  superimposes both muons → 2 ribbon + 2 fiber bits → `_reconstruct_coord`
  returns `None` → quality `'unresolved'` → `stage5._decode_cluster`
  rejects at the quality gate.
- **Different windows, <200 ns apart** → two telescope events → stage-2
  transitive-closure cluster has `2 tel + 1 prb` → `_decode_cluster`'s
  `len(tel_refs) != 1` guard rejects the *whole* coincidence (the genuine
  pairing is lost too). This is the deliberate purity-over-efficiency
  choice from the `44ef3ed` double-counting fix.

## Per-case handling (all asserted)

| Case | Outcome | Caught at |
|---|---|---|
| E1 genuine | accepted (~85%) | — |
| E2 pile-up, non-adjacent | rejected, tel `unresolved` | stage-5 quality gate |
| E3 pile-up, adjacent | **accepted** as blurred `cluster`, ~5 mm bias | *slips through* |
| E4 two-window double track | rejected, `2 tel + 1 prb` | stage-5 count gate |
| E5 accidental | accepted, then Mahalanobis outlier (d≈230 vs genuine max 3.2) | stage-5 outlier cut |
| E6 probe-only | no cluster | stage 2 |
| E7 telescope-only | no cluster (stage-4 only) | stage 2 |
| E8 charge-share | accepted as `cluster` | — |
| E9 probe noise | rejected, probe `unresolved` | stage-5 quality gate |
| E10 probe invalid | rejected, probe `invalid` | validity prefilter |

## Findings worth acting on later

1. **~15% of genuine corner tracks fail the χ²<4 line cut.** Strip
   quantization on near-vertical tracks alone trips it, *amplified*
   because `stage5._tel_line_fit` applies plane-0's golden σ uniformly to
   all three planes (`stage5.py:446,449`) — a `cluster` plane's larger σ
   is ignored. Real efficiency cost, not a correctness bug. Consider
   passing per-plane σ into the line fit.

2. **`z_p` is the soft direction at the corner.** With z_p≈0 and
   near-vertical tracks the slope leverage is small (DESIGN §8.6); the fit
   gives σ_zp ≈ 2× σ_tx. A single high-leverage accidental can then run
   away along z_p and **defeat the one-pass Mahalanobis cut** when genuine
   statistics are thin: at N_GOOD=200 the accidental drags z_p 16→3 mm and
   the cut trims the wrong points; by N_GOOD≈250 dilution restores clean
   recovery. The test asserts the deterministic facts (the accidental is
   cleanly *separable* — d≈230 — and z_p is the softest parameter) rather
   than the fragile mixed-fit recovery. Mitigations: accumulate enough
   coincidences before trusting z_p, seed/constrain z_p from a tape
   measure (DESIGN §8.6 already suggests this), or make the Mahalanobis
   cut iterative instead of one-pass.

3. **`_decode_cluster` does not call `disambiguate_telescope_hits`** — a
   coincidence telescope hit that decodes as `unresolved` is dropped, not
   recovered via the two-plane predictor that stage-3 offers. Possible
   efficiency win if accidentals stay rare.

## State

132 tests pass; ruff clean. Findings are advisory — no code changed in
`src/` this session.
