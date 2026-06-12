# Session handoff — 2026-06-12 (per-plane / per-axis σ in stage 5)

## Goal

Make stage 5's weighting comply with DESIGN.md §6.4 / §8.2 / §8.3: use the
**per-plane, per-axis** position uncertainties that `stage3.Hit` already
carries, instead of collapsing them to a single broadcast scalar. This is
the fix for finding #1 of the 2026-06-11 corner-probe audit (below):
`_tel_line_fit` applied plane-0's `sigma_x` uniformly to all 3 planes and
both axes, so a `cluster` plane's larger σ was ignored, mis-scaling the
line covariance and the χ²<4 track cut.

## Origin of the task

Started from a code-reading question: "is it correct for `_tel_line_fit`
to expect `sigma_hit` to be a float?" Answer: no. DESIGN.md §8.2 says
`Σ_line` is "derived from the **per-plane** position uncertainties (§6.4)",
and §6.4 computes σ per axis (`σ = cluster_width·strip/√12`). The scalar
only cancelled for the line *point estimates*; it wrong-scaled `cov_ab` and
χ² whenever plane/axis widths differed.

## What was changed (all in this session)

Done in three rounds, each requested and verified:

1. **Telescope per-plane/per-axis** (`src/monrad/stage5.py`):
   - `_tel_line_fit` now takes `sigma_x`, `sigma_y` (each scalar **or**
     `(n,)` array), builds independent diagonal weight matrices per axis,
     and returns genuinely distinct `cov_x` / `cov_y`. χ² is the weighted
     sum `Σ wₖrₖ²` over both axes.
   - `Coincidence.cov_ab` split into `cov_ab_x` / `cov_ab_y`.
   - `_decode_cluster` builds `sigma_x_arr` / `sigma_y_arr` from all 3
     planes (was `tel_hits[0].sigma_x`) and stores both covariances.
   - `_weighted_residuals` + Mahalanobis cut use the per-axis covariances.

2. **Probe per-axis** (`src/monrad/stage5.py`):
   - `Coincidence.sigma_prb` split into `sigma_prb_x` / `sigma_prb_y`
     (from `prb_hit.sigma_x` / `.sigma_y`).
   - `_linear_solve_fixed_theta` **substantive change**: it previously took
     a single `sigma_prb` used only as a uniform post-hoc χ² divisor — the
     θ-scan was effectively *unweighted*. It now weights each row of the 2N
     system by its per-axis probe weight `1/σ_prb,{x,y}` (`Aw = A·√w`,
     `bw = b·√w`, χ² = weighted SSR). The scalar param is gone. **This
     changes the recovered θ/pose on data with mixed golden/cluster probe
     hits** — correctly, not a regression.
   - Removed `sigma_prb = coincs[0].sigma_prb` and threaded the param out
     of all 3 call sites (coarse scan, fine scan, half-consistency).

3. **Tests updated to the new API + removed the back-compat crutch**:
   - `_tel_line_fit`'s `sigma_y` was briefly given a `None` default (→
     reuse `sigma_x`) to keep legacy scalar test calls working. User then
     asked to update the tests and drop the default, so `sigma_y` is now
     **required**. The fallback line is gone.
   - `tests/test_stage5.py`: `_tel_line_fit(...)` calls pass both σ;
     `Coincidence(...)` positional ctor gets the extra cov + probe-σ args;
     `_linear_solve_fixed_theta(coincs, c, s)` dropped its scalar arg.
   - `tests/test_corner_probe_edge_cases.py`: `_line_chi2` passes both σ;
     the in-loop `_tel_line_fit` now builds per-plane `sx`/`sy` arrays from
     the decoded hits (mirrors production); `_maha` uses `sigma_prb_x/y`
     and `cov_ab_x/y`.

## State

**133 tests pass; `ruff check` + `ruff format --check` clean** on all three
touched files. Changes are **uncommitted and on `main`** — needs a branch
before committing (repo convention: branch off main for PRs).

## Files actively edited

- `src/monrad/stage5.py`        (the real change)
- `tests/test_stage5.py`        (API updates)
- `tests/test_corner_probe_edge_cases.py` (API updates)

## What did NOT work / was reversed

- No technical dead-ends — the implementation passed on first run each
  round. The only reversal was the `sigma_y=None` back-compat default,
  added then removed at the user's request once the tests were updated.
- Pre-existing `ty` diagnostics persist and were **not** addressed (out of
  scope): `scipy.optimize` unresolved import; `_decode_cluster` accesses
  `.quality`/`.x_mm`/`.sigma_*` on `decode_position`'s `Hit | None` return
  without a None guard. Same list as the prior session's follow-ups.

## Next step

**Close the coverage gap, then commit.** `synth.generate()` only emits
**golden** hits (`_ch_to_u64` sets one fiber + one ribbon bit → width 1 →
`σ_x == σ_y`, identical across planes); `fold=True` yields `unresolved`,
not `cluster`. So the new heteroscedastic branches *run* but always with
equal weights — **no test actually proves distinct per-axis/per-plane σ
changes the fit.** The stage5 unit tests also pass equal `cov`/σ for both
axes.

1. Extend `src/monrad/synth.py` with a way to emit `cluster` hits of
   controllable width, **differing per axis and per plane** (set 2
   contiguous fiber/ribbon bits so a plane decodes `cluster`, width 2,
   σ≈5.77 mm on one axis vs width 1 on the other). Likely a
   `cluster_widths` param threaded into `_ch_to_u64`.
2. Add a stage5 test asserting the sharper plane/axis is weighted more
   heavily (e.g. cov / residual shifts toward the low-σ plane).
3. Branch (`fix/per-axis-sigma` or similar), commit, open PR.

Secondary: address the `Hit | None` None-guard in `_decode_cluster` while
in the file.

---

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

## Open PR

Branch `fix/coincidence-double-counting` →
https://github.com/gallog-hash/00_monrad-py/pull/new/fix/coincidence-double-counting

## Reviewer checklist

- [ ] Confirm the transitive-closure semantics match DESIGN.md §5.1 (a cluster
      spans events with consecutive gaps ≤ `window_ns`, so a cluster can be
      wider than `window_ns` end-to-end). Confirm this is intended vs. a hard
      `window_ns` span cap.
- [ ] Confirm rejecting (not keeping) clusters with ≥2 telescope or ≥2
      same-probe events is the desired policy for the pose fit.
- [ ] `git diff main...HEAD` and verify the ruff-format reflow didn't bury a
      logic change (the hook reformatted both source files on commit).

## Follow-ups (out of scope for this PR)

- [ ] **Validate on real detector data.** The synthetic data spaces tracks
      ~10 ms apart, so it never exercised the high-rate growing-window path.
      Run the pipeline on a real telescope+probe pair and confirm `n_inliers`
      and pose covariance are no longer inflated.
- [ ] **Quantify the impact.** Compare `n_inliers` / `cov` before vs. after on
      real data to document how much the bug was inflating results.
- [ ] **Multi-probe path.** `_decode_cluster` already supports it, but there is
      no end-to-end test with ≥2 probe detectors sharing telescope events. Add
      one when a multi-probe dataset exists.
- [ ] **Stage 2 multi-probe clusters.** `coincidence_stream` can still yield
      probe-only clusters (≥2 probes, no telescope). Harmless for the current
      single-`PoseFitter` use, but decide whether stage 2 should filter them.
- [ ] **Window tightening.** DESIGN.md §5 notes Δt may drop from 200 ns to
      ~100 ns once the empirical Δt distribution is measured.
- [ ] **Pre-existing ty diagnostics** (not introduced here):
      `decode_position` returns `Hit | None` but `_decode_cluster` accesses
      `.quality`/`.x_mm` without a None guard; scipy stubs unresolved. Address
      separately.

## Housekeeping

- [ ] Untracked scratch still in the tree to delete or `.gitignore`:
      `memory/`, `pipeline_out/`, `.claude/`, `1`. (`handoff.md` and the new
      test are now committed; `to-do.md` and `scripts/synth_plot.png` removed.)

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
