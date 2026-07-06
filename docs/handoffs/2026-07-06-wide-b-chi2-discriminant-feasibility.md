# Handoff: can a per-coincidence wide-|b| / χ² discriminant be implemented?

Written 2026-07-06. Branch `feat/probe-monitoring`. Working tree clean (this
session was investigation-only — no source changes; all scripts live in the
session scratchpad, see "Artifacts" below).

Continues the testLab 20210723 anomaly thread from
[`2026-07-06-fixed-zp-experiment-outcome.md`](2026-07-06-fixed-zp-experiment-outcome.md).
**Do not re-derive the diagnosis** — it is fully captured in project memory
[`testlab-20210723-anomaly-no-raw-telescope-signature.md`](../../memory/testlab-20210723-anomaly-no-raw-telescope-signature.md)
(indexed in `MEMORY.md`). Read that first; this doc only covers the
implementation question the next session should answer.

## Why this discriminant is now on the table (one-paragraph result)

This session traced the 17:08–18:26 UTC `z_p` excursion to its root. The anomaly
is **two sharp 5-min bursts (17:16:21 & 18:11:21 UTC)** of wide-angle
**ghost coincidences**: |b|=hypot(b_x,b_y) reaches the geometric aperture limit
0.70 (baseline median 0.18, p99 0.44), intercept scatter ~2×. Established, in
order: raw telescope singles flat; raw probe singles flat; coincidence rate flat;
per-coincidence Δt=t_tel−t_prb tight & centred (**timing/mis-pairing refuted** —
they are genuine coincidences); and the wide coincidences sit on telescope blocks
with ~1.5× more per-plane candidates. **Conclusion: the wide tracks are a
stage-5 combinatorial ghost-track artifact** — the finder assembles a wide
(large-b, large-a) mirror-fold/double-hit triple that threads the correctly-paired
probe hit. There is no raw-data veto; the correct lever is a per-coincidence
discriminant in stage 5. Hence this task.

## Answer sketch: YES, and |b| is the right discriminant (not χ²)

- **A telescope-triple χ² gate already exists and does NOT catch these.**
  `src/monrad/pose/fitter.py:229`, `best_chi2 >= _CHI2_TRACK` (`_CHI2_TRACK = 4.0`,
  fitter.py:25), reason `"chi2_track_cut"`. Ghost tracks fit a *straight line*
  well (that is why the finder selects them), so their telescope χ² is **low** —
  below the gate. A 4-point χ² that adds the probe point is also likely low,
  because the ghost threads the probe *by construction* under the fitted pose.
  So χ² is a weak discriminant here.
- **|b| is a geometric discriminant that works.** A genuine coincidence is one
  particle through the 40 mm probe at z_p≈840 mm, hence physically near-vertical;
  a wide |b| is geometrically impossible for a real single track. Real coincidence
  |b| distribution: median 0.18, p90 ~0.29, p99 ~0.44, whole-run wide(|b|>0.5)
  frac 0.004 (38/10668). Ghosts pile up at 0.5–0.70. A hard cut at **|b| ≳ 0.5**
  (or 0.55) removes the ghost population with essentially no cost to real tracks.
  Verify the exact threshold against the whole-run |b| distribution the way
  `--max-resid-rms` is tuned per-setup (there is no universal value; but the
  physical max is ~0.75 so the safe band is narrow).

## Concrete implementation plan (mirror the shipped `--max-abs-resid` plumbing)

The gate belongs in `_decode_cluster`, as a **sibling of the existing
chi2_track_cut**, applied to the winning triple's slope right after `best_fit`
is unpacked (fitter.py:236, where `b_x, b_y` become available):

1. **`src/monrad/pose/fitter.py`**
   - `PoseFitter.__init__`: add `max_track_slope: float | None = None`
     (store on `self`), next to `max_abs_resid_mm` (fitter.py:50/76).
   - `_decode_cluster`: after line 236, if `max_track_slope is not None and
     hypot(b_x, b_y) > max_track_slope: _report("wide_track_cut", ...); return None`.
   - Add `"wide_track_cut"` to `GATE_ORDER` in `src/monrad/pose/types.py:79`
     (place it after `chi2_track_cut`, before `probe_quality`, matching check
     order). Extend `DecodeReport` handling if a slope field is wanted (optional).
2. **`src/monrad/monitor/io.py`** `stream_coincidences` (~line 116): add a
   `max_track_slope` kwarg and pass it into the `PoseFitter(...)` construction
   (line 135). NOTE: unlike `max_abs_resid_mm` (a *pose-refit* param threaded via
   `timeseries._emit`→`fit_probe_pose`), the slope gate is a *per-cluster decode*
   param and rides on `PoseFitter` construction — so it flows through
   `stream_coincidences`, **not** through `_emit`.
3. **`src/monrad/monitor/timeseries.py`** `monitor_probe` + its argparse: add
   `--max-track-slope` and pass to `stream_coincidences` (compare the
   `--max-abs-resid` wiring at timeseries.py:112/431/482, but route to the
   stream call ~line 237, not `_emit`).
4. **`scripts/run_pipeline.py`**: add `--max-track-slope` (mirror `--max-abs-resid`
   at run_pipeline.py:181) and pass into the `PoseFitter`/stream construction
   (~line 592).
5. **Tests**: add a `tests/test_stage5.py` case — a synthetic wide triple that
   passes chi2_track_cut but exceeds the slope gate → `wide_track_cut`, and a
   near-vertical control that survives. There is a known pre-existing `ty`
   invariance warning on `PoseFitter.add(list[tuple[int, TimedEvent, PosRef]])`
   in test_stage5.py / run_pipeline.py — unrelated, do not chase it.

## Validation (before committing)

- Re-run the monitor on testLab 20210723 (`--z-tel 0 -1340 -670`,
  `--min-anchor-planes 1`) with `--max-track-slope 0.5` and confirm the two
  bursts' `z_p`/`resid_rms` normalise while the other windows are unchanged. The
  clean way to A/B: dump per-coincidence |b| with/without the gate (the session's
  `coinc_b_dist.py` pattern) and confirm only the 38 wide tracks are removed.
- Cross-check: the gate should drop ~12 (17:16) + ~8 (18:11) coincidences and
  near-zero elsewhere. If it drops a broad tail everywhere, the threshold is too
  tight — raise toward 0.6.

## Open question (secondary — not required for this task)

*Why the ghosts cluster in exactly two 5-min windows* is still unexplained: no
raw-singles quantity changes there. Leading candidate = transient telescope
alignment micro-shift; decisive test = time-resolved stage-4 alignment (refit
per-plane offsets/rotations for the burst windows vs neighbours). The |b| gate
mitigates the *symptom* regardless of this cause, so it can ship independently.

## Artifacts from this session

The inspectors are committed under **`scripts/diagnostics/`** (see its
`README.md`): `tel_raw_inspect.py`, `tel_plane_inspect.py`, `tel_time_inspect.py`
(raw telescope), `probe_raw_inspect.py` (raw probe), `coinc_b_dist.py` (|b|
dist), `coinc_dt.py` (Δt test), `wide_block_inspect.py` (double-occupancy). Run
from the repo root; scripts that persist an `.npz` honour `$MONRAD_DIAG_OUT`
(default: cwd). The `.npz` outputs themselves were NOT committed (regenerable;
one is ~39 MB) — re-run to regenerate. File→UTC mapping: local filename =
UTC + 1h59m40s (CEST); the two burst files are `20210723_191534` (tel) /
`…191558` (prb) and `20210723_201033` / `…201058`.

## Suggested skills for the next session

- **`astral:ruff`** — lint/format the fitter + CLI changes (pre-commit enforces).
- **`astral:ty`** — type-check the new optional param plumbing.
- **`/verify`** — drive `monrad-monitor` end-to-end on testLab to confirm the
  gate drops exactly the two burst windows' ghosts and nothing else.
- **`/code-review`** (medium) on the diff before committing.
- **`/run`** or `monrad-monitor` directly — reproduce the per-window timeseries.
