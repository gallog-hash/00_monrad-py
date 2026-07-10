# Handoff: residual-RMS window gate (shipped) + follow-ups

Written 2026-07-03. Continue from here on another machine.

Repo: `00_monrad-py`
Branch: `feat/probe-monitoring` — **pushed to `origin`**, HEAD `9918b45`
("Add residual-RMS window gate to monrad-monitor").

> The corrected diagnosis in this file lived in a **local, untracked** memory
> dir on the origin machine, so the essentials are reproduced inline below —
> this handoff is self-contained.

## Status: the residual-RMS window gate is DONE, committed, and pushed.

Two follow-ups remain (both deferred, see end). Nothing else is in flight.

## What shipped (commit `9918b45`)

A per-window quality gate in `monrad-monitor` that **drops (and logs)** windows
contaminated by an excess of wide-angle "wild" telescope tracks — internally
consistent 3-plane lines that pass `chi2_track_cut` but miss the probe.

All in `src/monrad/monitor/timeseries.py`:
- `--max-resid-rms MM` CLI flag / `max_resid_rms_mm: float | None` param on
  `monitor_probe` (default **OFF**; `None` preserves prior behaviour).
- `resid_rms` recorded on every `WindowResult` and written to the CSV
  (`pose_timeseries.csv`), always — even for accepted windows, for tuning.
- Whole-run RMS distribution (min/median/max) printed at end of a run.
- Gate lives in the shared `_emit` closure, so it covers BOTH count-based and
  hybrid batching automatically.
- Tests: 6 new gate tests in `tests/test_monitor_timeseries.py`; full file 33
  passed. Ruff clean (enforced by pre-commit).

### The key design decision: which residual RMS

The gate keys on `_window_resid_rms(pose)` — the combined absolute-mm residual
RMS over **ALL coincidences fed to the fit (inliers + Mahalanobis-cut
outliers)** against the fitted pose, NOT `PoseResult.residuals_x/y`.

Why: `residuals_x/y` are **inlier-only** (post the fit's `d>4` Mahalanobis cut).
The cut rejects the wild tracks, so those inlier residuals stay flat at ~14 mm
even for a badly contaminated window — no separation. Counting the rejected
tracks back in is what exposes the contamination.

## Empirical results (these CORRECT the original investigation's predictions)

Verified on `data/0_testLab_20210723` (Base telescope, Probe_0,
`--z-tel 0 -1340 -670 --min-fit 2000 --min-anchor-planes 0`, 9 windows over
~11.5 h). The original handoff predicted "~10 mm normal / ~300 mm bad / ~30×
margin / 50 mm threshold" — those numbers were measured on a **different
quantity** (per-coincidence residual vs a FIXED reference pose, over ALL
coincidences, at min_fit=300 sub-spans) and **do NOT hold** for the shipped
full-window gate. What the live gate actually shows:

| Window (UTC) | n_inliers | resid_rms (mm) |
|---|---|---|
| 09:36–10:50 | 1421 | 157.1 |
| 10:50–12:06 | 1318 | 159.0 |
| 12:06–13:21 | 1354 | 154.2 |
| 13:21–14:36 | 1443 | 131.9 |
| 14:36–15:54 | 1331 | 156.7 |
| 15:54–17:08 | 1341 | 153.1 |
| **17:08–18:26** | **656** | **281.6**  ← anomaly, the max |
| 18:26–19:47 | 1203 | 173.7 |
| 19:47–21:04 | 1353 | 152.3 |

- **Inlier-only RMS is FLAT ~14 mm across all 9 windows — no separation.**
  (This is why the gate can't use `residuals_x/y`.)
- **All-coincidence RMS**: anomaly **281.6 vs a 132–174 mm baseline** → only
  **~1.6× margin, NOT 30×**. Every window sits at ~150 mm because ALL windows
  carry an ever-present ~15–28% wild-track baseline the fit correctly ignores;
  the anomaly just adds enough extra wild tracks to rise ~1.6× above the pack.
- The original 50 mm threshold would drop **all 9** windows. **Correct
  threshold for THIS setup ≈ 220 mm** — drops only 17:08, keeps the other 8
  (verified against the CSV; the gate is a pure threshold on the streamed
  value). There is no universal mm — tune per setup from the printed
  distribution. The CLI/docstring say this.

### The still-open observation: inlier fraction separates BETTER

The **inlier fraction** discriminates this anomaly more cleanly than RMS:
656/2000 = **33% vs 60–72%** (~2× margin, vs RMS's 1.6×). The fit throwing away
half the coincidences is the sharpest fingerprint of this window. This
contradicts an earlier "inlier fraction is not a usable signal" claim — that
held only at **sub-span** granularity (a wild-ONLY 11-min span keeps 300/300);
at full-window granularity mixing good+wild, the cut rejects the wild ones and
the fraction separates. See Follow-up 1.

## Reproduce

Two ~15-min runs (the `z-tel` order for THIS dataset is `0 -1340 -670`, NOT
file-column order):

```bash
# No gate — prints every window's resid_rms so you can see the separation
uv run monrad-monitor --telescope data/0_testLab_20210723/Base \
  --probe data/0_testLab_20210723/Probe_0 \
  --z-tel 0 -1340 -670 --min-fit 2000 --min-anchor-planes 0 \
  --out pipeline_out/monitor_rms_allcoinc

# Gated at 220 mm — drops the 17:08–18:26 window (logged), keeps 8
uv run monrad-monitor --telescope data/0_testLab_20210723/Base \
  --probe data/0_testLab_20210723/Probe_0 \
  --z-tel 0 -1340 -670 --min-fit 2000 --min-anchor-planes 0 \
  --max-resid-rms 220 --out pipeline_out/monitor_gated
```

Tests: `uv run pytest tests/test_monitor_timeseries.py` (33; ~4 min because the
real-data fixtures run the full testLab dataset). The synthetic-only gate tests:
`uv run pytest tests/test_monitor_timeseries.py -k "gate or resid"` (fast).

## Follow-up 1 (needs a user decision): `--min-inlier-fraction` gate

The inlier fraction is the cleaner discriminator (see above). Considered adding
a second drop criterion but did NOT — it's a bigger API change (new CLI flag +
param) that shouldn't be made unilaterally. If wanted: add
`--min-inlier-fraction` / `min_inlier_fraction: float | None`, dropping a window
if `pose.n_inliers / len(coincs_fed) < threshold`. Wire it into the same `_emit`
gate so a window drops if **either** gate trips; record the fraction alongside
`resid_rms`. Threshold ~0.5 cleanly separates this dataset (0.33 vs 0.60–0.72).

## Follow-up 2 (NEXT session, explicitly wanted): in-fit absolute-mm recovery

Rationale: dropping whole windows is cheap when the probe is close (high
coincidence rate, as in this test data). When the probe is **farther away** the
good-coincidence rate falls and every good coincidence matters — recovering the
good core beats dropping the window.

What to build: add an **absolute-mm residual rejection** to the stage-5 robust
step (`fit_probe_pose` in `src/monrad/pose/optimize.py`, the Mahalanobis d>4 cut
+ one-pass refit ~line 259), **complementing** (not replacing) the Mahalanobis
cut that the inflated wide-angle covariances defeat. The investigation proved a
correct pose exists: refitting the two worst sub-spans on only their
`resid < 20 mm` core (~113 of 300 coincidences) gave z_p = 839.0 / 840.7 mm
(normal), σ inflated only ~1.3×.

Caveats: this touches the **core fitter shared by stages 4 and 5** and
`scripts/run_pipeline.py`, so it needs regression coverage on normal windows —
which already carry the ~15–28% baseline wild coincidences the current fit
handles fine; the new cut must NOT over-trim those. Treat as a separate,
well-tested change.

## Key gotchas (carry these to the other machine)

- **z-tel order for testLab 20210723 is `0 -1340 -670`** — the telescope file
  columns are NOT in z order. Filenames are NOT UTC either (first file
  `20210723_114032` but first window starts 09:36 UTC); bucket by decoded
  `t_ns`, never by filename.
- **`PoseResult.residuals_x/y` are inlier-only** (post-Mahalanobis). For any
  "does this window's raw data agree" question, use inliers **+** outliers vs
  the fitted pose (that's what `_window_resid_rms` does).
- **The anomaly is telescope-side, NOT probe motion.** Probe-frame `u,v` are
  steady throughout; the 17:08 window's apparent excursion (t_x≈188 vs ~180,
  z_p≈845 vs ~837) is the pose fit landing wrong on wild-track-contaminated
  data. Do not re-investigate it as a probe move.
- **Absolute-time anchoring**: `reconstruct_stream` anchors `t_ns` to the FIRST
  PPS (= header `utc0`) with `pps_count` cumulative across files, so you CANNOT
  front-slice the file list to restrict the interval — it mis-anchors every
  timestamp. Stream from the run start and early-stop past the interval.
- The `"GPS events but N position blocks"` ±1–2 warnings at file boundaries are
  pre-existing and benign.

## Housekeeping

`pipeline_out/`, `.claude/`, `memory/` are intentionally untracked (local only).
The `pipeline_out/monitor_rms_allcoinc/` and `.../monitor_gated/` reference runs
from this session are local to the origin machine — regenerate with the
reproduce commands above.
