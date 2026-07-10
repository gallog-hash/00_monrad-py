# `--max-resid-rms` removed: also redundant once the rigidity gate is on

Written 2026-07-07. Branch `feat/probe-monitoring`. Follow-up to
`docs/handoffs/2026-07-07-abs-resid-cut-removed.md` (same-day removal of the
in-fit `--max-abs-resid` cut for the same reason) and
`docs/handoffs/2026-07-07-rigidity-footprint-gates-validated.md` (rigidity/
footprint gates, shipped and validated earlier the same day).

## What happened this session

1. Re-read the abs-resid-removed handoff and ran the same style of check
   against the other pre-existing window-quality gate, `--max-resid-rms`
   (the *all-coincidence* residual-RMS gate, as opposed to abs-resid's
   in-fit per-point cut) — does it still fire once the rigidity gate
   (`--max-rigidity-resid-mm`) is already active?
2. Ran the full `data/0_testLab_20210723` day with
   `--z-tel 0 -1340 -670 --n-probe-ch 40 --min-anchor-planes 1
   --window-s 300 --max-rigidity-resid-mm 100 --max-off-probe-mm 100
   --max-resid-rms 220` (220 mm being the threshold this gate was
   originally tuned to in `memory/testlab-20210723-anomalous-window-
   telescope-side.md`, back before the rigidity gate existed).
3. Result: **zero** `"Dropping window ... residual RMS"` warnings anywhere
   in the run. 140/140 windows survived — identical count to the
   rigidity-gate-alone baseline. Whole-run resid-RMS distribution
   (`min=15.9 median=35.1 max=129.7 mm`) exactly reproduced the
   rigidity-only numbers from the validation handoff, and the 129.7 mm max
   sits comfortably under the 220 mm threshold. The only windows actually
   dropped (`12:36:07`, `17:14:31`, `18:11:06`) were all caught by the
   pre-existing `< min_fit` check downstream of the rigidity gate, not by
   `max_resid_rms_mm`.
4. Conclusion: same story as abs-resid. The rigidity gate — cross-
   coincidence, pose-free, anchored to the *previous* window's accepted
   pose — already exiles the wide-angle "wild" tracks that would otherwise
   inflate a window's all-coincidence RMS. Once it's on, there is nothing
   left for `--max-resid-rms` to catch on this dataset.
5. **Removed `max_resid_rms_mm` / `--max-resid-rms` from the codebase**:
   the parameter, its docstring block, the drop-and-log branch in
   `monitor_probe`'s `_emit`, the CLI flag, and the `main()` wiring, all in
   `src/monrad/monitor/timeseries.py`. Also reworded the whole-run-summary
   comment (used to say "helps set --max-resid-rms per setup") since the
   flag no longer exists. **Left `resid_rms` itself in place** — it is
   still computed by `_window_resid_rms`, printed in the end-of-run
   summary, and written as a CSV column; only the threshold-and-drop
   behavior was removed, since it remains a useful window-quality
   diagnostic to eyeball even with no gate attached to it.
   `tests/test_monitor_timeseries.py`: removed the four gate-behavior
   tests (`test_gate_high_threshold_keeps_all`,
   `test_gate_drops_high_rms_keeps_low`, `test_gate_drops_all_below_min`,
   `test_gate_logs_warning_on_drop`); kept the two tests that exercise
   `resid_rms` as a plain diagnostic (`test_resid_rms_populated`,
   `test_gate_csv_has_resid_rms_column`).

## Current state — committed

`ruff check`/`ruff format --check` clean on both touched files. Full suite
green: `uv run pytest` → 215 passed. Diff: `+3/-118` across
`src/monrad/monitor/timeseries.py` and `tests/test_monitor_timeseries.py`.

## Not done / open follow-ups

- Same open items as the abs-resid handoff, unaffected by this change:
  rigidity/footprint gates remain opt-in in `monrad-monitor` and are not
  wired into `PoseFitter`'s streaming pipeline path
  (`scripts/run_pipeline.py`); no CLI diagnostic prints the rigidity
  pairwise-score distribution yet; the third dropped window in the
  143→140 full-day run (`12:36:07`, distinct from the two known bursts)
  is still not identified or investigated.
- With both `--max-abs-resid` and `--max-resid-rms` now gone, the rigidity
  (+ optional footprint) gate is the sole contamination defense in
  `monrad-monitor`. If a future dataset surfaces contamination the
  rigidity gate doesn't catch (e.g. a rigidly-but-wrongly-drifted window,
  the scenario the footprint gate exists for per the strategy doc), the
  all-coincidence RMS is still available as a printed/CSV diagnostic to
  notice it by eye even without a gate.

## Suggested skills

- `verify` / `astral:ruff` — same as the abs-resid handoff, for any
  follow-up edits in this area.
