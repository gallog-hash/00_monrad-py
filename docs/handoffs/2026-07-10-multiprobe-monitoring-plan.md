# Multi-probe monitoring (Step 3) — updated plan

## Context

`~/.claude/plans/it-s-time-to-adapt-bubbly-sunrise.md` laid out four steps for
real-world probe-position monitoring: 0a (package refactor), 0b (optimization
pass), 1 (resolution characterization), 2 (time-windowed monitoring), 3
(multi-probe). Steps 0a, 0b, 1, and 2 are done — Step 2 (`monrad-monitor`,
`src/monrad/monitor/timeseries.py`) has grown well past its original sketch:
count-based/hybrid windowing, a rigidity gate, an off-probe footprint gate, a
post-fit continuity gate, cold-start bootstrapping, and a raw-batch growth
cap. Step 3 (multi-probe) was never implemented and its original sketch
predates all of that — this plan re-derives Step 3 against the current code
instead of the stale sketch.

**Confirmed still missing:** no `src/monrad/monitor/multiprobe.py`, no
`tests/test_monitor_multiprobe.py`, and `timeseries.py`'s `--probe` CLI flag
is `required=True` single-value, not repeatable.

**Confirmed still true from the design:** `PoseFitter` is already keyed by
`tel_id`/`prb_id` (`src/monrad/pose/fitter.py:38-60`), and
`_decode_cluster`'s docstring (`fitter.py:145-152`) already states the intended
multi-probe contract: *"a single telescope event may legitimately be in
coincidence with several distinct probes, each handled by its own
PoseFitter."* `coincidence_stream` (`src/monrad/coincidence/search.py`) is
already k-way over an arbitrary `detector_ids` list. `io.py`'s module
docstring already lists `multiprobe` among the monitoring drivers it exists to
serve. So the substrate is ready; only the driver/CLI multiplexing is
missing, exactly as the original plan anticipated.

## Design decisions (new, not in the original sketch)

1. **New console script, not an extended `monrad-monitor`.** Multi-probe
   output is inherently per-probe (`pose_timeseries_probe{k}.csv`), and
   `monrad-monitor`'s existing single-probe output filename
   (`pose_timeseries.csv`) is relied on by current tests/tooling. Add
   `monrad-multiprobe = monrad.monitor.multiprobe:main` instead of overloading
   `--probe` on the existing command.
2. **Extract the windowing/gating state machine out of `monitor_probe` into a
   reusable class.** Today the raw-batch growth, gates, fit, continuity check,
   and result bookkeeping are closures inside `monitor_probe`
   (`timeseries.py:167-530`, ~360 lines). Multi-probe needs N independent
   instances of exactly this state machine (each probe has its own
   `prev_pose`, cold-start anchor, raw batch) driven off a **shared** cluster
   stream. Factor it into a class (`_WindowAccumulator`) that `monitor_probe`
   itself is rewritten to use (so single-probe behavior is unchanged, not
   duplicated) and that `multiprobe.py` instantiates once per probe.
3. **One shared `coincidence_stream`, N `PoseFitter`s.** Per the fitter's own
   documented contract, decode every cluster once per probe (`fitter[k].
   decode_cluster(cluster)` for each `k`); a cluster only yields a
   `Coincidence` for the probe(s) it's actually consistent with (exactly one
   telescope entry + exactly one entry for that `prb_id`). No new
   demultiplexing logic is needed at the cluster level — `_decode_cluster`
   already ignores other probes' events.
4. **Known, deferred inefficiency: redundant telescope track-finding.**
   `reconstruct_plane_candidates` + the combinatorial χ² search inside
   `_decode_cluster` depend only on the telescope entry, not on `prb_id` — so
   with N probes every cluster pays that search N times over. Not fixed here
   (would need a per-cluster memoization keyed off the telescope `PosRef`,
   shared across fitters); flagged as a follow-up, same as the Step 0b
   deferred-optimization list, since it doesn't block correctness and N is
   expected to be small.
5. **Gate thresholds stay global; probe footprint size can vary per probe.**
   `--max-rigidity-resid-mm`, `--max-off-probe-mm`, `--max-pose-jump-mm/-deg`,
   `--min-fit`, `--window-s`, `--min-anchor-planes` apply identically to every
   probe's independent accumulator (adding a whole matrix of per-probe
   overrides is out of scope until something demands it). `--n-probe-ch` is
   the one exception: probe channel count (hence physical footprint size,
   `CLAUDE.md`'s "channel count unknown a priori") plausibly differs per
   physical probe and feeds directly into the off-probe gate and the
   centre-covariance propagation, so it's made **repeatable to match
   `--probe` 1:1**, falling back to one shared value applied to all probes
   when given once.

## Changes

### `src/monrad/monitor/timeseries.py` (refactor, no behavior change)

- Add `_WindowAccumulator`: constructor takes the per-run config currently
  closed over in `monitor_probe` (`min_fit`, `window_ns`,
  `max_rigidity_resid_mm`, `max_off_probe_mm`, `max_pose_jump_mm/deg`,
  `probe_size_mm`, `z_corr`, `alignment`, `n_probe_ch`) plus a `label: str =
  ""` used to prefix its log messages (so interleaved multi-probe logs are
  attributable). Methods:
  - `push(co: Coincidence) -> WindowResult | None` — the body of the current
    `for co in stream: ...` loop (`timeseries.py:428-500`): append to batch,
    check span/min_fit, run `_run_gates`, grow-or-drop on the raw cap, fit,
    check continuity, `_record` or drop. Returns the new `WindowResult` when a
    window closes, else `None`.
  - `finalize() -> None` — the trailing-batch warning
    (`timeseries.py:502-511`).
  - `results: list[WindowResult]` accumulated internally (same as today).
  - Keeps `_run_gates`/`_record` as private methods carrying `prev_pose`,
    `cold_start_z_ref`, `cold_start_n` as instance state instead of
    `nonlocal`.
- Rewrite `monitor_probe` to build one `_WindowAccumulator`, drive it off
  `stream_coincidences(...)` (unchanged), then reuse the existing
  print/CSV/plot tail (`timeseries.py:513-530`) unchanged. Public signature
  and return type (`list[WindowResult]`) unchanged.
- `_window_resid_rms` and `_pose_jump` stay module-level pure functions,
  called from the accumulator.
- **Gate:** `uv run pytest tests/test_monitor_timeseries.py -q` green with
  zero test changes — this refactor must be behavior-preserving.

### `src/monrad/monitor/io.py` (additive)

- Add `build_cluster_stream(tel: DetectorFiles, probes: list[DetectorFiles], *,
  window_ns=...) -> Iterator[list[tuple[int, TimedEvent, PosRef]]]`: builds
  one `reconstruct_stream` per detector (telescope + each probe) and calls
  `coincidence_stream([tel_stream, *prb_streams], detector_ids=[0, 1, ...,
  len(probes)])`. Telescope is always `det_id=0`; probe `k` (0-indexed in the
  `probes` list) is `det_id=k+1`, matching `PoseFitter.prb_id`'s existing
  convention.
- `stream_coincidences` (single-probe) stays as-is — `resolution.py` and
  `monitor_probe` keep using it unchanged.

### `src/monrad/monitor/multiprobe.py` (new)

Console script `monrad-multiprobe`.

- `monitor_probes(tel_dir: Path, prb_dirs: list[Path], *, window_s, z_tel,
  n_probe_ch: list[int], out_dir, min_fit, min_anchor_planes,
  max_rigidity_resid_mm, max_off_probe_mm, max_pose_jump_mm,
  max_pose_jump_deg, tot_thresh, tot_weights, make_plots) -> list[list[
  WindowResult]]` (one inner list per probe, same order as `prb_dirs`):
  - `load_detector(tel_dir)` + `load_detector(d)` for each `d` in `prb_dirs`;
    `fit_alignment` once (shared telescope alignment, same as today).
  - Build one `PoseFitter` per probe: `tel_id=0`, `prb_id=k+1`,
    `tel_pos_paths=tel.pos_paths`, `prb_pos_paths=probes[k].pos_paths`, same
    `tot_thresh`/`tot_weights`/`min_anchor_planes` for all (per decision 5).
  - Build one `_WindowAccumulator` per probe (`label=f"probe{k+1}"`), sized
    per-probe `n_probe_ch[k]`.
  - `for cluster in build_cluster_stream(tel, probes): for k, fitter in
    enumerate(fitters): co = fitter.decode_cluster(cluster); if co is not
    None: acc[k].push(co)`.
  - `finalize()` every accumulator; write `pose_timeseries_probe{k+1}.csv` +
    (if `make_plots`) `pose_timeseries_probe{k+1}.png` per probe under
    `out_dir`, reusing `_write_csv`/`_plot_timeseries` from `timeseries.py`.
    Print the same per-probe residual-RMS summary `monitor_probe` prints
    today, once per probe.
- CLI: `--telescope DIR` (required, single), `--probe DIR` (required,
  `action="append"`, at least one), `--n-probe-ch` (`nargs="+"`, default
  `[30]`; error at parse time unless `len(...) in {1, len(probes)}` — a
  single value broadcasts to all probes). All other flags mirror
  `monrad-monitor`'s (`--z-tel`, `--min-fit`, `--min-anchor-planes`,
  `--max-rigidity-resid-mm`, `--max-off-probe-mm`, `--max-pose-jump-mm/-deg`,
  `--window-s`, `--out`, `--tot-thresh`, `--tot-weights`, `--no-plots`) —
  same parsing/logging pattern as `timeseries._parse_args`/`main` (explicit
  vs. default tagging in the run-configuration log).

### `pyproject.toml`

- Add `monrad-multiprobe = "monrad.monitor.multiprobe:main"` under
  `[project.scripts]`.

### Tests

- `tests/test_monitor_multiprobe.py`, following the `synth_run` fixture
  pattern in `tests/test_monitor_timeseries.py`:
  - Two `generate()` calls with the **same `seed`/`n_tracks`** (identical
    telescope tracks — verified: `_sample_tracks` runs before any
    pose-dependent step, and with `fold=False`/`fold_crosstalk_rate=0.0` the
    telescope encoding draws no further `rng` state, so both calls' `tel_dir`
    content is byte-identical) but distinct `(t_x, t_y, z_p)` and distinct
    `out_dir`s — two independent probe poses sharing one telescope
    acquisition. Use one call's `tel_dir`, both calls' `probe_dir`s.
  - `monitor_probes(tel_dir, [prb_dir_1, prb_dir_2], window_s=150.0, ...)` →
    assert each returned list recovers its own probe's `(t_x, t_y, z_p,
    theta)` truth within a few σ, and that the two probes' `WindowResult`
    lists are independent (different `n_inliers`/timestamps are fine; each
    just needs internal consistency against its own truth).
  - `--n-probe-ch` broadcast vs. per-probe list: one test with a single
    shared value, one with two distinct values matching each probe's actual
    `n_probe_ch`; one test asserting a parse error when the count matches
    neither `1` nor `len(probes)`.
  - CSV/plot output: both `pose_timeseries_probe1.csv` and
    `pose_timeseries_probe2.csv` (+ pngs) are written to `out_dir`.
- `tests/test_monitor_timeseries.py`: no new tests, but re-run unchanged as
  the regression gate for the `_WindowAccumulator` extraction.

## Verification

```bash
uv run ruff check . && uv run ruff format --check .

# Regression: timeseries refactor must not change single-probe behavior
uv run pytest tests/test_monitor_timeseries.py -q

# New: multi-probe
uv run pytest tests/test_monitor_multiprobe.py -q

# Full suite
uv run pytest -q

# CLI smoke (two synthetic or real probe dirs sharing one telescope acquisition)
monrad-multiprobe --telescope <tel_dir> --probe <prb1_dir> --probe <prb2_dir> \
    --z-tel 0 400 800 --min-fit 30 --out pipeline_out/multiprobe
```

Confirm: `--help` lists repeatable `--probe`; each probe's CSV recovers its
own truth independently; `monrad-monitor`'s existing single-probe tests and
CLI output are byte-for-byte unaffected by the `_WindowAccumulator` refactor.

## Critical files

- **Refactor:** `src/monrad/monitor/timeseries.py` (extract
  `_WindowAccumulator`; `monitor_probe` becomes a thin driver over it).
- **Add:** `src/monrad/monitor/io.py` (`build_cluster_stream`),
  `src/monrad/monitor/multiprobe.py`, `tests/test_monitor_multiprobe.py`.
- **Modify:** `pyproject.toml` (new console script).
- **Reuse unchanged:** `PoseFitter`/`decode_cluster` (`pose/fitter.py`),
  `coincidence_stream` (already k-way), `fit_alignment`/`load_detector`/
  `centre_cov_2x2` (`monitor/io.py`), `_write_csv`/`_plot_timeseries`
  (`monitor/timeseries.py`).

## Deferred (out of scope for this pass)

- Per-cluster telescope-track-finding memoization across probes (decision 4).
- Per-probe overrides of gate thresholds / `min_fit` / `window_s` (decision
  5) — add only if a real multi-probe deployment needs heterogeneous
  windows.
- A combined cross-probe plot (each probe currently gets its own PNG; an
  overlay figure is a nice-to-have, not required by the original plan's
  verification step 5).
