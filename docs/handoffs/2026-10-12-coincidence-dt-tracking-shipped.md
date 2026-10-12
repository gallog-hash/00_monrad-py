# Handoff — probe−telescope Δt tracking shipped (instrumentation only)

**Date:** 2026-10-12
**Branch:** `feat/coincidence-dt-tracking` — committed **and pushed** to `origin`, **no PR opened**
**Commit:** `f1a0bc6` — "Track probe-minus-telescope event time (dt_ns)"
**Cut from:** `main` (`9ae941a`), deliberately *not* from the session's working branch `study/geometric-cut-scan`

PR link, if wanted:
https://github.com/gallog-hash/00_monrad-py/pull/new/feat/coincidence-dt-tracking

## How this came about

The session started as a code-reading question about `src/monrad/coincidence/search.py`.
That surfaced a real observation from the user: stage 2 pairs events on time
proximity alone, so stage 5 can pair a probe event whose timestamp *follows*
the telescope's — even though the probe sits above the telescope and should
physically fire first.

Verified: there is no Δt sign check, no Δt magnitude check, and no cluster
time-span check anywhere between stage 2 and stage 5. The probe's
`TimedEvent` was bound to `_ev` and discarded. The conclusion — that the
ordering is real but unmeasurable per event, while the *distribution* is a
genuine calibration observable — is written up in **`DESIGN.md` §5.3**
(new in this commit). **Read that section rather than re-deriving it here**;
it carries the time-of-flight-vs-noise-floor table and the argument against a
one-sided cut.

## What shipped

Instrumentation only — **no gate was added, and no accepted-coincidence count
changes**. See commit `f1a0bc6` for the full rationale and `git show f1a0bc6`
for the diff. One-line map:

| file | role |
|---|---|
| `src/monrad/pose/types.py` | `Coincidence.dt_ns`, `DecodeReport.dt_ns` |
| `src/monrad/pose/fitter.py` | keeps the probe `TimedEvent`; stamps `t_ns` on the telescope rejection returns so Δt is reported on rejections too |
| `src/monrad/pose/dt.py` *(new)* | `dt_stats` / `dt_histogram` / `dt_counter` / `DtStats` — shared by both report surfaces |
| `scripts/run_pipeline.py` | whole-run Δt summary + 20 ns-binned histogram |
| `src/monrad/monitor/timeseries.py` | `dt_mean_ns` / `dt_std_ns` per window → CSV + 4th plot panel |
| `tests/test_coincidence_dt.py` *(new)* | 27 tests |

Design decision worth not re-litigating: the stats helpers live in the
**library**, not in `run_pipeline.py`. Scripts here are not importable by the
test suite, and the monitor and pipeline report had already drifted to two
different σ conventions (population vs sample). One implementation now,
sample σ throughout.

## Verification (done — do not re-run unless something regresses)

- `uv run pytest -q` → **518 passed** (491 pre-existing + 27 new). Takes
  **~20 minutes**; budget for it, or run the affected subset
  (`tests/test_coincidence_dt.py tests/test_stage2.py tests/test_pose_gates.py`
  is ~0.5 s and covers most of the new logic).
- `uv run ruff check .` and `ruff format --check .` → clean. Pre-commit hooks
  also ran green on the commit.
- End-to-end on synthetic data: the `run_pipeline.py` Δt block renders, and
  `monrad-monitor` writes the two new CSV columns and a correct 4-panel PNG.

## ⚠️ The measurement itself has NOT been made

**Synthetic data gives Δt ≡ 0.** `monrad.synthetic.generate()` has no
inter-detector timing jitter model — both detectors receive identical
timestamps. Every e2e check above therefore shows a single histogram bin at
zero. That proves the plumbing, and **nothing physical**. The instrumentation
has never been pointed at a real acquisition.

This is the obvious next session's work:

1. Run `scripts/run_pipeline.py` against `data/0_testLab_20210723`
   (`Base` + `Probe_0`, `--z-tel 0 -1340 -670`) and read the Δt block.
2. Run `monrad-monitor` over the same acquisition and plot `dt_mean_ns` per
   window. There is an independent prior observation to check it against: a
   **±120 ns Δt swing** across that run, recorded as benign and uncorrelated
   with `z_p` (memory `testlab-20210723-anomaly-root-cause`). If the new
   per-window column does not reproduce that swing, something is wrong with
   the instrumentation, not with the data.
3. For a second, independent geometry: `data/0_testLab_20220204` is present
   on this machine but its plane z-order/probe size were not established in
   this session — check before trusting defaults (`0_testLab_20210723` needs
   `--z-tel 0 -1340 -670`, memory `testlab-20210723-plane-z-order`, and a
   400 mm / 40-channel probe, memory `testlab-20210723-probe-size`). The
   `testMili_20220905` acquisition referenced in memory
   `testmili-20220905-plane-config` is **not** under `data/` on this machine;
   `data/` holds only `0_testLab_20210723`, `0_testLab_20220204` and
   `2023_Probe_New_Test_1`.

Then — and only then — the two decisions this measurement exists to inform:

- **Tighten the stage-2 coincidence window** from 200 ns (DESIGN.md §5 has
  wanted ≈100 ns pending exactly this measurement). Read the mean as the
  constant inter-detector offset and the std as the combined timing
  resolution. Note there is **no CLI flag for it on this branch** —
  `--coincidence-window-ns` exists only on `study/geometric-cut-scan`
  (commit `7a76882`); here the value is `coincidence.WINDOW_NS_DEFAULT` and
  `monitor.io.stream_coincidences(window_ns=...)`. Beware the name collision:
  `monitor.timeseries`'s own `window_ns` is the **monitoring** window derived
  from `--window-s`, an unrelated quantity.
- **Whether to add a symmetric `|Δt| ≤ window_ns` pair check** — see below.

## Known open item: multi-probe cluster chaining

Not a defect introduced here; found while reading `search.py` and left
unfixed on purpose.

Stage 2 clusters by transitive closure (consecutive gaps ≤ `window_ns`) while
stage 5 filters per probe, ignoring *other* probes' events. So in a
multi-probe run a chain of foreign-probe events can stretch a cluster until an
accepted telescope–probe pair sits **further apart than `window_ns`** — up to
≈ `k × window_ns` with `k` probes. Demonstrated in-session with a 3-detector
case: a 200 ns window admitted a pair at Δt = −380 ns.

Single-probe runs are immune (any third event is a duplicate on one of the two
detectors → `ambiguous_cluster`).

A symmetric `abs(dt_ns) <= window_ns` check in `_decode_cluster` is the honest
fix. It was **deliberately not added in this commit**: it changes which
coincidences the fit accepts, and adding it simultaneously would mean
measuring the Δt distribution through the very cut the measurement is meant to
justify. Do step 1–3 above first — the tails beyond ±`window_ns` will be
visible in the histogram. `PoseFitter` does not currently receive `window_ns`;
it would need threading.

Note this is untestable against `0_testLab_20210723`'s multiprobe setup:
`Probe_0` and `Probe_1` there are byte-identical duplicate data (memory
`testlab-20210723-probe0-probe1-identical`), so both streams carry the same
timestamps and chaining never lengthens anything.

## Do NOT add a one-sided (sign) cut

The user's original observation was correct physics, and the temptation to act
on it directly is the trap. DESIGN.md §5.3 records why: the per-event sign is
near-random at this resolution, so a `prb.t_ns <= tel.t_ns` cut discards ~half
of genuine coincidences, and — because the Δt mean itself wanders over a run —
it gives the pipeline a **time-dependent acceptance**, manufacturing a
spurious slow drift in exactly the observable `monrad-monitor` measures. This
is the same failure mode as the `--max-track-slope` / `|b|` gate that was
implemented, proven wrong and reverted (memory `wide-track-cut-gate-shipped`).

If the measurement turns out to show a mean well separated from zero relative
to σ, a one-sided window becomes defensible — but that is a conclusion from
data, not a starting assumption.

## Unrelated gap noticed, not fixed

`DESIGN.md` §5.1 specifies a `max_lag_s` (default 5 s) warning when the k-way
heap stalls on a slow stream. It is not implemented — `log` in
`src/monrad/coincidence/search.py` is created and never used. Harmless (the
heap absorbs latency naturally), but the documented diagnostic does not exist.

## Environment notes

- The **GitHub MCP server failed to connect** this whole session:
  `plugin:github:github (400): Authorization header is badly formatted`. The
  `GITHUB_PERSONAL_ACCESS_TOKEN` in `~/.claude/settings.json` is a committed
  placeholder that must be replaced with this machine's real token locally
  (see `~/.claude/CLAUDE.md`). Until then, use `gh` CLI for PR work.
- `study/geometric-cut-scan` is the branch the user was on when this started
  (geometric-cut scan harness, `mahal_cut`/`max_per_plane`/`window_ns` flags).
  It is untouched by this work and has its own handoff committed on it. This
  Δt work was split onto `main` precisely because it is independent of that
  study.
- The full test suite is slow (~20 min). Two concurrent `pytest` runs will
  roughly double that — run one at a time.
- `macros/monitor.args` and `macros/multiprobe.args` are **placeholders**
  (`data/tel_dir` / `data/prb_dir`) — fill in real paths before `@`-loading
  them, despite what `CLAUDE.md`'s example commands imply.

## Suggested skills

- **`astral:uv`** — every command in this repo goes through `uv run`
  (`uv sync`, `uv run pytest`). Invoke before touching dependencies or
  the test invocation.
- **`astral:ruff`** — mandated by `CLAUDE.md`, and CI blocks on
  `ruff check .` + `ruff format --check .`. Note there is deliberately **no**
  Ruff config; do not add one.
- **`code-review`** — if the `|Δt| ≤ window_ns` gate gets implemented. It
  changes accepted-coincidence counts, which is exactly the class of change
  that has produced real-data-only bugs here before (memory
  `rigidity-footprint-gates-shipped`).
- **`claude-md-management:revise-claude-md`** — only if a new CLI flag lands;
  `CLAUDE.md`'s command block documents the physics-cut flags and would go
  stale.
