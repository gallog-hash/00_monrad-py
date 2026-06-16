# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Editable install + dev tooling (run once, or after adding dependencies)
# All dev deps (pytest, scipy, matplotlib, ruff) live in the `dev`
# dependency-group, which uv installs by default.
uv sync

# Run all tests
uv run pytest

# Run a single test file or test function
pytest tests/test_foo.py
pytest tests/test_foo.py::test_bar

# Inspect a GPS timing file
monrad-decode-gps data/.../20230418_192121_GPS.bin
monrad-decode-gps data/.../20230418_192121_GPS.bin --csv out.csv

# Inspect a position file (--or shows the OR-folded hit reconstruction)
monrad-decode-bin data/.../20230418_192121.bin --or 5
monrad-decode-bin data/.../20230418_192121.bin --csv out.csv

# Run the full pipeline (stages 1–5) against real data
python scripts/run_pipeline.py --telescope <tel_dir> --probe <prb_dir>
python scripts/run_pipeline.py --telescope <tel_dir> --probe <prb_dir> \
    --z-tel 0 400 800 --tot-thresh 2 --tot-weights
```

## Linting and formatting

This project uses Ruff for both linting and formatting. Always invoke
through `uv run`. When working with Ruff, invoke `/astral:ruff` to follow
Astral's recommended usage.

- Lint: `uv run ruff check .`
- Lint and auto-fix: `uv run ruff check --fix .`
- Format: `uv run ruff format .`
- Configuration lives in `pyproject.toml` under `[tool.ruff]`.

## Architecture

`DESIGN.md` is the authoritative algorithm reference. When code and `DESIGN.md` disagree, the code wins. Read §11 of `DESIGN.md` first — the synthetic end-to-end test described there is the intended starting point for new pipeline code.

### Source layout

```
src/monrad/
    decoders/        # low-level format readers
        header.py    # parse_header() + decode_ubx_tm2()
        gps.py       # GPSDecoder — reads *_GPS.bin
        position.py  # BinDecoder  — reads *.bin, reconstructs hits
    synth.py         # generate() — synthetic test-data generator
    stage1.py        # reconstruct_stream(), load_header_params(), find_file_pairs()
    stage2.py        # coincidence_stream()
    stage3.py        # Hit, decode_position()
    stage4.py        # AlignmentAccumulator, AlignmentCorrection, fit_telescope_alignment()
    stage5.py        # PoseFitter, PoseResult, fit_probe_pose()
```

### The five pipeline stages

| Stage | Input | Output |
|---|---|---|
| 1 — time reconstruction | `*_GPS.bin` + header per detector | `Iterator[(TimedEvent, PosRef)]` |
| 2 — coincidence search | n+1 `reconstruct_stream()` iterators | `Iterator[list[(det_id, TimedEvent, PosRef)]]` |
| 3 — position decoding | `PosRef` + `*.bin` paths | `list[Hit | None]` (one per plane) |
| 4 — telescope alignment | all telescope events | per-plane offsets/rotations (parallel to stages 2–3) |
| 5 — probe pose fit | coincidence events + `AlignmentCorrection` | `PoseResult`: `(t_x, t_y, θ, z_p)` + covariance |

Stages 1–3 share the same logic for telescope and probes; only the event selection differs. Stage 3 is a procedure called by both stage 4 (all telescope events) and stage 5 (coincidence survivors only).

### Streaming design

All stage boundaries are iterator boundaries — no stage accumulates a full
list before the next stage starts. Peak RAM is bounded (~1 s of events per
detector in stage 1; the 200 ns window in stage 2).

**Two-stream pattern for stages 4 + 5.** Stage 4 consumes *all* telescope
events; stage 5 consumes only coincident ones. Pass two independent
`reconstruct_stream()` calls — one to `AlignmentAccumulator`, one to
`coincidence_stream()`. Do **not** `itertools.tee` a single stream; the
telescope files are small and iterating them twice is cheaper than the
buffer `tee` would need.

### Tests

Per-stage tests: `tests/test_stage{1..5}.py`. Full streaming pipeline
(stages 1–5 end-to-end with a 512 MB memory bound):
`tests/test_pipeline_stream.py`. All tests use synthetic data from
`monrad.synth.generate()`; no real detector files are required.

### Key invariants to preserve

- **Integer nanoseconds throughout stage 1.** Use `f_local = (C_{k+1} − C_k) / Δsec` (measured PPS interval, not the nominal `f₀`) to absorb oscillator drift. Never use floats for `t_ns`.
- **`*.bin` rows come in blocks of 16** (one 80 ns acquisition window). By default each block is bitwise-OR'd across all 16 rows before hit reconstruction (`tot_thresh=1`). `decode_position(tot_thresh=N)` keeps only bits that fired in ≥ N rows — an intentional noise filter, not a bug. Row count must be a multiple of 16.
- **`*.bin` and `*_GPS.bin` are always decoded as a pair.** Row count / 16 in `*.bin` must equal the number of event records (FLAG=0) in `*_GPS.bin`. GEN fields must agree.
- **5-minute file boundaries are transparent.** `evt_seq`, the PPS chain, and 16-row blocks all continue across file boundaries. The pipeline stitches split blocks when the DAQ rotates files mid-block.
- **GEN is 11-bit and wraps every 2048 events.** Always use the unwrapped `evt_seq` (monotonically assigned as events are encountered) for cross-file bookkeeping.

### Detector geometry

- Telescope: 3 planes, 99 channels per axis, 100 cm × 100 cm active area.
- Probe: 1 plane, 30 cm × 30 cm active area, channel count unknown a priori.
- Channel → coordinate: `coord_mm = (ch + 0.5) × 10 mm`, channel 0 at one physical edge.
- Fiber × ribbon encoding: `ch = 10 × ribbon_bit + fiber_bit` (both are LSB-indexed bit positions in the respective 10-bit half of the 20-bit X or Y field).
