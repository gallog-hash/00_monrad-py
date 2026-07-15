# monrad

Muon coincidence and probe alignment pipeline for BuS_Tracker detector systems.

Given raw files from a muon telescope and one or more probes, the pipeline:
1. reconstructs absolute UTC timestamps from GPS-disciplined clock counters,
2. identifies time-coincident events across detectors (200 ns window),
3. decodes strip positions from the folded fiber × ribbon encoding,
4. calibrates the telescope's internal geometry, and
5. fits each probe's pose (translation, rotation, depth) relative to the telescope.

The full algorithm is described in `DESIGN.md`.

## Installation

```bash
uv sync
```

Installs the package (editable) together with the `dev` dependency group
(pytest, scipy, matplotlib, plotly, ruff). Requires Python ≥ 3.10 and
NumPy ≥ 1.24.

## Data layout

An acquisition produces one directory per detector, each containing:

| File | Description |
|---|---|
| `*_header.txt` | INI-style config: clock frequency, GPS UBX-TIM-TM2 anchor |
| `yyyyMMdd_hhmmss_GPS.bin` | Timing stream — 64-bit records (52-bit clock, 11-bit GEN, 1-bit FLAG) |
| `yyyyMMdd_hhmmss.bin` | Position stream — rows of 64-bit words (20-bit X, 20-bit Y, 11-bit GEN), grouped in blocks of 16 |

Files are written in 5-minute chunks; the pipeline stitches them into one logical stream per detector.

## Quick usage

### Inspect raw files

```python
from monrad.decoders.header import parse_header, decode_ubx_tm2

modules = parse_header("20230418_191621_header.txt")
# GPS string keys are named GPS_String_00, GPS_String_01, …
gps_bytes = next(v for k, v in modules["GPS"].items()
                 if k.startswith("GPS_String"))
gps_frame = decode_ubx_tm2(gps_bytes)
print(gps_frame["timeR"])   # UTC datetime of the TIMEPULSE rising edge
print(gps_frame["accEst"])  # timing accuracy estimate in ns
```

```bash
monrad-decode-header data/.../20230418_191621_header.txt

monrad-decode-gps data/.../20230418_192121_GPS.bin
monrad-decode-gps data/.../20230418_192121_GPS.bin --csv out.csv

monrad-decode-bin data/.../20230418_192121.bin --or 5   # first 5 event groups
monrad-decode-bin data/.../20230418_192121.bin --csv out.csv
```

### Run the full pipeline (CLI smoke test)

```bash
python scripts/run_pipeline.py \
    --telescope data/telescope \
    --probe     data/probe \
    --out       pipeline_out
```

Prints event counts, alignment corrections, coincidence count, hit quality
breakdown, and fitted pose parameters.  Saves a plain-text summary to
`pipeline_out/summary.txt`.

### Run the full pipeline (Python API)

```python
import numpy as np
from pathlib import Path
from monrad.timing import reconstruct_stream, load_header_params, find_file_pairs
from monrad.coincidence import coincidence_stream
from monrad.reconstruction import decode_position
from monrad.alignment import AlignmentAccumulator
from monrad.pose import PoseFitter

tel_dir = Path("data/telescope")
prb_dir = Path("data/probe")

# Header files may carry a numeric suffix (e.g. *_header000.txt).
tel_utc0, tel_f0 = load_header_params(next(tel_dir.glob("*_header*.txt")))
prb_utc0, prb_f0 = load_header_params(next(prb_dir.glob("*_header*.txt")))
tel_gps, tel_pos = find_file_pairs(tel_dir)
prb_gps, prb_pos = find_file_pairs(prb_dir)

# Pass 1: telescope alignment (stage 4).
accum = AlignmentAccumulator(z_tel=np.array([0., 400., 800.]))
for _ev, ref in reconstruct_stream(tel_gps, tel_pos, tel_utc0, tel_f0):
    accum.add(decode_position(ref, tel_pos, n_cols=3))
alignment = accum.flush()

# Pass 2: coincidence search + probe pose fit (stages 2 + 5).
# Open two independent telescope streams — do NOT tee a single one.
tel_stream = reconstruct_stream(tel_gps, tel_pos, tel_utc0, tel_f0)
prb_stream = reconstruct_stream(prb_gps, prb_pos, prb_utc0, prb_f0)

fitter = PoseFitter(
    tel_z=np.array([0., 400., 800.]),
    alignment=alignment,
    tel_id=0,
    prb_id=1,
    tel_pos_paths=tel_pos,
    prb_pos_paths=prb_pos,
)
for cluster in coincidence_stream([tel_stream, prb_stream], detector_ids=[0, 1]):
    fitter.add(cluster)

result = fitter.flush()
if result is not None:
    print(f"t_x={result.t_x:.1f} mm  t_y={result.t_y:.1f} mm  "
          f"theta={np.degrees(result.theta):.1f} deg  z_p={result.z_p:.1f} mm")
```

### Monitor probe resolution and position

Two console scripts build on the pipeline to characterize and track the probe
(`monrad.monitor`):

```bash
# Probe-resolution study (fully synthetic): σ(N) sweeps over depth/offset,
# N_required tables, and diagnostic plots written under the output directory.
monrad-resolution --out reports/resolution

# Monitoring of a real acquisition: stream coincidences, fit one probe pose per
# batch, and report per-batch centre uncertainty (writes pose_timeseries.csv +
# .png).  By default batches are count-based: one fit per --min-fit
# coincidences.  Pass --window-s for hybrid windows: each spans at least
# --window-s seconds AND holds at least --min-fit coincidences (whichever is
# longer), so sparse windows stretch to reach the count.
monrad-monitor \
    --telescope data/telescope \
    --probe     data/probe \
    --z-tel     0 400 800 \
    --min-fit   30 \
    --out       pipeline_out/monitor

# Multi-probe monitoring: N probes sharing one telescope acquisition, fit
# independently (--probe is repeatable).  Per-probe overrides like
# --n-probe-ch and --fibers-per-ribbon take either one value (broadcast to
# every probe) or one value per --probe, in the same order.
monrad-multiprobe \
    --telescope data/telescope \
    --probe     data/probe1 \
    --probe     data/probe2 \
    --z-tel     0 400 800 \
    --n-probe-ch 30 40 \
    --min-fit   30 \
    --out       pipeline_out/multiprobe
```

Both scripts' flag lists have grown long enough that a run is easier to keep
in a macro file than to retype: any `@path/to/file.args` argument is expanded
as one flag per line (`#` comments and blank lines ignored), and flags typed
directly on the command line after the `@file` still apply on top of it. See
`macros/monitor.args` / `macros/multiprobe.args` for barebone templates.

### Daily alignment calibration + hardware-drift monitor

The telescope stack is rigidly mounted, so its internal alignment
(DESIGN.md §7) is a stable, once-a-day calibration — not something worth
refitting on every monitoring run. `monrad-align` computes the correction
**once per day** from the first few telescope files of that day, saves it as a
reusable artifact, and appends the fit to a running drift log:

```bash
# Fit the earliest day in the directory (or pick one with --date YYYYMMDD)
# from its first --n-files pairs (default 3). Writes alignment_<date>.json,
# updates alignment_history.csv, and regenerates alignment_history.png.
monrad-align \
    --telescope data/telescope \
    --z-tel     0 400 800 \
    --out       pipeline_out/alignment
```

The fit doubles as a **hardware monitor**: `fit_telescope_alignment` flags
`needs_correction` when any plane's offset / rotation / z-offset / tilt exceeds
a mechanical limit. Each day's per-plane parameters are appended to
`alignment_history.csv` (one row per date, idempotent — re-running a day
replaces its row) and plotted against date in `alignment_history.png` with the
limits drawn in, so drift is visible over time. A breach is logged loudly, but
the tool still exits 0 — it reports, it does not gate.

Feed the saved correction to the monitoring drivers with `--alignment`, which
loads it and **skips the in-run alignment fit** (dropping a full telescope
pass). The saved `--z-tel` must match the run's — the `delta_z`/`tilt` fit is
z-order-dependent, and this is enforced on load:

```bash
monrad-monitor @macros/monitor.args \
    --alignment pipeline_out/alignment/alignment_20230418.json
```

```bash
monrad-monitor @macros/monitor.args --out pipeline_out/monitor

# CLI flags after the @file override single-value flags (e.g. --out here),
# but --probe is append-only in monrad-multiprobe: a CLI --probe on top of a
# macro file adds a probe rather than replacing the file's list.
monrad-multiprobe @macros/multiprobe.args --out pipeline_out/multiprobe
```

## Package structure

Each stage is a domain package whose public API is re-exported from its
`__init__.py`:

```
src/monrad/
    decoders/         # low-level format readers
        header.py     # header.txt parser and UBX-TIM-TM2 decoder
        gps.py        # *_GPS.bin reader (clock ticks, GEN, FLAG)
        position.py   # *.bin reader, OR-fold, hit reconstruction
    timing/           # stage 1: time reconstruction → reconstruct_stream()
    coincidence/      # stage 2: coincidence search  → coincidence_stream()
    reconstruction/   # stage 3: position decoding   → decode_position()
    alignment/        # stage 4: telescope alignment → AlignmentAccumulator
                      #   io.py: save_alignment/load_alignment (JSON)
    pose/             # stage 5: probe pose fit       → PoseFitter
    monitor/          # monitoring drivers (resolution, timeseries,
                      #   multiprobe, align — daily calibration/drift monitor)
    synthetic/        # synthetic-data generator (for testing)
```

## Development

Run tests:

```bash
pytest
```

Per-stage tests live in `tests/test_stage{1..5}.py`. The full streaming
pipeline (stages 1–5 end-to-end with a memory bound) is in
`tests/test_pipeline_stream.py`. Synthetic input is generated by
`monrad.synthetic.generate()`.

The authoritative bit-level reference for each format is the corresponding
decoder module; if `DESIGN.md` and the code disagree, the code wins.
