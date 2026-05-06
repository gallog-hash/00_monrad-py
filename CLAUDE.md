# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Editable install (run once, or after adding dependencies)
pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test file or test function
pytest tests/test_foo.py
pytest tests/test_foo.py::test_bar

# Inspect a GPS timing file
monrad-decode-gps data/.../20230418_192121_GPS.bin
monrad-decode-gps data/.../20230418_192121_GPS.bin --csv out.csv

# Inspect a position file (--or shows the OR-folded hit reconstruction)
monrad-decode-bin data/.../20230418_192121.bin --or 5
monrad-decode-bin data/.../20230418_192121.bin --csv out.csv
```

No linter is configured yet. Python ≥ 3.10 is required (`int | None` and `tuple[A, B]` generics are used throughout). Maximum line length is 80 characters.

## Architecture

`DESIGN.md` is the authoritative algorithm reference. When code and `DESIGN.md` disagree, the code wins. Read §9 of `DESIGN.md` first — the synthetic end-to-end unit test described there is the intended starting point for new pipeline code.

### Source layout

```
src/monrad/
    decoders/        # low-level format readers (the only code that exists so far)
        header.py    # parse_header() + decode_ubx_tm2()
        gps.py       # GPSDecoder — reads *_GPS.bin
        position.py  # BinDecoder  — reads *.bin, reconstructs hits
```

### The five pipeline stages (not yet implemented)

| Stage | Input | Output |
|---|---|---|
| 1 — time reconstruction | `*_GPS.bin` + header per detector | `(t_ns, evt_seq, quality)` stream |
| 2 — coincidence search | n+1 time-sorted streams | clusters of `(detector, evt_seq, t_ns, quality)` |
| 3 — position decoding | `*.bin` + evt_seq lookup | `(x, y, σ_x, σ_y, quality)` per plane |
| 4 — telescope alignment | all telescope events | per-plane offsets/rotations (parallel to stages 2–3) |
| 5 — probe pose fit | coincidence events, corrected telescope geometry | `(t_x, t_y, θ, z_p)` + covariance |

Stages 1–3 share the same logic for telescope and probes; only the event selection differs. Stage 3 is a procedure called by both stage 4 (all telescope events) and stage 5 (coincidence survivors only).

### Key invariants to preserve

- **Integer nanoseconds throughout stage 1.** Use `f_local = (C_{k+1} − C_k) / Δsec` (measured PPS interval, not the nominal `f₀`) to absorb oscillator drift. Never use floats for `t_ns`.
- **`*.bin` rows come in blocks of 16** (one 80 ns acquisition window). Each block must be bitwise-OR'd across all 16 rows before hit reconstruction. Row count must be a multiple of 16.
- **`*.bin` and `*_GPS.bin` are always decoded as a pair.** Row count / 16 in `*.bin` must equal the number of event records (FLAG=0) in `*_GPS.bin`. GEN fields must agree.
- **5-minute file boundaries are transparent.** `evt_seq`, the PPS chain, and 16-row blocks all continue across file boundaries. The pipeline stitches split blocks when the DAQ rotates files mid-block.
- **GEN is 11-bit and wraps every 2048 events.** Always use the unwrapped `evt_seq` (monotonically assigned as events are encountered) for cross-file bookkeeping.

### Detector geometry

- Telescope: 3 planes, 99 channels per axis, 100 cm × 100 cm active area.
- Probe: 1 plane, 30 cm × 30 cm active area, channel count unknown a priori.
- Channel → coordinate: `coord_mm = (ch + 0.5) × 10 mm`, channel 0 at one physical edge.
- Fiber × ribbon encoding: `ch = 10 × ribbon_bit + fiber_bit` (both are LSB-indexed bit positions in the respective 10-bit half of the 20-bit X or Y field).
