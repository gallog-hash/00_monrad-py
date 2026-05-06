# monrad

Muon coincidence and probe alignment pipeline for BuS_Tracker detector systems.

Given raw files from a muon telescope and one or more probes, the pipeline:
1. reconstructs absolute UTC timestamps from GPS-disciplined clock counters,
2. identifies time-coincident events across detectors (200 ns window),
3. decodes strip positions from the folded fiber × ribbon encoding, and
4. fits each probe's pose (translation, rotation, depth) relative to the telescope.

The full algorithm is described in `DESIGN.md`.

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.10 and NumPy ≥ 1.24.

## Data layout

An acquisition produces one directory per detector, each containing:

| File | Description |
|---|---|
| `*_header.txt` | INI-style config: clock frequency, GPS UBX-TIM-TM2 anchor |
| `yyyyMMdd_hhmmss_GPS.bin` | Timing stream — 64-bit records (52-bit clock, 11-bit GEN, 1-bit FLAG) |
| `yyyyMMdd_hhmmss.bin` | Position stream — rows of 64-bit words (20-bit X, 20-bit Y, 11-bit GEN), grouped in blocks of 16 |

Files are written in 5-minute chunks; the pipeline stitches them into one logical stream per detector.

## Quick usage

### Decode a header file

```python
from monrad.decoders.header import parse_header, decode_ubx_tm2

modules = parse_header("20230418_191621_header.txt")
gps_frame = decode_ubx_tm2(modules["GPS"]["GPS_String"])
print(gps_frame["timeR"])   # UTC datetime of the TIMEPULSE rising edge
print(gps_frame["accEst"])  # timing accuracy estimate in ns
```

### Inspect a GPS timing file

```bash
monrad-decode-gps data/.../20230418_192121_GPS.bin
monrad-decode-gps data/.../20230418_192121_GPS.bin --csv out.csv
```

### Inspect a position file

```bash
monrad-decode-bin data/.../20230418_192121.bin --or 5   # first 5 event groups
monrad-decode-bin data/.../20230418_192121.bin --csv out.csv
```

## Package structure

```
src/monrad/
    decoders/
        header.py    # header.txt parser and UBX-TIM-TM2 decoder
        gps.py       # *_GPS.bin reader (clock ticks, GEN, FLAG)
        position.py  # *.bin reader, OR-fold, hit reconstruction
```

## Development

Run tests:

```bash
pytest
```

The authoritative bit-level reference for each format is the corresponding decoder module; if `DESIGN.md` and the code disagree, the code wins.
