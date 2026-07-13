# Muon coincidence and probe alignment pipeline — design document

This document specifies the algorithms used to (1) reconstruct time coincidences
between a muon telescope and one or more probes from raw detector files, and
(2) fit each probe's pose (position and rotation) relative to the telescope from
the coincidences thus identified.

It is organised top-down: first the data model on disk, then the data-flow
model, then the processing stages, then the open items that should be confirmed
against real data before code is finalised. Each section includes both the
*what* and the *why* — the rationale is preserved so that future revisions of
the algorithm can be made with full context.

The reference decoders for the on-disk formats are
`decoders/header.py`, `decoders/gps.py`, and `decoders/position.py`. This
document treats their bit-level behaviour as authoritative; if anything
stated here disagrees with the scripts, the scripts win.

> **Note.** This document incorporates the streaming redesign previously
> described in `DESIGN_UPDATE.md`. The algorithms are unchanged; the
> data-flow model (§3) and the stage-level implementation notes (§4–§8)
> have been updated to reflect the generator-based pipeline that is
> implemented in the code.


## 1. Hardware and acquisition model

There are `n + 1` detectors: one **telescope** with three position-sensitive
planes, and `n` **probes**, each with a single position-sensitive plane.

A position-sensitive plane is built from 1 cm-wide plastic scintillator bars
arranged in two perpendicular layers — one for X, one for Y. Each strip is
read out as a single channel. The telescope planes have **99 channels per
axis** spanning a nominal 100 cm × 100 cm active area (the slight excess
reflects the optical coatings on the bars, which extend their effective width
beyond 1 cm). The probes use the same plastic scintillator bars as the telescope. 
They typically use 30 × 30 or 40 × 40 bars, providing a nominal active area of 
30 cm² or 40 cm², respectively. 

Channel 0 is at one physical edge of the active area. The three telescope
planes are mounted with channel 0 aligned across planes — that is, the
telescope's X axis (and separately the Y axis) is internally consistent across
its three planes. The probes' X and Y axes have arbitrary orientation with
respect to the telescope's; recovering that orientation is the goal of stage 5.

Each detector has its own management system, its own GPS receiver, and its
own data acquisition. Acquisition starts independently per detector. Once
running, the system writes one `*_header.txt` once at start, then
every five minutes saves a pair of files — `yyyyMMdd_hhmmss.bin` (positions)
and `yyyyMMdd_hhmmss_GPS.bin` (timing) — until acquisition stops. The 5-minute
file boundaries do not align across detectors.

Event rates are roughly **tens of Hz** for the telescope (tracks crossing all
three planes) and **a few Hz** for each probe.


## 2. Data model

### 2.1 The header file

A small INI-like text file with bracketed module sections (`[J11]`, `[GPS]`,
…) and `key=value` entries. The only field the pipeline needs is:

- The **GPS string** in the `[GPS]` section, written as latin-1 with `\XX` hex
  escapes for non-printable bytes. This is a UBX-TIM-TM2 binary frame from
  the receiver, decoded by `decode_ubx_tm2()` in `decoders/header.py`. There is
  exactly **one such GPS string per acquisition**, capturing the absolute UTC
  time of one TIMEPULSE rising edge near the start of the run, plus an
  accuracy estimate `accEst` (in ns). Typical `accEst` is 20–50 ns.

### 2.2 `*_GPS.bin` — the timing stream

```
bytes 0–3        u32   number of records (little-endian)
bytes 4 … end    n × 8 bytes of u64 records (little-endian)

Per-record bit layout (LSB = bit 0):
  bits  0–51   tick     52-bit clock counter latch
  bits 52–62   GEN      11-bit event sequence (low 11 bits of the running counter)
  bit  63      FLAG     1 = PPS record, 0 = event record
```

The file mixes two record kinds:

- **Event records** (`FLAG = 0`) — one per detector event. The counter is
  latched at the event's hardware trigger time. GEN advances by one per event.
- **PPS records** (`FLAG = 1`) — one per second, latching the counter at the
  rising edge of the GPS 1 Hz pulse. PPS records do **not** advance GEN; the
  GEN field of a PPS record holds the value GEN had at the moment of the
  pulse.

Records appear in the file in acquisition order.

A 52-bit counter at any plausible `f₀` does not roll over in any practical
acquisition (at 100 MHz it runs for ~520 days), so absolute time within a
single run is unambiguous from the tick value alone.

The 11-bit GEN field, on the other hand, wraps every 2048 events. At the
telescope's rate of tens of Hz, GEN wraps every minute or two — many times
within a single 5-minute file. We therefore reconstruct an unwrapped event
sequence number (call it **`evt_seq`**) by tracking GEN decreases as we walk
the event records. A counter **`RESET`** is incremented every time GEN
decreases relative to its predecessor; `evt_seq = RESET · 2048 + (GEN − GEN_at_segment_start)`,
or equivalently: assign `evt_seq = 0, 1, 2, …` monotonically to event records
as they are encountered. RESET and `evt_seq` continue across the detector's
5-minute file boundaries — there is one logical stream per detector.

### 2.3 `*.bin` — the position stream

```
bytes 0–3        u32   number of rows (little-endian)
bytes 4–7        u32   number of u64 columns per row (little-endian)
bytes 8 … end    n_rows × n_cols × 8 bytes of u64 records (little-endian, row-major)

Per-u64 bit layout (LSB = bit 0):
  bits  0– 9   Y_ribbon  10-bit hit mask
  bits 10–19   Y_fiber   10-bit hit mask
  bits 20–31   unused
  bits 32–41   X_ribbon  10-bit hit mask
  bits 42–51   X_fiber   10-bit hit mask
  bits 52–62   GEN       11-bit event sequence (low 11 bits, matches *_GPS.bin GEN)
  bit  63      unused
```

`n_cols` is **the number of position-sensitive planes** in the detector: 1
for a probe, 3 for the telescope. The `n_cols` u64s on a single row therefore
all describe the same event, on different planes; their GEN fields must
agree.

**Rows are grouped in blocks of 16** representing 16 successive 5 ns samples
of an 80 ns acquisition window per event. All 16 rows in a block share the
same GEN value, and there are no PPS records in `*.bin` — PPS lives only in
`*_GPS.bin`. The position information for an event is recovered by bitwise-OR
across the 16 samples (see §6).

In a clean acquisition, `*.bin` row count is a multiple of 16, and
`*.bin row count / 16` equals the number of non-PPS records in the
corresponding `*_GPS.bin`. The row-count-multiple-of-16 part holds
universally; the count-equality part can be violated by ±1–2 blocks at a
file rotation (the `.bin` and `*_GPS.bin` streams are flushed at slightly
offset points), so the pipeline treats it as a **warning, not a hard
assertion**, and proceeds — see §4.4 and §10 ("Cross-file 16-row block
continuity") for the real-data behaviour.

### 2.4 Position encoding — fiber and ribbon

Each axis's 20 bits encode one or more 1 cm strips firing using a folded
fiber × ribbon scheme: the physical channel index is

```
ch = N · ribbon_bit + fiber_bit
```

where `ribbon_bit` is the LSB-indexed position of the bit set in the 10-bit
ribbon mask and `fiber_bit` likewise for the fiber mask. The 10-bit fiber
mask's bit width is fixed by the readout ASIC, but `N` — the number of those
10 fiber positions actually wired for a given detector — is a **per-detector
parameter**, not a hardware constant: a detector may only connect the first
`N` of the 10 available fiber positions, leaving the rest permanently unset.
`N` defaults to 10 (all positions wired, giving 10 fiber × 10 ribbon = 100
channel codes, comfortably covering the 99-channel telescope). The telescope
is currently assumed fixed at `N = 10`; probes may wire fewer and are
configured via `n_fibers_per_ribbon`/`--fibers-per-ribbon` (default 10).
**Channel 0 is at one physical edge** of the active area.

A clean event has exactly one fiber bit and one ribbon bit set per axis (per
plane), giving an unambiguous channel ("golden hit"). Events with broader
clusters require a small reconstruction step described in §6.


## 3. Data-flow model

Every stage boundary is an iterator boundary. No stage materialises more than
a bounded window of events at once.

```
stage 1 (per detector)          stage 2               stages 4 / 5
─────────────────────           ──────────            ──────────────
reconstruct_stream()   ──────►  coincidence_  ──────► AlignmentAccumulator /
                                stream()               PoseFitter
                       n+1 streams merged via min-heap
```

The maximum in-memory working set at any point in the pipeline is:

- **Stage 1**: one PPS interval (~1 s) of buffered events per detector.
- **Stage 2**: the coincidence window (200 ns) plus inter-detector PPS
  latency (≤ a few seconds).
- **Stages 4 / 5**: one accumulator buffer (configurable; default 10 000
  events for stage 4, 500 coincidences for stage 5).

**Motivation.** At a telescope rate of 10 Hz, one week of continuous
acquisition produces approximately 6 000 000 events per detector. A batch
design that accumulates all events into Python lists before returning costs
roughly 3.4 GB of RAM per detector in stage 1 alone. Running telescope +
two probes concurrently saturates a typical analysis machine. The streaming
design bounds peak RAM to a few hundred megabytes regardless of run length.

Stage 4 (telescope alignment) runs on a dedicated `reconstruct_stream()` for
the telescope, consuming all telescope events. Stage 5 runs on a separate
`coincidence_stream()` that gets its own `reconstruct_stream()` for both
detectors. The telescope GPS and position files are therefore iterated twice,
but since file-header reads cost at most 24 bytes per file and the GPS files
are small, the I/O overhead is negligible.


## 4. Stage 1 — per-detector time reconstruction

Goal: convert each detector's `*_GPS.bin` records into a stream of
`(TimedEvent, PosRef)` pairs, where `t_ns` is the event's UTC time in
integer nanoseconds and `PosRef` carries the event's location in the `*.bin`
files. This is run independently — and identically — for the telescope and
for each probe.

### 4.1 Anchoring with PPS

The header gives one absolute UTC anchor `UTC₀` (decoded from the UBX-TIM-TM2
GPS string). The nominal counter frequency `f₀` is **not** carried in the
header; `load_header_params` uses the fixed `F0_DEFAULT = 100 MHz` (see §10,
"Clock frequency source"). PPS records inside `*_GPS.bin` then provide a
stream of further anchors at 1 Hz, latching the counter value `C_k` at each
successive PPS edge.

Walk the PPS records maintaining `(C_k, N_k)` where `N_k` is the integer
number of seconds since the run started and `N_0 = 0`. For each next PPS
latch `C'`, compute

```
ΔC  = C' − C_last
n   = round(ΔC / f₀)
res = |ΔC − n · f₀| / (n · f₀)
```

If `res ≤ τ` (start at `τ = 1e-4`, tighten later if real data warrants),
accept the pair as spanning exactly `n` seconds. When `n > 1` this implies
`n − 1` dropped pulses — log them but proceed normally. Otherwise mark the
interval `[C_last, C']` as **untrusted** and propagate that flag to any events
falling inside it; do not attempt to invent timestamps.

### 4.2 Event timestamps

Between any two accepted PPS anchors, the locally measured frequency is

```
f_local = (C_{k+1} − C_k) / (N_{k+1} − N_k)
```

and an event at counter `C_e` between them takes

```
t_ns = UTC0_ns + N_k · 10⁹ + ((C_e − C_k) · 10⁹) // f_local
```

Stay in **integer nanoseconds** throughout. A 64-bit unsigned ns counter
covers ~584 years; double-precision floats lose nanosecond resolution within
a single day. Using `f_local` rather than `f₀` quietly absorbs the local
oscillator's short-term drift, which is the whole point of being PPS-disciplined.

The conversion has unavoidable latency: an event between PPS_k and PPS_{k+1}
can only be timestamped once PPS_{k+1} arrives. The per-detector iterator
therefore buffers up to ~1 s of events and emits them in order on each new
good PPS. Events before the first accepted PPS are back-extrapolated using
the `f_local` measured by PPS_1→PPS_2; events after the last good PPS are
forward-extrapolated; both are tagged with a degraded `quality` flag.

### 4.3 Back-extrapolation of pre-PPS_1 events

The streaming design cannot resolve back-extrapolation until PPS_2 is seen.
The procedure is:

1. Buffer all events and PPS records until PPS_2 is observed.
2. At PPS_2: build `_Interval(PPS_1, PPS_2)` as `back_iv`.
3. Back-extrapolate buffered pre-PPS_1 events using `back_iv`, then
   timestamp PPS_1→PPS_2 events normally.
4. From PPS_2 onward: standard one-interval-at-a-time flow.

This introduces a one-time startup latency of at most 2 s. Pre-PPS_1 events
are tagged `Quality.DEGRADED`.

### 4.4 File-boundary handling

The acquisition writes a new `(*_GPS.bin, *.bin)` pair every 5 minutes
without flushing per-detector pipeline state. Concretely:

- GEN and the unwrapped `evt_seq` continue across the boundary.
- The PPS anchoring chain continues across the boundary (the next PPS may
  come from the new file).
- A 16-row block of position data may, in principle, be split across two
  consecutive `*.bin` files if the DAQ closes a file mid-block. The pipeline
  detects this by checking that the row count of each `*.bin` is a multiple
  of 16 *and* that the GEN field of the first row of file `k+1` is the
  successor (mod 2048) of the GEN field in the last row of file `k`. If a
  block straddles the boundary, the two halves are stitched before being
  passed to the position decoder.

Split-block detection must be eager: when opening file `k+1`, check GEN
continuity before yielding any events from file `k` that might belong to a
split block. If the last pending event of file `k` has the same GEN as the
first row of file `k+1`, hold that event and construct a `PosRef` with the
correct `split_rows` value before yielding.

### 4.5 Output

`reconstruct_stream()` is the primary API. It is a generator that yields
`(TimedEvent, PosRef)` pairs in time order, emitting each PPS interval's
events as soon as the closing PPS record is observed.

```
TimedEvent  (t_ns, evt_seq, quality)
PosRef      (file_idx, row_offset, split_rows)
```

`PosRef` is carried inline with the `TimedEvent` and passed directly to
stage 3 decode calls by the caller; no side-table lookup is required.


## 5. Stage 2 — coincidence search

Goal: identify time-coincident clusters of events across the n+1 detector
streams using a 200 ns sliding window.

The window `Δt = 200 ns` was chosen as a comfortable upper bound given the
GPS receiver's `accEst` of 20–50 ns per detector. With two detectors the
combined timing uncertainty adds in quadrature to roughly 30–70 ns — well
inside Δt. The window may be tightened later (likely to ≈ 100 ns) once the
empirical Δt distribution between true telescope-probe coincidences is
measured.

### 5.1 The merge

`coincidence_stream()` consumes n+1 `reconstruct_stream()` iterators via a
k-way min-heap keyed on `t_ns`. It maintains a sliding deque of events within
`[t_now − Δt, t_now]`. On each pop:

1. evict deque entries with `t_ns < t_now − Δt`;
2. append the new event;
3. if the deque now contains entries from ≥ 2 distinct detectors, an open
   cluster exists.

Emit a cluster only when its last-added event falls off the window — i.e.
when no further event within Δt could extend it. This produces transitive-
closure clusters (consecutive events ≤ Δt apart), which is the standard
convention and avoids artifacts from arbitrary seed-choice.

Complexity is `O(N log(n+1))` for `N` total events. At the expected rates
the deque length is almost always 0–1 and the dominant cost is I/O. Random
coincidences can be neglected for now (per assumption).

Each `reconstruct_stream()` buffers up to ~1 s of events waiting for the
next PPS. The k-way heap stalls on the slowest-advancing stream; no special
handling is required since the heap naturally absorbs the latency. If the
heap stalls beyond a configurable `max_lag_s` (default 5 s), a warning is
logged.

### 5.2 Output

A generator yielding clusters, each a list of `(detector_id, TimedEvent,
PosRef)` tuples. `PosRef` is carried transparently so that stage 3 callers
downstream never need to look anything up.


## 6. Stage 3 — position decoding

Goal: decode positions from `*.bin` into hits `(x, y, σ_x, σ_y, quality)`
per plane. This is a **procedure**, not a stage in its own right — it is
invoked by two consumers:

- the **telescope internal alignment** branch (§7), which feeds it the full
  stream of telescope events from §4.5;
- the **probe pose fit** branch (§8), which feeds it the coincidence-surviving
  events from all detectors emitted by §5.2.

Both consumers use exactly the same decoding logic; only the upstream event
selection differs.

### 6.1 Interface

```python
def decode_position(
    pos_ref: PosRef,
    pos_paths: list[Path],
    n_cols: int,
    tot_thresh: int = 1,
    tot_weights: bool = False,
) -> list[Hit]:
```

`pos_ref` is received directly from the stream — no `pos_map` lookup is
needed. Returns one entry per plane (`n_cols` elements), always a real `Hit`
(the list never contains `None`): when reconstruction fails the plane is
returned with quality `'invalid'` or `'unresolved'`, so callers test on
`quality` instead of null-checking.

`tot_thresh` keeps only bits that fired in ≥ `tot_thresh` of the 16 block
rows (1 = plain bitwise OR; 2–4 filters single-row noise). `tot_weights`
weights cluster centroids by each bit's per-row TOT count (no effect on
golden hits). Both back the `--tot-thresh` / `--tot-weights` pipeline flags.

### 6.2 Random access into `*.bin`

For the `(file_idx, row_offset)` in `pos_ref`, read 16 × `n_cols` u64s
starting at `row_offset`. This is a single O(1) seek per event, costing at
most 16 × 3 × 8 = 384 bytes per telescope event and 128 bytes per probe event.

For each of the `n_cols` planes, collapse the 16 samples' X and Y fields to
a single per-axis mask (`reconstruction/hit.py::_or_masks`). At the default
`tot_thresh=1` this is a plain bitwise OR of the 16 samples; at `tot_thresh>1`
it is **not** an OR but a per-bit count filter — a bit is kept only if it
fired in ≥ `tot_thresh` of the 16 rows (§6.1). Verify that all 16 GEN values
within the block agree, and
that the GEN matches `evt_seq mod 2048` from `*_GPS.bin`. A mismatch on
either is a structural error — the file pair is corrupted or the join is
wrong; halt and report.

### 6.3 Validity prefilter

A column is **invalid** if any of the four 10-bit halves of the OR equals
1023 (all bits set, indicating channel saturation), or if either ribbon half
is zero (no ribbon channel fired, so no coordinate can be recovered). This
matches the existing logic in `decoders/position.py::BinDecoder._is_valid`.

### 6.4 Hit reconstruction

For each valid axis (X or Y separately):

1. Find contiguous bit clusters in the fiber and ribbon halves.
2. If exactly one fiber cluster and one ribbon cluster exist, compute the
   candidate channels `{N · r + f : r ∈ ribbon_cluster, f ∈ fiber_cluster}`.
3. If those candidates form a contiguous range of integers, accept; the
   reconstructed channel is the centroid (the mean of the candidates); the
   uncertainty is `σ = (cluster_width × strip_pitch) / √12`, where
   `strip_pitch = 1 cm` and `cluster_width` is the number of candidates.
4. Special case `cluster_width = 1` (single fiber bit and single ribbon bit):
   "golden hit", `σ = strip_pitch / √12 ≈ 2.9 mm`.
5. Otherwise: **unresolved**.

A hit is delivered to the caller only if both X and Y are reconstructed. The
quality flag is `golden`, `cluster`, `unresolved`, or `invalid`.

`unresolved` means an axis did not collapse to one clean channel. When only
**one** axis failed, the axis that *did* reconstruct is retained as a single
candidate hypothesis (its centroid and width) and the failed axis carries its
real multi-candidate list; these per-axis hypotheses (`Hit.candidates_x/y`)
are kept for diagnostics (e.g. `scripts/investigate_single_axis.py`). They are
**not** consumed by a plane-recovery step: Stage 5 enumerates its own per-plane
candidates independently via `reconstruct_plane_candidates` (§8.2) and never
reads these lists.

### 6.5 Channel → physical coordinate

```
coord_mm = (ch + 0.5) × strip_pitch_mm   #   strip_pitch_mm = 10
```

with channel 0 at one physical edge of the active area. The same convention
is used for telescope and probes. Any per-detector edge offset (e.g. a frame
that prevents the leftmost strip from being at exactly x = 0) is absorbed
into the alignment fits in §7 and §8. `ch` here is computed with that
detector's own fiber×ribbon combine factor `N` (§2.4) — `strip_pitch_mm`
stays a single global constant, but `ch` itself is `N`-dependent per
detector.

### 6.6 Time-over-threshold weight and future refinement

The 16 per-sample bit patterns carry information that a plain OR discards.
The number of active samples per strip is a **time-over-threshold (TOT)
quality weight**, and it **is** implemented: with `tot_weights=True` (§6.1,
`_tot_weighted_centroid`) each cluster centroid is weighted by its per-bit
row counts (no effect on golden hits). It backs the `--tot-weights` pipeline
flag.

What remains future work is the *sub-event-window timestamp*: the first
sample in which a strip fires gives a finer-than-80 ns time stamp useful for
time-walk corrections. This is not yet used.


## 7. Stage 4 — telescope internal alignment (parallel branch)

Goal: validate (and if necessary calibrate) the telescope's internal
geometry — that its three planes are mutually parallel and X-Y aligned —
**before** any probe pose fit is attempted.

### 7.1 Why this is its own stage, and why it runs first

The probe pose fit in §8 assumes a perfectly self-consistent telescope: planes
parallel, X axes aligned across planes, Y axes aligned across planes. If that
assumption is wrong by some `Δx_2 ≈ 1 mm` translation of plane 2, every probe
fit silently absorbs that shift into its own `(t_x, t_y)` and rotation θ —
with no diagnostic to detect it, since the probe sees only the *combined*
track passing through all three planes.

The internal alignment must therefore be measured independently of any probe.
It also wants the largest possible track sample with the broadest possible
angular and spatial coverage, for two reasons:

- **Statistics.** Per-plane systematics are detectable down to the strip
  resolution divided by √N. With thousands of telescope tracks the
  statistical floor sits well below 1 mm; with the few-hundred surviving
  coincidences from a 5 m probe configuration it would not.
- **Lever arm.** A per-plane rotation about z manifests as a residual whose
  magnitude grows with distance from the rotation axis, and a per-plane tilt
  (out-of-plane, about x or y) manifests as a residual that grows with track
  angle. Both diagnostics need the full angular and spatial extent of the
  telescope, not the narrow cone selected by a coincidence with a probe.

Concretely, the stage fits a **middle-plane** out-of-plane tilt about x and y
(`tilt_x`/`tilt_y`, §7.3), the per-plane non-parallelism the telescope
mechanics actually permit; outer-plane tilts remain degenerate with track
slope and are left at 0 (§10, "Plane tilt detection"). Rotation about z is
mechanically suppressed by the mounting, so `rotation_z` is fitted only as a
quality monitor (expected ≈ 0), not as a routine correction. (This is the
telescope's *internal* z-rotation; the probe's own orientation θ about z in
§8.1 is a separate, legitimate parameter.)

For all the above reasons this stage runs on **all telescope tracks** via a
dedicated `reconstruct_stream()`, independent of the coincidence pipeline.

### 7.2 Inputs

The full telescope event stream from §4.5, with positions decoded by §6
(applied to all telescope events, not coincidence survivors). Filter to
tracks where all three planes have a valid hit (`golden` or `cluster`). At
tens of Hz over a 5-minute file this yields thousands of tracks per file
and easily tens of thousands per multi-hour acquisition.

### 7.3 Diagnostics

Two complementary tests are run.

**(a) Three-plane fit and per-plane residuals.** Fit a straight line through
all three telescope hits (independent linear fits in `x(z)` and `y(z)`) and
look at the residual on each plane in turn. For plane `k`:

- A non-zero mean of `(x, y)` residual indicates a translational misalignment
  of plane `k` in `(x, y)`.
- A linear correlation between `x`-residual and `y`-position on plane `k` (or
  between `y`-residual and `x`-position) indicates a rotation about z by an
  angle equal to the slope.
- A linear correlation between `x`-residual and the track's x-direction
  cosine indicates a tilt of plane `k` about the y-axis (and likewise for y
  about x). Equivalently, the residual carries a slope×lever-arm term
  `r ≈ φ·b·coord`, distinct from the Z-offset term `r ≈ δz·b` (slope only)
  and the z-rotation term `r ∝ ⊥ coordinate`. For the middle plane these are
  separated by a joint regression of the residual on `(b, b·coord)`; the two
  regressors are strongly correlated for a cosmic-ray sample, so a univariate
  fit per term would cross-contaminate `δz` and `φ`.

**(b) Two-plane prediction.** For each plane `k` in turn, fit a line through
the *other two* planes and predict the hit on plane `k`. Compare to the
measurement. The diagnostic information is the same as (a) but cleaner:
plane `k` is tested against a track it did not help define, so its
contribution to the residual is unbiased.

With three planes, (b) gives three independent views of each per-plane
systematic.

### 7.4 Decision

If all per-plane offsets are below ~1 mm (sub-strip) and rotations below
~1 mrad, the nominal telescope geometry is good as-is and the pipeline
proceeds to §8 with no corrections. If systematics exceed those thresholds,
the recovered offsets and rotations are folded into the telescope geometry
as corrections that propagate to every subsequent line fit in §8.2.

The same applies to the two slope-dependent middle-plane corrections — the
Z offset `δz` (threshold ~5 mm) and the out-of-plane tilt `φ` (threshold
~5 mrad, set above the ~3 mrad statistical floor at ~1000 tracks). Because
the telescope mechanics suppress rotation about z but permit a small tilt
about x or y, `rotation_z` is in practice a quality monitor (expected ≈ 0)
while the tilt is the per-plane non-parallelism actually worth correcting.

Thresholds are set by physics, not statistics: with thousands of tracks the
statistical uncertainty per plane is well below 0.1 mm, so any systematic
above the per-strip resolution of ~3 mm should be visible and worth
correcting.

### 7.5 Accumulator design

Stage 4 is implemented as `AlignmentAccumulator`, which buffers decoded
three-plane hits and emits an `AlignmentCorrection` each time the buffer
reaches `flush_every` hits (default 10 000).

```python
accum = AlignmentAccumulator(flush_every=10_000)
for ev, ref in reconstruct_stream(tel_gps, tel_pos, utc0, f0):
    hit = decode_position(ref, tel_pos_paths, n_cols=3)
    if hit and hit.quality in ('golden', 'cluster'):
        correction = accum.add(hit)
        if correction is not None:
            log.info('Alignment updated: %s', correction)
```

`AlignmentCorrection` carries the per-plane `(Δx, Δy, rotation_z, δz,
tilt_x, tilt_y)` values (the last three non-zero only for the middle plane)
and a `needs_correction` boolean that downstream consumers (stage 5) read to
decide whether to apply it.

### 7.6 Continuous monitoring

Each flush produces a timestamped `AlignmentCorrection`. Persisting these to
a log file indexed by the UTC timestamp of the first hit in the batch would
directly implement continuous drift monitoring — slow changes indicate
mechanical settling or thermal expansion; sudden jumps indicate a discrete
physical event (apparatus bumped, DAQ restarted). **This log export is not
yet implemented**; corrections are only printed to `summary.txt`. A real
drift-log exporter remains an open item (§10, "Diagnostic plots /
alignment drift log").


## 8. Stage 5 — probe pose alignment

Goal: for each probe, fit four parameters `(t_x, t_y, θ, z_p)` describing the
probe's pose relative to the telescope, given the surviving telescope-probe
coincidences.

This stage assumes that §7 has run and that the telescope geometry passed
to it is internally consistent within tolerance. If §7 has reported per-plane
systematics above the strip resolution and they have not been folded into
the telescope geometry as corrections, **stop and fix that first** — every
probe pose returned by this stage will otherwise absorb the telescope's
internal misalignment into its own parameters with no diagnostic to detect
it.

### 8.1 Geometry and parameterisation

We assume — and have validated by §7 — that all telescope planes are
mutually parallel and X-Y aligned (or have been made so by the §7
corrections, including the middle-plane tilt), and that the probe plane is
parallel to the telescope planes. Place the telescope frame so the planes are at
constant `z` values `z₁, z₂, z₃` (the geometric centre of the top plane is
the reference origin in `z`). The probe plane then sits at a single unknown
`z = z_p`, and the probe's own (u, v) coordinates are related to
telescope-frame (x, y) by a translation plus a rotation about z:

```
x = t_x + u cos θ − v sin θ
y = t_y + u sin θ + v cos θ
z = z_p
```

Four unknowns. Each coincidence gives two scalar constraints (the predicted
telescope hit on the probe plane vs. the measured probe hit), so three
coincidences suffice in principle and many more are available in practice.

### 8.2 Telescope line fit per coincidence

For each coincidence, the three telescope hits `(x_k, y_k, z_k)`, `k = 1, 2,
3` define a 3D line via two independent linear least-squares fits in
`x(z) = a_x + b_x · z` and `y(z) = a_y + b_y · z`. Each fit yields the four
parameters and a 4 × 4 covariance `Σ_line` derived from the per-plane
position uncertainties (§6.4) and the corrected plane `z` values from §7.

**Resolving plane ambiguity — combinatorial candidate search.** A telescope
plane is frequently *ambiguous*: a too-low acquisition threshold lets extra
adjacent strips fire, so an axis yields a wide cluster or several plausible
channels rather than one clean hit. Such a pattern is indistinguishable from
that of two overlapping particles. Rather than recover one plane from the
other two, Stage 5 enumerates candidates on **all** planes and lets the
geometry choose:

- `reconstruct_plane_candidates()` (stage 3) returns, per plane, the list of
  plausible `(x, y)` positions — a single candidate when both axes resolve
  cleanly (`golden`/`cluster`), or the full ribbon × fiber cross-product
  (capped at 16/plane) when an axis is ambiguous.
- `PoseFitter._decode_cluster` then searches the Cartesian product
  `cands₀ × cands₁ × cands₂` of the three planes' candidate lists, evaluates
  the weighted line fit above for **every** candidate triple, and keeps the
  triple with the minimum line-fit χ². Discrete candidate disambiguation is
  thus folded into the same χ² the continuous track fit minimises.

Two guards protect this search:

- **zero-candidate plane** — if any plane yields no candidate (e.g. a
  single-half dropout), the triple cannot be formed and the coincidence is
  rejected.
- **anchor plane** (`no_anchor_plane`) — by default (`min_anchor_planes=1`)
  the search requires at least one plane that already decoded to a single
  resolved candidate. With no resolved plane to anchor it, a genuine pile-up
  (two particles in one window) can minimise χ² by coincidence just as a real
  track does; the bit patterns are identical, so the search cannot tell them
  apart. This guard is **tunable, not mandatory**: `min_anchor_planes=0`
  disables it (every cluster is searched — more tracks, far heavier compute,
  pile-up can fabricate a low-χ² track) and `min_anchor_planes=3` tightens it
  to demand every plane already resolved.

The winning triple is fit in the alignment-corrected frame (`coord − δ`), so
the line fit and the χ² cut below use the corrected plane positions. A
coincidence is fit only if a valid anchored triple survives this step.

When a middle-plane tilt has been fitted, the X and Y fits use *different*
plane `z` values: a tilted plane reports its hit at an effective
`z = z_k + φ·coord` (a tilt about y shifts the x measurement, a tilt about x
shifts y), so the x fit uses `z_k + tilt_y·x` and the y fit uses
`z_k + tilt_x·y`. This places each measurement at its true z and removes the
tilt exactly, without iteration, since the coordinate setting the shift is
itself measured.

A **track quality cut** (e.g. χ² of the line fit < 4, equivalent to ≤ 1 strip
of residual on each plane) is applied here to remove ghost tracks before
they corrupt the alignment fit. Loose cuts are preferred over tight ones at
this stage.

Each accepted triple is labelled per plane by its winning candidate's own
quality (`golden`/`cluster`); this `tel_quality` is surfaced by
`run_pipeline.py` (Stage 3 funnel) and carried on the `Coincidence`, but
nothing in the fit reads it.

### 8.3 The residual

For each coincidence `i`, the predicted telescope position at the probe
plane is

```
x_pred(z_p) = a_x,i + b_x,i · z_p
y_pred(z_p) = a_y,i + b_y,i · z_p
```

and the measured probe hit, mapped into the telescope frame, is

```
x_meas(θ, t_x) = t_x + u_i cos θ − v_i sin θ
y_meas(θ, t_y) = t_y + u_i sin θ + v_i cos θ
```

The residual is `r_i = (x_meas − x_pred, y_meas − y_pred)`. The covariance
of `r_i` at the probe plane is

```
Σ_i(z_p) = Σ_probe,i + J_i Σ_line,i J_iᵀ ,

J_i = [ 1  z_p  0   0  ]
      [ 0   0   1  z_p ]
```

`W_i = Σ_i⁻¹` is the weight matrix. The objective is the χ²

```
χ²(θ, t_x, t_y, z_p) = Σ_i r_iᵀ W_i(z_p) r_i .
```

Two things make this objective well-behaved despite four unknowns and a
nonlinearity in θ:

- **`χ²` is linear in `(t_x, t_y, z_p)` at fixed θ.** Plugging the residual
  expressions in, the model at fixed `(c, s) = (cos θ, sin θ)` is a standard
  weighted linear least-squares with three unknowns and 2N equations,
  closed-form solution.
- **The remaining nonlinearity is in a single bounded scalar θ ∈ [−π, π]**,
  evaluable cheaply.

### 8.4 The optimiser

The recipe has four steps.

**Step 1 — coarse θ scan.** For each θ on a 1° grid over [−π, π], hold W_i
fixed at the probe-only weight (no z_p dependence yet) and solve the linear
problem for `(t_x, t_y, z_p)`. Record `χ²_min(θ)`.

**Step 2 — diagnostic plot of χ²(θ).** Inspect the full landscape before
trusting any number. Because the `(u, v)` channels are already decoded and
held fixed during the scan, χ²(θ) has a **single sharp global minimum at the
true mounting orientation** — the θ±90° hypotheses fit far worse, since the
square-probe symmetry also requires an `(u, v) → (v, L−u)` channel relabeling
the scan does not perform (this is the relabeling ambiguity of §8.5, *not* a
degeneracy of this fit — see the note there). A global minimum away from the
expected orientation, or competing wells of comparable depth, **signals a
wiring or axis problem and should halt the fit until investigated.** This step
is the most important consistency check in this stage; it is plotted per `z_p`
by `monrad-resolution` (`chi2_theta_z*.png`, §10).

**Step 3 — fine θ scan.** Pick the global minimum. Refine over ±2° at 0.01°
steps with the same linear solve.

**Step 4 — joint Levenberg-Marquardt polish.** Seed from the fine-scan
optimum and run LM on all four parameters simultaneously, updating
`W_i(z_p)` at each iteration. Typically converges in 3–5 iterations.

Outliers (e.g. random coincidences with an uncorrelated track and an
unrelated probe hit) are handled by a one-pass cut on Mahalanobis distance:
compute `d_i = √(r_iᵀ W_i r_i)`, drop coincidences with `d_i > 4`, refit. With
the low accidental rate assumed, one pass is enough.

### 8.5 The 4-fold rotation ambiguity

A square probe with identical X and Y strip layouts is physically ambiguous
under mountings rotated by 90°, 180°, or 270°: a 90° physical rotation maps the
X strips onto the Y strips, so it is equivalent to **relabeling the decoded
`(u, v)` channels** (`(u, v) → (v, L−u)`) together with `θ → θ+90°`. Crucially,
this is an ambiguity in *interpreting the hardware*, not a degeneracy the
optimizer exhibits: any single recorded dataset already fixes one labeling, so
the §8.4 χ²(θ) scan — which holds `(u, v)` fixed — has a unique minimum (the
θ±90° branches, lacking the channel relabel, fit far worse). The four
solutions are the *same fitted θ* reinterpreted under the four equivalent
channel labelings, not four competing minima. The pipeline either:

- accepts the ambiguity and reports all four equivalent interpretations, or
- breaks it externally — by knowing the nominal mounting orientation to
  ±45°, by using a marked corner on the probe, or by exploiting any X/Y
  asymmetry (e.g. unequal channel counts on the two axes).

This is documented in the report; it is not an algorithmic failure. (Because
the χ²(θ) fit is itself unambiguous, `monrad-resolution` scores fitted poses
directly against synthetic ground truth with no branch disambiguation.)

### 8.6 Expected precision

With per-coordinate strip resolution `σ_strip = 10 mm / √12 ≈ 2.9 mm` and
N coincidences, the `(t_x, t_y)` precision is roughly `σ_eff / √N`, where

```
σ_eff² = σ_probe² + (telescope angular σ × z_p)²
       = σ_strip² + (3 mrad × z_p)²
```

(the 3 mrad comes from `σ_strip × √(2/3) / 800 mm` for an 80 cm telescope
lever arm).

| z_p | σ_eff per coord | N for σ_t = 1 mm | N for σ_t = 0.3 mm |
|---|---|---|---|
| 0 cm | ≈ 4 mm | ~16 | ~180 |
| 30 cm | ≈ 4 mm | ~20 | ~200 |
| 5 m | ≈ 15 mm | ~225 | ~2500 |

`z_p` itself is well-determined at small distances (mm-level) but increasingly
degenerate with `(t_x, t_y)` at large distances; at 5 m, expect σ on `z_p` of
tens of cm. If a tape-measure value is available externally it should be
compared against the fit as a sanity check.

**Geometry-independent form.** The `3 mrad` above is `σ_strip·√(2/3)/L_tel` with
the telescope depth `L_tel = 800 mm`, so it scales as `1/L_tel`. Writing the
probe distance as the dimensionless `ρ = z_p/L_tel` removes that dependence:

```
σ_eff / σ_strip = √(1 + C_ρ · ρ²),   C_ρ = (σ_strip·√(2/3)/L_tel · L_tel/σ_strip)² = 2/3
```

is the same curve for any plane spacing, and `N_required = (σ_eff/target)²`
becomes a function of `ρ` alone (for a fixed strip pitch and target). A result
measured on the 800 mm testlab telescope therefore transfers to another
telescope by reading it at the same `z_p/L_tel`. The natural lateral coordinate
is likewise the **normalized polar angle** `η = α/α_max = (r/z_p)·(L_tel/active)`,
the probe's off-axis angle as a fraction of the telescope's footprint-limited
slope acceptance `α_max ≈ active/L_tel`; polar angle alone is *not* geometry-
independent — it must be normalized by `α_max`. `monrad-resolution` reports both
absolute mm and these `(ρ, η)` coordinates (`sigma_eff_vs_rho.png`).

**Azimuthal dependence.** For a probe offset by magnitude `r` from the axis, the
lab-frame resolution `(σ_x, σ_y)` is nearly independent of the offset *direction*
(azimuth `φ`): only `r` matters. The `σ_x ≠ σ_y` anisotropy is set by the fixed
probe mounting rotation `θ`, not by `φ`. The square telescope's 4-fold symmetry
means only one azimuth quadrant `φ ∈ [0, π/2]` need be swept (the probe's fixed
`θ` breaks the `x↔y` diagonal reflection, so it is a quadrant, not an octant);
the footprint's `√2×` longer diagonal reach extends the maximum usable offset
(acceptance), not the per-`N` resolution.

### 8.7 Output and diagnostics

The fitter returns a bundle, **not just four numbers**:

- the four fitted parameters and the 4 × 4 covariance from the inverse
  Hessian at the optimum;
- the χ²(θ) curve from §8.4 step 1;
- residual histograms in `x` and `y` at the probe plane (expected shape:
  triangular if errors are uniform, or roughly Gaussian if dominated by
  track extrapolation; mean should be zero);
- a stratified-half consistency test — split the dataset by event-time
  parity, fit each half, and report whether the parameters agree within
  their σ; disagreement indicates either a systematic (drift, miscounted
  coincidences, an overlooked tilt) or an underestimated covariance;
- the kept (`inliers`) and Mahalanobis-cut (`outliers`) coincidences, drawn
  as a 3D track plot (`run_pipeline.py --plot` → `pose_3d.html`) with the
  LM-polish-removed outlier tracks styled distinctly from the inliers, so the
  rejected tracks are visible alongside the ones that defined the fit.

These diagnostics are what tell you whether to trust the four numbers.

### 8.8 Accumulator design

Stage 5 is implemented as `PoseFitter`, which consumes the
`coincidence_stream()` generator and accumulates coincidences into a rolling
buffer. A refit is triggered every `refit_every` new coincidences once the
minimum `MIN_FIT` threshold is reached.

```python
tel_stream_a = reconstruct_stream(tel_gps, tel_pos, utc0, f0)
tel_stream_b = reconstruct_stream(tel_gps, tel_pos, utc0, f0)
prb_stream   = reconstruct_stream(prb_gps, prb_pos, prb_utc0, f0)

accum  = AlignmentAccumulator(flush_every=10_000)
fitter = PoseFitter(tel_z=Z_TEL, alignment=AlignmentCorrection.identity())

# Stage 4: all telescope events on stream A
for ev, ref in tel_stream_a:
    hit = decode_position(ref, tel_pos_paths, n_cols=3)
    if hit:
        corr = accum.add(hit)
        if corr:
            fitter.update_alignment(corr)

# Stage 5: coincidences on stream B + probe stream
for cluster in coincidence_stream([tel_stream_b, prb_stream],
                                   detector_ids=[TEL_ID, PRB_ID]):
    result = fitter.add(cluster)
    if result:
        log.info('Pose updated: %s', result)
```

`PoseFitter.update_alignment()` accepts a new `AlignmentCorrection` at any
time; it is applied to telescope hits at the next refit.


## 9. Module layout

```
src/monrad/                  # each stage is a domain package; the public API
                             # listed below is re-exported from its __init__.py
    decoders/
        header.py    # parse_header(), decode_ubx_tm2()
        gps.py       # GPSDecoder — reads *_GPS.bin
        position.py  # BinDecoder — reads *.bin, reconstructs hits
    timing/          # stage 1
        reconstruct.py   # reconstruct_stream(), load_header_params(),
                         # find_file_pairs()
    coincidence/     # stage 2
        search.py        # coincidence_stream()
    reconstruction/  # stage 3
        hit.py           # Hit, GOOD_QUALITIES, decode_position()
        candidates.py    # reconstruct_plane_candidates(), PlaneCandidate
    alignment/       # stage 4
        accumulator.py   # PlaneCorrection, AlignmentCorrection,
                         # AlignmentAccumulator, fit_telescope_alignment()
    pose/            # stage 5
        types.py         # Coincidence, DecodeReport, GATE_ORDER, PoseResult
        optimize.py      # fit_probe_pose() + line-fit / residual helpers
        fitter.py        # PoseFitter, _decode_cluster()
    synthetic/       # generate() — synthetic dataset for testing
        generate.py
    monitor/         # probe-position monitoring drivers (Steps 1-3)
```

Key types:

| Type | Module | Description |
|---|---|---|
| `Quality` | `timing` | GOOD / DEGRADED / UNTRUSTED |
| `TimedEvent` | `timing` | `(t_ns, evt_seq, quality)` |
| `PosRef` | `timing` | `(file_idx, row_offset, split_rows)` |
| `Hit` | `reconstruction` | `(x_mm, y_mm, sigma_x, sigma_y, quality, candidates_x, candidates_y)`; `quality` ∈ `golden`/`cluster`/`unresolved`/`invalid`; `candidates_*` carry per-axis hypotheses on `unresolved` hits, retained for diagnostics only |
| `PlaneCandidate` | `reconstruction` | one enumerated `(x_mm, y_mm, sigma_x, sigma_y, quality)` candidate for a plane |
| `PlaneCorrection` | `alignment` | `(delta_x, delta_y, rotation_z, delta_z, tilt_x, tilt_y)` |
| `AlignmentCorrection` | `alignment` | list of `PlaneCorrection` + `needs_correction` |
| `Coincidence` | `pose` | decoded coincidence ready for pose fit |
| `DecodeReport` | `pose` | per-cluster decode outcome (`accepted`, `reason`, `cand_counts`, `chi2`, `prb_quality`, `tel_quality`); `GATE_ORDER` is the rejection-funnel order |
| `PoseResult` | `pose` | full fit bundle (params, cov, diagnostics) |


## 10. Open items and assumptions to verify

The following items are reasonable defaults but should be confirmed against
real data on first inspection:

- **First-PPS handling.** *Checked against `data/0_testLab_20210723`.* The
  first record in `*_GPS.bin` is **not** a PPS — it is an event record. Both
  detectors open with several event records before the first PPS (telescope:
  first PPS after 7 events, at GEN 8; probe: after 28, at GEN 29), so every
  run begins with a block of pre-PPS_1 events. These are back-extrapolated and
  tagged `Quality.DEGRADED` (§4.3), so the "first record is a PPS" assumption
  is not required and not relied on. The full-run `Quality` tally corroborates
  this (telescope 35 / probe 46 `DEGRADED` events out of 1.46 M / 1.31 M — the
  pre-PPS_1 back-extrapolated head plus the post-last-PPS forward-extrapolated
  tail). The pipeline still anchors UTC₀ to the first PPS edge; that produces
  internally consistent timestamps and coincidences on this run, but the
  absolute UTC ↔ first-PPS correspondence has not been independently
  cross-checked against the GPS string.
- **Cross-file PPS continuity.** *Confirmed against `data/0_testLab_20210723`.*
  The PPS chain continues smoothly across `*_GPS.bin` boundaries with no
  synthetic gap: at the first file boundary the last PPS of file k and the
  first PPS of file k+1 are Δtick = 99 997 462 apart = 1.0000 s (residual
  2.5×10⁻⁵, within τ = 10⁻⁴). The ~25 ppm shortfall below the nominal 10⁸
  ticks/s is the local-oscillator offset that §4.2's `f_local` (not `f₀`) is
  designed to absorb. The full run confirms this end to end: **0 `UNTRUSTED`
  events** on either detector (1.46 M telescope / 1.31 M probe), i.e. every PPS
  interval across all 148 files — including every file boundary — passed the
  `res ≤ τ` acceptance.
- **Cross-file 16-row block continuity.** *Checked against
  `data/0_testLab_20210723` — assumption only partly holds.* Every `*.bin`
  row count (all 148 files, both detectors) is an exact multiple of 16, so a
  file is never truncated mid-block, and no boundary shares a GEN between the
  last row of file k and the first row of file k+1. **But** the `.bin` and
  `*_GPS.bin` streams are not always block-aligned per file: a file's `rows/16`
  occasionally differs from its `*_GPS.bin` event count — **4 of 148**
  telescope files and **16 of 148** probe files, by ±1 or ±2 blocks. The
  offsets occur in **compensating consecutive pairs** (a `+1` file immediately
  followed by a `−1` file; net 0 per detector), i.e. at a file rotation a whole
  16-row block lands in the `.bin` of the adjacent file relative to where its
  GPS event record was written. Because the misalignment is whole blocks (not a
  partial mid-block split), §4.4's `split_rows` stitch never fires here — it is
  a no-op when `rows % 16 == 0`. The pipeline instead detects the count
  mismatch (`reconstruct.py` logs "N GPS events but M position blocks") and
  continues; the affected handful of events (~0.001 %) are absorbed by the
  downstream quality/χ² cuts. The clean "split block with shared GEN, stitched
  via `split_rows`" case §4.4 describes is therefore **not** what this DAQ
  produces and remains unexercised.
- **GEN behaviour at acquisition start.** *Confirmed against
  `data/0_testLab_20210723`.* GEN starts at **1** (the first event record has
  GEN = 1) for both telescope and probe — not 0, and not an arbitrary value.
  The pipeline depends only on GEN's monotonicity and the §6.2 agreement
  check, both of which hold from the first event; the run completes with no
  GEN-mismatch halt.
- **Probe channel count.** *Resolved against `data/0_testLab_20210723`.* It is
  **not** in the header (the probe header carries only module and `[GPS]`
  sections, no channel count). It is determined by inspection of which bits
  ever fire: across all probe position files the highest bit set is ribbon
  bit 3 and fiber bit 9 on both axes, so the maximum channel is
  10·3 + 9 = 39 → **40 channels per axis** (a 40 × 40 cm probe, `n_cols = 1`).
  Per the "Probe active area inference" item below, this max-channel scan over
  the full run is exactly how §6.5's channel → coordinate mapping should size
  the probe.
- **Saturation interpretation.** The `_is_valid` filter treats any 10-bit
  half equal to 1023 as invalid (saturated). Whether a partially-saturated
  event is recoverable by trusting the unsaturated half is left as future
  refinement.
- **Clock frequency source.** Real headers do not carry an `f₀` field;
  `load_header_params` always uses `F0_DEFAULT = 100_000_000 Hz`.  Confirm
  this is correct for all detector models before any multi-detector run.
- **Probe active area inference.** Probe size cannot be assumed to be 30×30 cm².
  The physical active area should be inferred per-acquisition from the
  most-significant ribbon channel that ever fires.  Dark-current hits can
  illuminate ribbon channels beyond the geometrically active region; a
  per-axis maximum-channel scan over the full run should precede any
  channel→coordinate mapping.
- **Telescope plane z-coordinates.** `_Z_TEL = [0, 400, 800] mm` is hardcoded.
  Verify against hardware drawings before the first stage-4 run; a few-mm
  error biases the line-fit covariance and the probe z_p estimate.
- **Plane tilt detection.** *Implemented for the middle plane.* The telescope
  mechanics suppress rotation about z but permit a small tilt of a plane about
  x or y, breaking parallelism. Such a tilt manifests as a residual that
  correlates with the track direction cosine (§7.3). Stage 4 now fits a
  middle-plane `tilt_x`/`tilt_y` (joint `(b, b·coord)` regression, §7.3) and
  stage 5 applies it as a per-axis effective-z shift (§8.2). Outer-plane tilts
  remain degenerate with track slope and are left at 0; resolving them, like
  individual-plane offsets, needs external survey data or a 4-plane geometry.
  The 2 cm physical thickness of each plane (two perpendicular scintillator
  layers) also introduces a z-ambiguity that is not separately modelled.
- **Diagnostic plots.** A 3D pose plot is produced (`run_pipeline.py --plot`
  → `pose_3d.html`) showing the telescope planes, the fitted probe plane, and
  both the inlier and the LM-polish-removed (Mahalanobis-cut) outlier tracks,
  styled distinctly (§8.7).  The `monrad-resolution` study (monitoring Step 1)
  adds the **χ²(θ) consistency curve** (§8.4 step 2) and the **probe-plane x/y
  residual histograms** (§8.7), one of each per `z_p` (`chi2_theta_z*.png`,
  `residuals_z*.png`).  Still to implement before first real-data validation:
  stage-4 telescope-residual histograms and the alignment drift log (§7.6,
  a time-series concern belonging with monitoring Step 2).  These are the
  primary human-readable outputs for deciding whether to trust the fitted
  parameters.
- **Alignment curvature degeneracy.** `fit_telescope_alignment` uses the
  two-plane predictor (§7.3b).  For z = [0, 400, 800] mm the interpolation
  fractions are t₀ = −1, t₁ = 0.5, t₂ = 2, so the residuals satisfy
  r₀ = r₂ = x₀ − 2·x₁ + x₂ and r₁ = −r₀/2 for any dataset.  The predictor
  therefore measures only the second difference (curvature) of hit positions
  and cannot distinguish which individual plane is physically offset.  The
  corrections (Δ[0] = Δ[2], Δ[1] = −Δ[0]/2) should be interpreted as
  "curvature removal", not individual plane localisation.  Resolving
  individual plane offsets requires external survey data or a 4-plane geometry.
  For uneven plane spacing (e.g. z = [0, 630, 1350] mm) the identity breaks,
  so the `--z-tel` argument must always reflect the true hardware geometry.
- **Folded-fiber readout — telescope and probe (all datasets).**
  `scripts/diagnose_hits.py` has been run on three lab datasets
  (2021-07-23, 2022-02-04, 2023-04-18).  The key diagnostics are
  fold-symmetry score (ratio of mirror-pair firing rates; 1.00 = perfect
  fold), mean popcount after 16-row OR, and the all-bits-set rate (proxy
  for MAROC cross-talk).

  **Telescope fold-symmetry scores (excluding all-bits-set events):**

  | Run  | Pl | fib_X | rib_X | fib_Y | rib_Y | xtalk_fib_X% |
  |------|----|-------|-------|-------|-------|--------------|
  | 2021 | 0  | 0.90  | 0.93  | 0.91  | 0.95  | 2.7          |
  | 2021 | 1  | 0.90  | 0.77  | 0.89  | 0.90  | 3.4          |
  | 2021 | 2  | 0.87  | 0.86  | 0.91  | 0.89  | 3.1          |
  | 2022 | 0  | 0.88  | 0.93  | 0.73  | 0.89  | 13.4         |
  | 2022 | 1  | 0.91  | 0.80  | 0.89  | 0.90  | 2.4          |
  | 2022 | 2  | 0.87  | 0.82  | 0.91  | 0.88  | 7.0          |
  | 2023 | 0  | 0.87  | 0.78  | 0.85  | 0.88  | 3.9          |
  | 2023 | 1  | nan   | nan   | nan   | nan   | **89.5**     |
  | 2023 | 2  | 0.83  | 0.81  | 0.89  | 0.90  | 2.3          |

  (2023 column refers to `BuS_Tracker` — see note below on plane 1.)

  **Deductions:**

  1. *Folded readout is structural.*  Fold-symmetry ≈ 0.80–0.95 in the
     fiber half appears consistently across all three years and all planes
     that are functional.  The most likely explanation is that each physical
     scintillator strip is read out by two SiPMs/fibers routed to MAROC
     channels k and (9 − k) ("folded" or "mirror" wiring).  Mean popcounts
     of ~2 for telescope fiber halves (vs. ~1 for single-bit events)
     confirm that two bits fire per hit.

  2. *Ribbon fold in the telescope.*  Ribbon halves also show fold-symmetry
     scores of 0.77–0.95 and mean popcounts of ~1.8–2.3.  All 10 ribbon
     bits are active in telescope planes, implying the ribbon readout is
     also folded.  This means both fiber and ribbon contribute fold pairs
     for every hit; after fold-decoding, the effective channel count per
     axis is at most 5 × 5 = 25 (fiber positions 0–4 × ribbon positions
     0–4).  The physical coordinate mapping `x_mm = (ch + 0.5) × 10 mm`
     remains usable but implies an active area of ~250 mm per axis rather
     than 990 mm — the true strip-to-MAROC mapping requires hardware
     documentation or a test-pulse scan to determine the correct physical
     ordering.

  3. *Probe ribbon is NOT folded.*  All probe ribbon halves use only bits
     0–3 (4 active bits, bits 4–9 are always zero).  This matches a probe
     with 4 ribbon strips → ~40 mm active area in that dimension.  Fold
     symmetry is NaN for probe ribbon because no mirror pairs (k, 9−k) with
     k ≤ 3 exist above the 2 % activity threshold.  Probe fiber halves show
     moderate fold signatures (0.71–0.86), so the fiber half of the probe
     is also likely folded.

  4. *Current decoder impact.*  `BinDecoder._reconstruct_coord` treats two
     non-adjacent bit-clusters per half as irreconcilable and returns
     `unresolved`, which accounts for the high (~83 %) unresolved rate in
     telescope coincident hits.  The fold-symmetry tables above are raw
     measurements and stand on their own; their *causal* interpretation,
     however, is secondary.  The primary driver of the multi-candidate
     patterns is a **too-low acquisition threshold** that lets extra adjacent
     strips fire — §8.2 now treats the threshold as the primary ambiguity
     source and resolves the resulting candidates geometrically, via the
     combinatorial line-fit χ² search, rather than by recognising a specific
     (k, 9−k) fold pattern in the decoder.

  5. *Synthetic test coverage.*  `monrad.synthetic.generate()`'s `fold=True` /
     `fold_planes` path defaults to an idealised, perfectly periodic fold
     pattern (every mirror pair co-fires, no cross-talk) for backward
     compatibility with the existing fold-recovery tests.  Two additional
     parameters, `fold_symmetry` (probability the mirror partner bit also
     fires; default 1.0) and `fold_crosstalk_rate` (probability an extra,
     unrelated fiber bit also fires; default 0.0, fiber-only per the ~0 %
     ribbon cross-talk finding below), let a test inject the realistic
     0.71–0.95 fold-symmetry / 1.7–2.6 % fiber cross-talk statistics
     measured above instead of the idealised pattern.
     `tests/test_stage5.py::TestFoldedPoseRecovery2PlaneRealisticNoise`
     exercises the combinatorial track finder (§8.2) against this messier,
     non-idealised data on the two-ambiguous-plane case.

- **MAROC cross-talk — per-run severity and 2023 plane-1 failure.**
  The all-bits-set popcount (= 10) rate is the primary cross-talk proxy.
  All-bits-set events are already rejected by `BinDecoder._is_valid` and
  excluded from fold-symmetry calculations.

  *2021 telescope:*  All planes show uniform cross-talk of 2.7–3.5 % on
  both fiber and ribbon halves.  This is above the ~1 % target but modest
  enough that most planes remain functional; MAROC thresholds were
  apparently near the acceptable limit in this run.

  *2022 telescope:*  Plane 0 shows 12–14 % all-bits-set rate (previously
  documented as ~12.5 %).  Plane 2 shows 7 %.  Plane 1 is the healthiest
  at 1.8–2.5 %.  The Plane 0 fiber_Y fold-symmetry score drops to 0.73
  because the elevated cross-talk inflates individual bit counts unevenly
  even after cross-talk events are excluded.

  *2023 BuS_Tracker:*  **Plane 1 is catastrophically saturated: 89.5 %
  all-bits-set rate across all four axes.**  The remaining ~10 % events
  have a mix of single-bit and very-low-count patterns with no coherent
  fold structure (fold-symmetry = NaN, no active bits above 2 % threshold).
  Plane 1 of the 2023 BuS_Tracker should be treated as non-functional; the
  2023 run is effectively a 2-plane telescope (planes 0 and 2 only).  The
  MAROC chip for plane 1 requires threshold/gain adjustment before the next
  run.  Planes 0 and 2 of the 2023 BuS_Tracker have acceptable cross-talk
  (2–4 %) and clear fold-symmetry signatures (0.78–0.90).

  *2023 BuS_Probe J11_40x40:*  Fiber cross-talk is 1.7–2.6 %; ribbon
  shows no cross-talk (0 %).  The fiber_X fold-symmetry is notably lower
  (0.71) with strongly asymmetric per-bit rates (bits 4–7 fire at ~31–35 %,
  bits 0–3 and 8–9 at 15–28 %), suggesting either non-uniform illumination
  or an imperfect fold mapping on this probe board.


## 11. Synthetic end-to-end test

Before any of the stages is exercised against real data, the following
test guards the pipeline as a regression suite.

1. **Generate a synthetic dataset.** Pick ground-truth values
   `(t_x, t_y, θ, z_p)` for one probe, plus optional small per-plane
   misalignments for the telescope (a translation `Δx_k`, a rotation about
   z `α_k`, etc.) so that §7 has something to detect. Generate N = 1000
   muon tracks with random directions sampled from the cosmic-ray angular
   distribution, restricted to those crossing all three telescope planes.
   For each track, compute the three telescope plane intersections (with
   the misalignments applied), and where it also crosses the probe at
   `z_p`, the probe-frame `(u, v)`. Quantise each coordinate to the strip
   grid (1 cm pitch, 99 channels for the telescope, plausible count for the
   probe). Encode each as a `*_GPS.bin` event record (with synthetic PPS
   records at 1 Hz) and a `*.bin` 16-row block (golden hits — single fiber
   bit + single ribbon bit).
2. **Run the pipeline end-to-end.** Stage 1 → Stage 2 → Stage 3 → Stage 4
   → Stage 5. Stage 4 should recover the injected per-plane misalignments
   within their statistical uncertainties, and once those corrections are
   folded in, Stage 5 should recover `(t_x, t_y, θ, z_p)` within 3σ of its
   fitted uncertainties.
3. **Adversarial cases** — dropped PPS, GEN wraps, 16-row block split across
   files, occasional cluster hits and invalid hits, a few accidental
   coincidences, a telescope with a deliberately uncorrected plane rotation,
   pre-PPS_1 events, stream exhaustion mid-buffer, back-to-back untrusted
   intervals — verify that each is detected, flagged, or absorbed correctly.

**Implementation status.** All three requirements above are implemented:

- `monrad.synthetic.generate()` produces the synthetic dataset described in
  step 1.
- `tests/test_pipeline_stream.py` runs the complete two-pass streaming
  pipeline and asserts that recovered parameters lie within 3σ of ground
  truth and that peak heap allocation stays below 512 MB.
- Adversarial cases are covered by `tests/test_stage1.py` (pre-PPS events,
  stream exhaustion, untrusted intervals) and the per-stage test modules
  (`tests/test_stage2.py` through `tests/test_stage5.py`).
