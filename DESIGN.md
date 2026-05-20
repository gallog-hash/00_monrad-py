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
`decode_header.py`, `decode_gps.py`, and `decode_bin.py`. This document treats
their bit-level behaviour as authoritative; if anything stated here disagrees
with the scripts, the scripts win.

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
beyond 1 cm). Probes have a 30 cm × 30 cm active area; the channel count per
probe depends on how many bars are used and is not currently known to the
pipeline a priori.

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
…) and `key=value` entries. The fields the pipeline needs are:

- The **clock counter frequency** `f₀`, a single integer (Hz), the same for
  all detectors.
- The **GPS string** in the `[GPS]` section, written as latin-1 with `\XX` hex
  escapes for non-printable bytes. This is a UBX-TIM-TM2 binary frame from
  the receiver, decoded by `decode_ubx_tm2()` in `decode_header.py`. There is
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
corresponding `*_GPS.bin`. The pipeline asserts this invariant on each
file pair as a sanity check (see §4.4 for the file-boundary edge case).

### 2.4 Position encoding — fiber and ribbon

Each axis's 20 bits encode one or more 1 cm strips firing using a folded
fiber × ribbon scheme: the physical channel index is

```
ch = N · ribbon_bit + fiber_bit ,   N = 10
```

where `ribbon_bit` is the LSB-indexed position of the bit set in the 10-bit
ribbon mask and `fiber_bit` likewise for the fiber mask. With 10 fiber × 10
ribbon = 100 channel codes available, this comfortably covers the 99-channel
telescope and any practical probe. **Channel 0 is at one physical edge** of
the active area.

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
GPS string) and the nominal counter frequency `f₀`. PPS records inside
`*_GPS.bin` then provide a stream of further anchors at 1 Hz, latching the
counter value `C_k` at each successive PPS edge.

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

A deprecated `reconstruct()` batch wrapper is retained for backward
compatibility. It materialises the full stream into lists and returns
`(events, pos_map)` where `pos_map[evt_seq]` equals the corresponding
`PosRef`. Remove it once all callers have been migrated to
`reconstruct_stream()`.


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
) -> list[Hit | None]:
```

`pos_ref` is received directly from the stream — no `pos_map` lookup is
needed. Returns one `Hit | None` per plane (`n_cols` elements).

### 6.2 Random access into `*.bin`

For the `(file_idx, row_offset)` in `pos_ref`, read 16 × `n_cols` u64s
starting at `row_offset`. This is a single O(1) seek per event, costing at
most 16 × 3 × 8 = 384 bytes per telescope event and 128 bytes per probe event.

For each of the `n_cols` planes, compute the bitwise OR of the 16 samples'
X and Y fields. Verify that all 16 GEN values within the block agree, and
that the GEN matches `evt_seq mod 2048` from `*_GPS.bin`. A mismatch on
either is a structural error — the file pair is corrupted or the join is
wrong; halt and report.

### 6.3 Validity prefilter

A column is **invalid** if any of the four 10-bit halves of the OR equals
1023 (all bits set, indicating channel saturation), or if either ribbon half
is zero (no ribbon channel fired, so no coordinate can be recovered). This
matches the existing logic in `decode_bin.py::_is_valid`.

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

### 6.5 Channel → physical coordinate

```
coord_mm = (ch + 0.5) × strip_pitch_mm   #   strip_pitch_mm = 10
```

with channel 0 at one physical edge of the active area. The same convention
is used for telescope and probes. Any per-detector edge offset (e.g. a frame
that prevents the leftmost strip from being at exactly x = 0) is absorbed
into the alignment fits in §7 and §8.

### 6.6 Future refinement

The 16 per-sample bit patterns carry information that the OR discards: the
first sample in which a strip fires gives a sub-event-window time stamp
(useful for time-walk corrections), and the number of active samples per
strip is a time-over-threshold quality weight. Both are noted for future
work and are not used in the current design.


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
  manifests as a residual that grows with track angle. Both diagnostics need
  the full angular and spatial extent of the telescope, not the narrow cone
  selected by a coincidence with a probe.

For both reasons this stage runs on **all telescope tracks** via a dedicated
`reconstruct_stream()`, independent of the coincidence pipeline.

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
  about x).

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

`AlignmentCorrection` carries the per-plane `(Δx, Δy, rotation_z)` values
and a `needs_correction` boolean that downstream consumers (stage 5) read to
decide whether to apply it.

### 7.6 Continuous monitoring

Each flush produces a timestamped `AlignmentCorrection`. Persisting these
to a log file indexed by the UTC timestamp of the first hit in the batch
directly implements continuous drift monitoring: slow changes indicate
mechanical settling or thermal expansion; sudden jumps indicate a discrete
physical event (apparatus bumped, DAQ restarted).


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
mutually parallel and X-Y aligned, and that the probe plane is parallel to
the telescope planes. Place the telescope frame so the planes are at
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

A **track quality cut** (e.g. χ² of the line fit < 4, equivalent to ≤ 1 strip
of residual on each plane) is applied here to remove ghost tracks before
they corrupt the alignment fit. Loose cuts are preferred over tight ones at
this stage.

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
trusting any number. For a square probe, four equally deep minima at
multiples of 90° are expected; **unequal minima signal a wiring or axis
problem and should halt the fit until investigated.** This step is the most
important consistency check in this stage.

**Step 3 — fine θ scan.** Pick the global minimum (or the one nearest a
known nominal mounting orientation, when available, to break the 4-fold
ambiguity). Refine over ±2° at 0.01° steps with the same linear solve.

**Step 4 — joint Levenberg-Marquardt polish.** Seed from the fine-scan
optimum and run LM on all four parameters simultaneously, updating
`W_i(z_p)` at each iteration. Typically converges in 3–5 iterations.

Outliers (e.g. random coincidences with an uncorrelated track and an
unrelated probe hit) are handled by a one-pass cut on Mahalanobis distance:
compute `d_i = √(r_iᵀ W_i r_i)`, drop coincidences with `d_i > 4`, refit. With
the low accidental rate assumed, one pass is enough.

### 8.5 The 4-fold rotation ambiguity

A square probe with identical X and Y strip layouts is invariant under
rotations of 90°, 180°, and 270° from the data alone — the four θ minima
are mathematically equivalent fits. The pipeline either:

- accepts the ambiguity and reports all four solutions, or
- breaks it externally — by knowing the nominal mounting orientation to
  ±45°, by using a marked corner on the probe, or by exploiting any X/Y
  asymmetry (e.g. unequal channel counts on the two axes).

This is documented in the report; it is not an algorithmic failure.

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
  coincidences, an overlooked tilt) or an underestimated covariance.

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
src/monrad/
    decoders/
        header.py    # parse_header(), decode_ubx_tm2()
        gps.py       # GPSDecoder — reads *_GPS.bin
        position.py  # BinDecoder — reads *.bin, reconstructs hits
    synth.py         # generate() — synthetic dataset for testing
    stage1.py        # reconstruct_stream(), load_header_params(),
                     # find_file_pairs(), reconstruct() [deprecated]
    stage2.py        # coincidence_stream()
    stage3.py        # Hit, decode_position()
    stage4.py        # PlaneCorrection, AlignmentCorrection,
                     # AlignmentAccumulator, fit_telescope_alignment()
    stage5.py        # Coincidence, PoseResult,
                     # PoseFitter, fit_probe_pose()
```

Key types:

| Type | Module | Description |
|---|---|---|
| `Quality` | `stage1` | GOOD / DEGRADED / UNTRUSTED |
| `TimedEvent` | `stage1` | `(t_ns, evt_seq, quality)` |
| `PosRef` | `stage1` | `(file_idx, row_offset, split_rows)` |
| `Hit` | `stage3` | `(x_mm, y_mm, sigma_x, sigma_y, quality)` |
| `PlaneCorrection` | `stage4` | `(delta_x, delta_y, rotation_z)` |
| `AlignmentCorrection` | `stage4` | list of `PlaneCorrection` + `needs_correction` |
| `Coincidence` | `stage5` | decoded coincidence ready for pose fit |
| `PoseResult` | `stage5` | full fit bundle (params, cov, diagnostics) |


## 10. Open items and assumptions to verify

The following items are reasonable defaults but should be confirmed against
real data on first inspection:

- **First-PPS handling.** Whether the very first record in `*_GPS.bin` is
  always a PPS, and whether the header's UTC₀ corresponds to acquisition
  start, to the first PPS, or to something else. The pipeline currently
  assumes UTC₀ corresponds to a PPS edge near the start of the run, and
  that the first PPS record's tick value matches it.
- **Cross-file PPS continuity.** That the PPS chain continues smoothly
  across `*_GPS.bin` boundaries with no synthetic gap inserted by the DAQ.
- **Cross-file 16-row block continuity.** That a 16-row block of position
  data may occasionally be split between two `*.bin` files; the pipeline
  detects and stitches such cases (§4.4), but the DAQ behaviour on file
  rotation should be confirmed.
- **GEN behaviour at acquisition start.** Whether GEN starts at 0 or at an
  arbitrary value at run start. The pipeline does not depend on GEN's
  absolute starting value (only its monotonicity), but it does assume the
  GEN agreement check in §6.2 is meaningful from the first event onward.
- **Probe channel count.** Whether this is recorded in the header or
  determined by inspection of which bits ever fire. §6.5 needs this for
  the channel → coordinate mapping.
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
- **Plane tilt detection.** The 2 cm physical thickness of each
  position-sensitive plane (two perpendicular scintillator layers) introduces
  a z-ambiguity.  A tilt of a plane about x or y would manifest as a residual
  that correlates with the track direction cosine (§7.3).  Detecting and
  correcting tilts requires fitting one extra parameter per plane per axis and
  is deferred until a tilt signal is observed above noise in real data.
- **Diagnostic plots.** No visualisation is currently produced.  Before first
  real-data validation, implement residual histograms (stage 4), the χ²(θ)
  curve (stage 5), and the alignment drift log.  These are the primary
  human-readable outputs for deciding whether to trust the fitted parameters.
- **Alignment curvature degeneracy.** `fit_telescope_alignment` uses the
  two-plane predictor (§6.3b).  For z = [0, 400, 800] mm the interpolation
  fractions are t₀ = −1, t₁ = 0.5, t₂ = 2, so the residuals satisfy
  r₀ = r₂ = x₀ − 2·x₁ + x₂ and r₁ = −r₀/2 for any dataset.  The predictor
  therefore measures only the second difference (curvature) of hit positions
  and cannot distinguish which individual plane is physically offset.  The
  corrections (Δ[0] = Δ[2], Δ[1] = −Δ[0]/2) should be interpreted as
  "curvature removal", not individual plane localisation.  Resolving
  individual plane offsets requires external survey data or a 4-plane geometry.


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

- `monrad.synth.generate()` produces the synthetic dataset described in
  step 1.
- `tests/test_pipeline_stream.py` runs the complete two-pass streaming
  pipeline and asserts that recovered parameters lie within 3σ of ground
  truth and that peak heap allocation stays below 512 MB.
- Adversarial cases are covered by `tests/test_stage1.py` (pre-PPS events,
  stream exhaustion, untrusted intervals) and the per-stage test modules
  (`tests/test_stage2.py` through `tests/test_stage5.py`).
