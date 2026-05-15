# Streaming pipeline redesign

This document supersedes the batch-processing assumptions in `DESIGN.md` for
all five pipeline stages. The algorithms described there — PPS-anchored
timestamping (§3), sliding-window coincidence search (§4), fiber × ribbon
hit reconstruction (§5), telescope internal alignment (§6), and probe pose
fit (§7) — are **unchanged**. What changes is the data-flow model: instead
of each stage accumulating a complete list before the next stage starts,
stages 1 and 2 become Python generators that drive stages 3–5 in a single
forward pass over the data.

**Motivation.** At a telescope rate of 10 Hz, one week of continuous
acquisition produces approximately 6 000 000 events per detector. The
current `reconstruct()` accumulates all events into Python lists before
returning, costing roughly 3.4 GB of RAM per detector in stage 1 alone.
Running telescope + two probes concurrently saturates a typical analysis
machine. The streaming redesign bounds peak RAM to a few hundred megabytes
regardless of run length.


## 1. Core data-flow principle

Every stage boundary becomes an iterator boundary.

```
stage 1 (per detector)          stage 2               stages 4 / 5
─────────────────────           ──────────            ──────────────
reconstruct_stream()   ──────►  coincidence_  ──────► AlignAccum /
                                stream()               PoseFitter
                       n+1 streams merged via min-heap
```

No stage materialises more than a bounded window of events at once.  The
maximum in-memory working set at any point in the pipeline is:

- Stage 1: one PPS interval (~1 s) of buffered events per detector.
- Stage 2: the coincidence window (200 ns) plus inter-detector PPS latency
  (≤ a few seconds).
- Stages 4 / 5: one accumulator buffer (configurable, default 10 000
  events for stage 4, 500 coincidences for stage 5).


## 2. Stage 1 — streaming time reconstruction

### 2.1 Replacement for `reconstruct()`

Remove `reconstruct()` and replace it with `reconstruct_stream()`.  The
batch function is no longer part of the public API.  The `pos_map` dict
is also removed (see §2.3).

```python
def reconstruct_stream(
    gps_paths: list[Path],
    pos_paths: list[Path],
    utc0: datetime,
    f0: int = F0_DEFAULT,
    tau: float = PPS_TAU,
) -> Iterator[tuple[TimedEvent, PosRef]]:
    """
    Streaming Stage 1.  Yields (TimedEvent, PosRef) pairs in time order,
    emitting each PPS interval's worth of events as soon as the closing
    PPS record is observed.

    Memory bound: at most ~1 s of events are held in `_pending` at any
    time.

    Yields
    ------
    TimedEvent  (t_ns, evt_seq, quality) — same semantics as before
    PosRef      (file_idx, row_offset, split_rows) — carried inline;
                the caller passes it directly to stage 3 decode calls
    """
```

### 2.2 Internal structure of `reconstruct_stream()`

The function maintains a `_pending` list that accumulates event ticks
between PPS records.  On each new PPS record it timestamps every pending
event, yields the results, and clears `_pending`.  Monotonically assign
`evt_seq` as events are yielded (same semantics as before).

```
for each (gps_path, pos_path) in zip(gps_paths, pos_paths):

    validate pos-file meta (row count, event count match)  ← same as before
    check GEN continuity at boundary                       ← same as before

    for each record in gps_path (in acquisition order):

        if record is PPS:
            if _pending and prev_pps is not None:
                build _Interval(prev_pps, this_pps, ...)  ← _build_next_interval
                for each (tick, fi, li) in _pending:
                    t_ns, q = _linear(utc0_ns, iv, tick), quality_of(iv)
                    yield TimedEvent(t_ns, evt_seq, q), PosRef(fi, li*16)
                    evt_seq += 1
                _pending.clear()
            prev_pps = this_pps

        else (event record):
            _pending.append((tick, file_idx, local_event_idx))

# After all files: flush remaining _pending using forward-extrapolation
# (DEGRADED quality) exactly as the current _timestamp() DEGRADED branch.
```

The helper `_build_next_interval(prev_pps_tick, this_pps_tick, prev_n,
f0, tau)` encapsulates the residual check and returns an `_Interval`
(trusted or untrusted) — extracted from the current `_build_pps_chain()`
logic, operating on one pair at a time instead of the whole list.

The first PPS interval cannot be closed until PPS_1 is observed.  Events
that arrive before PPS_1 are buffered in `_pending` and back-extrapolated
using the rate measured by the PPS_1→PPS_2 interval, exactly as the
current DEGRADED branch does.  This requires buffering the PPS_1→PPS_2
interval before back-filling; see §2.4.

### 2.3 Elimination of `pos_map`

The `pos_map` dict (`{evt_seq: PosRef}`) is no longer needed.  Every
`PosRef` is yielded inline with its `TimedEvent` and carried through the
pipeline by the caller.  Stage 3 receives `PosRef` directly (§4).

Callers that need to retain all `PosRef` values for random-access (e.g.
a post-hoc debug tool) can build their own array:

```python
# evt_seq is 0, 1, 2, ... so a plain list works; convert once at the end.
pos_index: list[PosRef] = []
for ev, ref in reconstruct_stream(...):
    pos_index.append(ref)
# pos_index[evt_seq] == PosRef for that event — identical semantics to the
# old pos_map dict, but without the per-entry Python dict overhead.
```

For production pipelines this list is never materialised.  `PosRef`
values are consumed immediately by stage 3 decode calls and then
discarded.

### 2.4 Back-extrapolation of pre-PPS_1 events

The current batch design resolves back-extrapolation in pass 3 after the
full PPS chain is known.  In the streaming design:

1. Buffer all events and PPS records until PPS_2 is observed.
2. At PPS_2: build `_Interval(PPS_1, PPS_2)` as `back_iv`.
3. Back-extrapolate buffered pre-PPS_1 events using `back_iv`, then
   timestamp PPS_1→PPS_2 events normally.
4. From PPS_2 onward: standard one-interval-at-a-time flow.

This introduces a one-time startup latency of at most 2 s.  Tag
pre-PPS_1 events with `Quality.DEGRADED` as before.

### 2.5 Split-block detection

The split-block patch (currently applied retroactively to a previously
yielded `PosRef`) must be detected eagerly instead.  When opening file
`k+1`, check GEN continuity before yielding any events from file `k`
that might belong to a split block.  Concretely: if the last `_pending`
event of file `k` has the same GEN as the first row of file `k+1`, hold
that event in `_pending` and construct a `PosRef` with the correct
`split_rows` value before yielding.


## 3. Stage 2 — streaming coincidence search

### 3.1 Replacement for the batch merge

Stage 2 becomes a generator that consumes n+1 `reconstruct_stream()`
iterators.

```python
def coincidence_stream(
    streams: list[Iterator[tuple[TimedEvent, PosRef]]],
    detector_ids: list[int],
    window_ns: int = 200,
) -> Iterator[list[tuple[int, TimedEvent, PosRef]]]:
    """
    k-way min-heap merge over n+1 stage-1 generators.

    Yields clusters of (detector_id, TimedEvent, PosRef) tuples.
    A cluster spans ≥ 2 detectors with all t_ns within `window_ns`.

    PosRef is carried through transparently so that stage 3 callers
    downstream never need to look anything up.
    """
```

### 3.2 Internal structure

The heap stores `(t_ns, detector_id, TimedEvent, PosRef)` tuples.
Everything else follows `DESIGN.md` §4.1 exactly: maintain a sliding
deque, evict entries older than `t_now − window_ns`, emit a cluster when
it falls off the window.  No algorithm change.

```python
heap: list = []
for det_id, stream in zip(detector_ids, streams):
    item = next(stream, None)
    if item is not None:
        ev, ref = item
        heappush(heap, (ev.t_ns, det_id, ev, ref))

deque: list[tuple[int, TimedEvent, PosRef]] = []

while heap:
    t_now, det_id, ev, ref = heappop(heap)
    deque = [(d, e, r) for d, e, r in deque
             if e.t_ns >= t_now - window_ns]
    deque.append((det_id, ev, ref))
    if len({d for d, _, _ in deque}) >= 2:
        yield deque.copy()
    nxt = next(streams[det_id], None)
    if nxt is not None:
        nev, nref = nxt
        heappush(heap, (nev.t_ns, det_id, nev, nref))
```

### 3.3 Inter-detector PPS latency

Each `reconstruct_stream()` buffers up to ~1 s of events waiting for
the next PPS.  The k-way heap stalls on the slowest-advancing stream.
No special handling is required: the heap naturally absorbs the latency.
Document the assumption that inter-detector PPS latency does not exceed
a configurable `max_lag_s` (default 5 s); if the heap stalls beyond
this, log a warning and forward-extrapolate the lagging stream.


## 4. Stage 3 — position decoding (interface update only)

**No algorithm changes.**  The fiber × ribbon decoding, validity
prefilter, hit reconstruction, and coordinate mapping in `DESIGN.md` §5
are unchanged.

The only interface change: `decode_position()` (the procedure described
in §5) now receives `PosRef` directly instead of looking it up in
`pos_map`.

```python
def decode_position(
    pos_ref: PosRef,
    pos_paths: list[Path],
    n_cols: int,
) -> Hit | None:
    """
    Decode one event's position from its PosRef.
    Replaces the pos_map lookup that previously prefixed this call.
    Algorithm: DESIGN.md §5.1–§5.4, unchanged.
    """
```

Callers update from:
```python
ref = pos_map[evt_seq]          # old
hit = decode_position(ref, ...) # new — ref comes from the stream directly
```


## 5. Stage 4 — online telescope alignment

### 5.1 Accumulator design

Stage 4 consumes the telescope's `reconstruct_stream()` directly (not
the coincidence stream — it uses all telescope events, as specified in
`DESIGN.md` §6.2).

```python
class AlignmentAccumulator:
    """
    Collects telescope hits into a ring buffer; fits and emits an
    AlignmentCorrection every `flush_every` hits.

    Implements DESIGN.md §6.3 (three-plane fit + per-plane residuals).
    """

    def __init__(self, flush_every: int = 10_000):
        self.flush_every = flush_every
        self._hits: list[Hit3D] = []
        self.current_correction: AlignmentCorrection = AlignmentCorrection.identity()

    def add(self, hit: Hit3D) -> AlignmentCorrection | None:
        """
        Add one decoded three-plane hit.  Returns a new AlignmentCorrection
        when the buffer is full and a fit has been performed; otherwise None.
        """
        self._hits.append(hit)
        if len(self._hits) >= self.flush_every:
            return self._fit_and_flush()
        return None

    def _fit_and_flush(self) -> AlignmentCorrection:
        correction = fit_telescope_alignment(self._hits)  # DESIGN.md §6.3
        self.current_correction = correction
        self._hits.clear()
        return correction
```

The caller loop:

```python
accum = AlignmentAccumulator(flush_every=10_000)
for ev, ref in reconstruct_stream(tel_gps, tel_pos, utc0, f0):
    hit = decode_position(ref, tel_pos_paths, n_cols=3)
    if hit is None or hit.quality not in (Quality.GOLDEN, Quality.CLUSTER):
        continue
    correction = accum.add(hit)
    if correction is not None:
        log.info('Alignment updated: %s', correction)
```

### 5.2 Continuous monitoring

Each flush produces a timestamped `AlignmentCorrection`.  Persist these
to a small log file (`alignment_timeseries.csv`) indexed by the UTC
timestamp of the first hit in the batch.  This directly implements
`DESIGN.md` §6.5 (hourly drift monitoring) with no additional work.

### 5.3 Decision threshold

The decision logic of `DESIGN.md` §6.4 (offsets < ~1 mm, rotations <
~1 mrad → proceed; otherwise fold corrections in) applies to the
correction returned by each `_fit_and_flush()` call.  The
`AlignmentCorrection` object carries both the fitted parameters and a
boolean `needs_correction` flag that downstream consumers (stage 5) read
to decide whether to apply it.


## 6. Stage 5 — streaming probe pose fit

### 6.1 Accumulator design

Stage 5 consumes the `coincidence_stream()` generator and accumulates
coincidences into a rolling buffer.

```python
class PoseFitter:
    """
    Accumulates telescope-probe coincidences and refits the probe pose
    every `refit_every` new coincidences using DESIGN.md §7.

    Carries the most recent AlignmentCorrection from stage 4 and applies
    it to telescope hits before each fit (DESIGN.md §7 preamble).
    """

    MIN_FIT = 50     # minimum coincidences before first fit
    REFIT_EVERY = 500

    def __init__(
        self,
        tel_z: np.ndarray,
        alignment: AlignmentCorrection,
        refit_every: int = REFIT_EVERY,
    ):
        self.tel_z = tel_z
        self.alignment = alignment
        self.refit_every = refit_every
        self._coincidences: list[Coincidence] = []
        self._since_last_fit: int = 0
        self.result: PoseResult | None = None

    def update_alignment(self, correction: AlignmentCorrection) -> None:
        """Called whenever stage 4 emits a new AlignmentCorrection."""
        self.alignment = correction

    def add(self, cluster: list[tuple[int, TimedEvent, PosRef]]) -> PoseResult | None:
        """
        Decode positions for all detectors in the cluster, run track
        quality cut (DESIGN.md §7.2), and accumulate.  Returns a new
        PoseResult when a refit is triggered; otherwise None.
        """
        coincidence = self._decode_cluster(cluster)
        if coincidence is None:
            return None
        self._coincidences.append(coincidence)
        self._since_last_fit += 1
        if (len(self._coincidences) >= self.MIN_FIT
                and self._since_last_fit >= self.refit_every):
            return self._refit()
        return None

    def _refit(self) -> PoseResult:
        result = fit_probe_pose(
            self._coincidences, self.tel_z, self.alignment
        )  # DESIGN.md §7.3–§7.4
        self._since_last_fit = 0
        self.result = result
        return result
```

### 6.2 Caller loop

```python
tel_stream  = reconstruct_stream(tel_gps,  tel_pos,  tel_utc0,  f0)
prb_stream  = reconstruct_stream(prb_gps,  prb_pos,  prb_utc0,  f0)
align_accum = AlignmentAccumulator(flush_every=10_000)
pose_fitter = PoseFitter(tel_z=Z_TEL, alignment=AlignmentCorrection.identity())

# Stage 4 runs on the telescope stream; stage 5 runs on the coincidence stream.
# Interleave by teeing the telescope stream.
tel_stream_a, tel_stream_b = itertools.tee(tel_stream)

for ev, ref in tel_stream_a:           # stage 4 branch
    hit = decode_position(ref, tel_pos_paths, n_cols=3)
    if hit:
        correction = align_accum.add(hit)
        if correction:
            pose_fitter.update_alignment(correction)

for cluster in coincidence_stream(     # stage 5 branch
    [tel_stream_b, prb_stream], detector_ids=[0, 1]
):
    result = pose_fitter.add(cluster)
    if result:
        log.info('Pose updated: %s', result)
```

Note: `itertools.tee` buffers the telescope stream in memory if the two
consumers advance at different rates.  In practice stage 4 consumes the
full telescope stream while stage 5 consumes only the coincident subset,
so stage 5's branch falls behind.  The recommended implementation runs
stages 4 and 5 in two separate threads sharing a thread-safe queue, or
alternatively fuses them into a single loop that calls both accumulators
for telescope events and only `pose_fitter.add()` for coincidences.  The
fused single-loop approach is simpler and avoids buffering:

```python
# Fused loop (preferred)
for cluster in coincidence_stream([tel_stream, prb_stream], ...):
    tel_entries = [(ev, ref) for det, ev, ref in cluster if det == TEL_ID]
    for ev, ref in tel_entries:
        hit = decode_position(ref, tel_pos_paths, n_cols=3)
        if hit:
            correction = align_accum.add(hit)
            if correction:
                pose_fitter.update_alignment(correction)
    result = pose_fitter.add(cluster)
    if result:
        log.info('Pose updated: %s', result)
```

This is only correct if the telescope stream passed to
`coincidence_stream()` is the same stream used for stage 4.  Telescope
events with no probe coincidence never reach the fused loop and are
**not** seen by stage 4, so stage 4's statistical sample is limited to
coincident telescope events.  For telescope rates of tens of Hz and
probe coincidence fractions of a few percent, this reduces stage 4's
effective event rate by ~97%, which may be insufficient for the residual
diagnostics of `DESIGN.md` §6.3.

**Resolution**: run stage 4 on a dedicated `reconstruct_stream()` for
the telescope only, and run stage 5 on a separate `coincidence_stream()`
that gets its own `reconstruct_stream()` for both detectors.  This means
the telescope GPS and position files are iterated twice, but since
`_pos_file_meta()` reads only 24 bytes per file and the GPS files are
small, the I/O cost is negligible.  This is the recommended approach and
avoids any threading complexity.


## 7. Module layout changes

The following new symbols must be added to `src/monrad/stage1.py`:

| Symbol | Replaces | Notes |
|---|---|---|
| `reconstruct_stream()` | `reconstruct()` | generator, yields `(TimedEvent, PosRef)` |
| `_build_next_interval()` | part of `_build_pps_chain()` | single-pair interval builder |

Remove from `src/monrad/stage1.py`:

| Symbol | Reason |
|---|---|
| `reconstruct()` | replaced by `reconstruct_stream()` |
| `_build_pps_chain()` | logic absorbed into `_build_next_interval()` |

New modules:

| Module | Contents |
|---|---|
| `src/monrad/stage2.py` | `coincidence_stream()` |
| `src/monrad/stage4.py` | `AlignmentAccumulator`, `AlignmentCorrection`, `fit_telescope_alignment()` |
| `src/monrad/stage5.py` | `PoseFitter`, `PoseResult`, `fit_probe_pose()` |

`src/monrad/stage3.py` wraps the position-decoding procedure currently
in `src/monrad/decoders/position.py`, adding the new `decode_position()`
signature.


## 8. Testing strategy

`DESIGN.md` §9 describes the synthetic end-to-end test.  All existing
test cases remain valid; only the fixture that calls `reconstruct()` must
be updated.

### 8.1 Stage 1 fixture update

```python
# tests/test_stage1.py — replace the `tel` and `prb` fixtures

@pytest.fixture(scope='module')
def tel(synth):
    result, out = synth
    tel_dir = out / 'telescope'
    header  = next(tel_dir.glob('*_header.txt'))
    utc0, f0 = load_header_params(header)
    gps, pos = find_file_pairs(tel_dir)
    events_and_refs = list(reconstruct_stream(gps, pos, utc0, f0))
    events = [ev for ev, _ in events_and_refs]
    pos_index = [ref for _, ref in events_and_refs]
    return events, pos_index, utc0, f0
```

All assertions on `events` and `pos_map[i]` translate directly:
`pos_map[i]` becomes `pos_index[i]`.

### 8.2 Additional adversarial cases for `reconstruct_stream()`

Add the following to `DESIGN.md` §9 adversarial cases:

- **Pre-PPS_1 events**: inject 10 events before the first PPS record;
  verify they are yielded with `Quality.DEGRADED` and correct
  back-extrapolated timestamps once PPS_2 is observed.
- **Stream exhaustion mid-buffer**: truncate the last GPS file after its
  last event record but before the final PPS record; verify that the
  remaining buffered events are forward-extrapolated and tagged
  `Quality.DEGRADED`.
- **Back-to-back untrusted intervals**: mark three consecutive PPS pairs
  as untrusted (residual > tau); verify events inside them are tagged
  `Quality.UNTRUSTED` and that trusted events before and after are
  unaffected.

### 8.3 End-to-end streaming test

Add `tests/test_pipeline_stream.py` that runs the complete streaming
pipeline on the synthetic dataset from `monrad.synth.generate()`:

1. Instantiate `reconstruct_stream()` for telescope and probe.
2. Run `AlignmentAccumulator` on the telescope stream (stage 4).
3. Run `coincidence_stream()` + `PoseFitter` on a second telescope
   stream and the probe stream (stage 5).
4. Assert that the recovered `(t_x, t_y, θ, z_p)` lie within 3σ of
   their ground-truth values (same criterion as `DESIGN.md` §9 step 2).
5. Assert that peak RSS memory during the run does not exceed a
   configurable limit (default 512 MB for 1000 synthetic tracks).

Use `tracemalloc` or `resource.getrusage()` for the memory assertion.


## 9. Backward compatibility

`reconstruct()` may be retained temporarily as a thin wrapper:

```python
def reconstruct(gps_paths, pos_paths, utc0, f0=F0_DEFAULT, tau=PPS_TAU):
    """Deprecated. Use reconstruct_stream() for production pipelines."""
    import warnings
    warnings.warn(
        'reconstruct() is deprecated; use reconstruct_stream()',
        DeprecationWarning, stacklevel=2,
    )
    events, pos_index = [], []
    for ev, ref in reconstruct_stream(gps_paths, pos_paths, utc0, f0, tau):
        events.append(ev)
        pos_index.append(ref)
    pos_map = {ev.evt_seq: ref for ev, ref in zip(events, pos_index)}
    return events, pos_map
```

Remove this wrapper once all callers (including existing tests) have been
migrated to `reconstruct_stream()`.
