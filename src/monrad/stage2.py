"""
Stage 2 — sliding-window coincidence search.

Public API
----------
coincidence_stream(streams, detector_ids, window_ns)
    -> Iterator[list[tuple[int, TimedEvent, PosRef]]]
"""

import logging
from collections.abc import Iterator
from heapq import heappush, heappop

from .stage1 import TimedEvent, PosRef

log = logging.getLogger(__name__)

_WINDOW_NS_DEFAULT = 200   # ns — see DESIGN.md §4


def coincidence_stream(
    streams: list[Iterator[tuple[TimedEvent, PosRef]]],
    detector_ids: list[int],
    window_ns: int = _WINDOW_NS_DEFAULT,
) -> Iterator[list[tuple[int, TimedEvent, PosRef]]]:
    """
    k-way min-heap merge over n+1 stage-1 generators.

    Yields clusters of (detector_id, TimedEvent, PosRef) tuples.
    A cluster spans ≥ 2 distinct detectors with all t_ns within
    `window_ns` nanoseconds of the most-recently-popped event.

    PosRef is carried through transparently so that stage 3 callers
    downstream never need to look anything up.

    Parameters
    ----------
    streams      : one reconstruct_stream() iterator per detector,
                   in the same order as detector_ids
    detector_ids : integer label for each stream (arbitrary, unique)
    window_ns    : coincidence window in nanoseconds (default 200 ns)
    """
    if len(streams) != len(detector_ids):
        raise ValueError(
            'streams and detector_ids must have the same length'
        )
    if len(set(detector_ids)) != len(detector_ids):
        raise ValueError('detector_ids must be unique')

    stream_map: dict[int, Iterator[tuple[TimedEvent, PosRef]]] = {
        det_id: stream
        for det_id, stream in zip(detector_ids, streams)
    }

    # Heap element: (t_ns, det_id, counter, ev, ref).
    # The counter breaks ties between equal (t_ns, det_id) entries —
    # NamedTuples are comparable but a counter is cheaper.
    heap: list[tuple[int, int, int, TimedEvent, PosRef]] = []
    _ctr = 0
    for det_id in detector_ids:
        item = next(stream_map[det_id], None)
        if item is not None:
            ev, ref = item
            heappush(heap, (ev.t_ns, det_id, _ctr, ev, ref))
            _ctr += 1

    # Sliding deque: (det_id, TimedEvent, PosRef)
    deque: list[tuple[int, TimedEvent, PosRef]] = []

    while heap:
        t_now, det_id, _, ev, ref = heappop(heap)

        # Evict entries older than window_ns before the current event.
        deque = [
            (d, e, r) for d, e, r in deque
            if e.t_ns >= t_now - window_ns
        ]
        deque.append((det_id, ev, ref))

        # Emit when the window contains events from 2+ distinct detectors.
        if len({d for d, _, _ in deque}) >= 2:
            yield list(deque)

        # Advance the stream for this detector.
        nxt = next(stream_map[det_id], None)
        if nxt is not None:
            nev, nref = nxt
            heappush(heap, (nev.t_ns, det_id, _ctr, nev, nref))
            _ctr += 1
