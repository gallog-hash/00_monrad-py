"""JSON (de)serialization for a fitted :class:`AlignmentCorrection`.

A telescope's internal alignment is a stable, once-a-day calibration (the
stack is rigidly mounted; see ``monrad-align``).  Persisting it lets the
monitoring drivers load a correction instead of refitting it on every run.

The saved file also carries the provenance needed to use it safely: the
``z_tel`` it was fit with (the ``delta_z``/``tilt`` fit picks the geometric
middle plane from ``z_tel``, so a correction is only valid for the same plane
z-ordering -- :func:`load_alignment` enforces this), the day and files it came
from, the event count, and the per-event quality histogram.
"""

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .accumulator import AlignmentCorrection, PlaneCorrection

SCHEMA = "monrad-alignment/1"

# The six PlaneCorrection fields, in NamedTuple order — the single source of
# truth for how a plane is written to / read from JSON.
_PLANE_FIELDS = PlaneCorrection._fields


def save_alignment(
    correction: AlignmentCorrection,
    path: Path,
    *,
    date: str,
    z_tel: Sequence[float] | np.ndarray,
    files: Sequence[str],
    n_events: int,
    quality: Mapping[str, int] | None = None,
    utc_start_ns: int | None = None,
    utc_end_ns: int | None = None,
) -> None:
    """Write *correction* to *path* as JSON, with provenance metadata.

    Parameters
    ----------
    date:
        The ``YYYYMMDD`` (or ``YYYYMMDD_HHMMSS``) window label the correction
        was fit for -- a *file-name* time, which for a DAQ that names files in
        local time is **not** UTC.  Prefer ``utc_start_ns`` for time-keying.
    z_tel:
        Telescope plane z-positions (mm) the fit used.  Recorded so
        :func:`load_alignment` can reject a mismatched reuse (the fit is
        z-order-dependent).
    files:
        Names of the position files the correction was fit from.
    n_events:
        Number of valid events the fit was computed from.
    quality:
        Optional per-event quality histogram (``{name: count}``).
    utc_start_ns, utc_end_ns:
        The window's true UTC coverage in integer nanoseconds -- the same
        absolute clock as ``TimedEvent.t_ns`` -- so a consumer (e.g.
        ``monrad-monitor``'s time-varying alignment) can map a coincidence to
        its window by real UTC time rather than by the file-name ``date`` label
        (the two differ by the DAQ's UTC offset).  Written both as exact
        integer-ns fields (authoritative) and as human-readable ISO strings.
        Omitted when ``None`` (e.g. an empty window, or a caller that has no
        timing) -- consumers then fall back to the ``date`` label.
    """
    path = Path(path)
    payload = {
        "schema": SCHEMA,
        "date": date,
        "computed_utc": datetime.now(timezone.utc).isoformat(),
        "z_tel": [float(z) for z in np.asarray(z_tel, dtype=float)],
        "files": list(files),
        "n_events": int(n_events),
        "quality": dict(quality) if quality is not None else {},
        "needs_correction": bool(correction.needs_correction),
        "planes": [
            {f: float(getattr(p, f)) for f in _PLANE_FIELDS} for p in correction.planes
        ],
    }
    if utc_start_ns is not None:
        payload["utc_start_ns"] = int(utc_start_ns)
        payload["utc_start"] = _ns_to_iso(utc_start_ns)
    if utc_end_ns is not None:
        payload["utc_end_ns"] = int(utc_end_ns)
        payload["utc_end"] = _ns_to_iso(utc_end_ns)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def _ns_to_iso(t_ns: int) -> str:
    """An integer-ns UTC time as a human-readable ISO-8601 string."""
    return datetime.fromtimestamp(t_ns / 1e9, tz=timezone.utc).isoformat()


def load_alignment(
    path: Path,
    *,
    expect_z_tel: Sequence[float] | np.ndarray | None = None,
) -> AlignmentCorrection:
    """Load an :class:`AlignmentCorrection` written by :func:`save_alignment`.

    Only the ``planes`` and ``needs_correction`` fields are reconstructed —
    the rest is provenance metadata.

    Parameters
    ----------
    expect_z_tel:
        If given, raise :class:`ValueError` when it differs from the ``z_tel``
        recorded in the file.  The ``delta_z``/``tilt`` fit depends on the
        plane z-ordering, so a correction fit with one ``z_tel`` must not be
        applied to a run using another.
    """
    path = Path(path)
    with open(path) as fh:
        payload = json.load(fh)

    schema = payload.get("schema")
    if schema != SCHEMA:
        raise ValueError(
            f"{path}: unexpected alignment schema {schema!r} (expected {SCHEMA!r})"
        )

    if expect_z_tel is not None:
        saved = np.asarray(payload.get("z_tel", []), dtype=float)
        want = np.asarray(expect_z_tel, dtype=float)
        if saved.shape != want.shape or not np.allclose(saved, want):
            raise ValueError(
                f"{path}: alignment was fit with z_tel={saved.tolist()}, but this "
                f"run uses z_tel={want.tolist()}; the delta_z/tilt fit is "
                "z-order-dependent, so the saved correction cannot be reused. "
                "Recompute the alignment with matching --z-tel."
            )

    planes = [
        PlaneCorrection(**{f: float(p[f]) for f in _PLANE_FIELDS})
        for p in payload["planes"]
    ]
    return AlignmentCorrection(planes, bool(payload["needs_correction"]))
