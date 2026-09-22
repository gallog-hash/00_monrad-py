#!/usr/bin/env python3
"""
Decoder for BuS_Tracker header.txt files.
Parses configuration parameters for detector modules.

One acquisition run writes one *or more* ``<stem>_header<NNN>.txt`` files that
share the same ``YYYYMMDD_HHMMSS`` stem.  The DAQ splits its capture buffer on
whatever boundary the GPS receiver happened to deliver, so the ``[GPS]``
section's UBX frame may be
  * complete in the single ``_header000.txt`` file,
  * split across ``_header000.txt`` + ``_header001.txt`` (each carrying its own
    ``[GPS]`` / ``GPS_String_00 = "…"`` prefix around its slice of the frame), or
  * absent from ``_header000.txt`` and living entirely in a higher-numbered
    sibling such as ``_header030.txt``.
:func:`parse_header` reads exactly one file; :func:`parse_header_group` (fed by
:func:`find_header_files`) merges a whole run and reassembles the frame, and is
what callers that need the GPS timestamp should use.
"""

import re
import struct
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

GPS_EPOCH = datetime(1980, 1, 6, 0, 0, 0)

UBX_SYNC = b"\xb5\x62"

# <stem>_header<NNN>.txt, with the index optional so synthetic "_header.txt"
# files (monrad.synthetic.generate) match too.
_HEADER_NAME_RE = re.compile(r"^(?P<stem>.+)_header(?P<idx>\d*)\.txt$", re.IGNORECASE)


def _decode_escaped_bytes(s: str) -> bytes:
    """
    Decode a string that mixes literal latin-1 bytes with \\XX hex escapes.

    Rules:
      \\XX  -> byte 0x XX  (two hex digits)
      \\\\  -> byte 0x5C   (literal backslash)
      c     -> ord(c)      (any other character)
    """
    result = bytearray()
    i = 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            nxt = s[i + 1]
            if nxt == "\\":
                result.append(0x5C)
                i += 2
            elif i + 3 <= len(s) and all(
                c in "0123456789ABCDEFabcdef" for c in s[i + 1 : i + 3]
            ):
                result.append(int(s[i + 1 : i + 3], 16))
                i += 3
            else:
                result.append(ord(s[i]))
                i += 1
        else:
            result.append(ord(s[i]) & 0xFF)
            i += 1
    return bytes(result)


def ubx_checksum(frame_body: bytes) -> tuple[int, int]:
    """
    8-bit Fletcher checksum (CK_A, CK_B) over a UBX frame body.

    ``frame_body`` is the frame from the class byte through the last payload
    byte — i.e. ``frame[2 : 6 + length]``, excluding the two sync chars and
    the checksum itself.
    """
    ck_a = ck_b = 0
    for byte in frame_body:
        ck_a = (ck_a + byte) & 0xFF
        ck_b = (ck_b + ck_a) & 0xFF
    return ck_a, ck_b


def ubx_frame_length(data: bytes) -> int | None:
    """
    Total on-the-wire length of the complete UBX frame at the start of ``data``.

    Returns ``None`` when ``data`` does not begin with a whole, checksum-valid
    frame — because it is truncated, does not start with the sync chars, or is
    corrupt.  Trailing bytes past the frame are ignored.
    """
    if len(data) < 8 or data[:2] != UBX_SYNC:
        return None
    length = struct.unpack_from("<H", data, 4)[0]
    total = 6 + length + 2
    if len(data) < total:
        return None
    if ubx_checksum(data[2 : 6 + length]) != (data[6 + length], data[6 + length + 1]):
        return None
    return total


def assemble_gps_frame(chunks: Sequence[bytes]) -> bytes:
    """
    Rebuild one UBX frame from the ``GPS_String`` chunks of an acquisition run.

    ``chunks`` are the decoded byte payloads in file order.  Handles all three
    ways the DAQ lays the frame out (see the module docstring), and does not
    care where the cut fell or how many chunks it produced:

      * a chunk that is already a complete, checksum-valid frame is returned
        as-is — this covers both the single-file case and the case where the
        frame sits alone in a higher-numbered sibling;
      * otherwise the chunks are consecutive slices of one frame, so they are
        concatenated in order and the first complete frame in the joined
        buffer is returned.  Searching the buffer rather than assuming the
        frame starts at byte 0 means a leading chunk that is not part of the
        frame does not defeat the reassembly, and trailing bytes past the
        frame are dropped.

    If nothing validates, the plain concatenation is returned so the caller
    (and :func:`decode_ubx_tm2`) can report a precise error instead of a
    generic "not found".
    """
    for chunk in chunks:
        if ubx_frame_length(chunk) is not None:
            return chunk

    buf = b"".join(chunks)
    start = buf.find(UBX_SYNC)
    while start != -1:
        total = ubx_frame_length(buf[start:])
        if total is not None:
            return buf[start : start + total]
        start = buf.find(UBX_SYNC, start + 1)
    return buf


def decode_ubx_tm2(data: bytes) -> Dict[str, Any]:
    """
    Decode a UBX-TIM-TM2 frame (class 0x0D, ID 0x03).

    Frame layout:
      [0-1]   sync    0xB5 0x62
      [2]     class   0x0D
      [3]     ID      0x03
      [4-5]   length  u16 little-endian (payload bytes, should be 28)
      [6-33]  payload 28 bytes (see below)
      [34-35] CK_A CK_B

    Payload:
      offset 0  U1  ch          Channel (0 = TIMEPULSE)
      offset 1  X1  flags       Flags
      offset 2  U2  count       Rising edge counter
      offset 4  U2  wnR         GPS week of last rising edge
      offset 6  U2  wnF         GPS week of last falling edge
      offset 8  U4  towMsR      TOW of rising edge (ms)
      offset 12 U4  towSubMsR   Sub-ms fraction of rising edge (ps)
      offset 16 U4  towMsF      TOW of falling edge (ms)
      offset 20 U4  towSubMsF   Sub-ms fraction of falling edge (ps)
      offset 24 U4  accEst      Accuracy estimate (ns)
    """
    if len(data) < 8:
        raise ValueError(f"Frame too short: {len(data)} bytes")
    if data[0] != 0xB5 or data[1] != 0x62:
        raise ValueError(f"Bad sync chars: {data[0]:02x} {data[1]:02x}")

    cls, msg_id = data[2], data[3]
    length = struct.unpack_from("<H", data, 4)[0]

    payload = data[6 : 6 + length]
    if len(payload) < 28:
        raise ValueError(f"Payload too short for TM2: {len(payload)} bytes")

    ch, flags, count, wnR, wnF, towMsR, towSubMsR, towMsF, towSubMsF, accEst = (
        struct.unpack_from("<BBHHHIIIII", payload, 0)
    )

    def gps_to_utc(week, tow_ms):
        return GPS_EPOCH + timedelta(weeks=week, milliseconds=tow_ms)

    return {
        "class": f"0x{cls:02X}",
        "id": f"0x{msg_id:02X}",
        "ch": ch,
        "flags": f"0x{flags:02X}",
        "count": count,
        "wnR": wnR,
        "towMsR": towMsR,
        "towSubMsR": towSubMsR,
        "timeR": gps_to_utc(wnR, towMsR),
        "wnF": wnF,
        "towMsF": towMsF,
        "towSubMsF": towSubMsF,
        "timeF": gps_to_utc(wnF, towMsF),
        "accEst": accEst,
    }


def parse_header(filename: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse a single BuS_Tracker header file.

    Returns a dict keyed by module name ([J11], [GPS], …).
    GPS string values are decoded to bytes and stored as bytes objects.

    This reads exactly the one file it is given, so the ``[GPS]`` section may
    be missing or hold only a slice of the UBX frame.  Use
    :func:`parse_header_group` to see a whole acquisition run.
    """
    modules = {}
    current_module = None

    # latin-1 preserves every byte value 0x00-0xFF unchanged
    with open(filename, "r", encoding="latin-1") as f:
        for line in f:
            # Only the line terminator comes off unconditionally: a quoted
            # value may legitimately start or end with a raw 0x09/0x20 payload
            # byte, and stripping the whole line would eat it.
            line = line.rstrip("\r\n")
            stripped = line.strip()

            module_match = re.match(r"\[(\w+)\]", stripped)
            if module_match:
                current_module = module_match.group(1)
                modules[current_module] = {}
                continue

            if not stripped or current_module is None:
                continue

            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()

                quoted = value.lstrip()
                if quoted.startswith('"'):
                    # Quote the value off without stripping its interior: the
                    # closing quote may be absent when the DAQ cut a split GPS
                    # string here, and then a trailing raw 0x20 is payload, not
                    # padding.
                    value = quoted[1:]
                    if value.endswith('"'):
                        value = value[:-1]
                else:
                    value = value.strip()

                # GPS string fields: decode escaped binary payload.
                # Real hardware writes "GPS String" (space); synth writes
                # "GPS_String_00" (underscore).  Normalise to underscore so
                # downstream code can use a single startswith check.
                norm_key = key.replace(" ", "_")
                if current_module == "GPS" and norm_key.startswith("GPS_String"):
                    value = _decode_escaped_bytes(value)
                    key = norm_key
                else:
                    try:
                        if "." not in value and value.replace("-", "").isdigit():
                            value = int(value)
                        elif (
                            value.replace(".", "")
                            .replace("-", "")
                            .replace("e", "")
                            .replace("E", "")
                            .isdigit()
                        ):
                            value = float(value)
                        elif "\t" in value or "\\0A" in value:
                            value = value.replace("\\0A", "")
                            value = [int(x) for x in value.split("\t") if x.strip()]
                    except ValueError:
                        pass

                modules[current_module][key] = value

    return modules


def split_header_name(path) -> tuple[str, int] | None:
    """
    Split a header file name into its ``(run_stem, index)``.

    ``20260910_094906_header001.txt`` -> ``("20260910_094906", 1)``.  A file
    with no numeric suffix (``…_header.txt``) gets index ``-1`` so it sorts
    ahead of any numbered sibling.  Returns ``None`` if the name is not a
    header file at all.
    """
    match = _HEADER_NAME_RE.match(Path(path).name)
    if match is None:
        return None
    idx = match.group("idx")
    return match.group("stem"), int(idx) if idx else -1


def find_header_files(path) -> List[Path]:
    """
    Every header file of the acquisition run that ``path`` belongs to.

    Given any one ``<stem>_header<NNN>.txt``, returns all siblings in the same
    directory that share ``<stem>``, ordered by ``<NNN>`` — which is the order
    the DAQ wrote them, and therefore the order a split GPS string must be
    reassembled in.  Falls back to ``[path]`` for a name that does not follow
    the convention.
    """
    p = Path(path)
    parts = split_header_name(p)
    if parts is None:
        return [p]

    stem = parts[0]
    if not p.parent.is_dir():
        return [p]

    group: list[tuple[int, str, Path]] = []
    for cand in p.parent.iterdir():
        cand_parts = split_header_name(cand)
        if cand_parts is not None and cand_parts[0] == stem:
            group.append((cand_parts[1], cand.name, cand))
    if not group:
        # `path` itself does not exist on disk; leave that for the caller.
        return [p]
    return [cand for _, _, cand in sorted(group)]


def parse_header_group(paths: Iterable[Path]) -> Dict[str, Dict[str, Any]]:
    """
    Parse and merge all header files of one acquisition run.

    ``paths`` must be in DAQ write order — use :func:`find_header_files`.
    Modules and keys are merged with later files winning, except for
    ``GPS_String*`` values: those are collected per key across the whole run
    and handed to :func:`assemble_gps_frame`, so the returned ``[GPS]`` section
    always carries the reassembled frame no matter which file (or files) it
    was spread over.
    """
    merged: Dict[str, Dict[str, Any]] = {}
    gps_chunks: Dict[str, List[bytes]] = {}

    for path in paths:
        for module, params in parse_header(str(path)).items():
            dst = merged.setdefault(module, {})
            for key, value in params.items():
                if (
                    module == "GPS"
                    and key.startswith("GPS_String")
                    and isinstance(value, bytes)
                ):
                    gps_chunks.setdefault(key, []).append(value)
                else:
                    dst[key] = value

    if gps_chunks:
        gps = merged.setdefault("GPS", {})
        for key, chunks in gps_chunks.items():
            gps[key] = assemble_gps_frame(chunks)

    return merged


def parse_header_run(path) -> Dict[str, Dict[str, Any]]:
    """Parse the whole acquisition run that the header file ``path`` belongs to."""
    return parse_header_group(find_header_files(path))


def print_header_info(modules: Dict[str, Dict[str, Any]]) -> None:
    """Pretty print the parsed header information."""
    for module_name, params in modules.items():
        print(f"\n{'=' * 60}")
        print(f"Module: {module_name}")
        print(f"{'=' * 60}")

        for key, value in params.items():
            if isinstance(value, bytes) and key.startswith("GPS_String"):
                print(f"{key} ({len(value)} bytes): {value.hex()}")
                try:
                    f = decode_ubx_tm2(value)
                    print(f"  class/id  : {f['class']} / {f['id']}")
                    print(
                        f"  ch        : {f['ch']}   flags: {f['flags']}   count: {f['count']}"
                    )
                    print(
                        f"  rising    : week {f['wnR']}  TOW {f['towMsR']} ms  sub {f['towSubMsR']} ps"
                    )
                    print(f"  rising UTC: {f['timeR']}")
                    print(
                        f"  falling   : week {f['wnF']}  TOW {f['towMsF']} ms  sub {f['towSubMsF']} ps"
                    )
                    print(f"  falling UTC: {f['timeF']}")
                    print(f"  accEst    : {f['accEst']} ns")
                except ValueError as e:
                    print(f"  (decode error: {e})")
            elif isinstance(value, bytes):
                print(f"{key} ({len(value)} bytes): {value.hex()}")
            elif isinstance(value, list):
                print(f"{key}:")
                print(f"  Length: {len(value)}")
                print(
                    f"  Values: {value[:10]}..."
                    if len(value) > 10
                    else f"  Values: {value}"
                )
            else:
                print(f"{key}: {value}")


def _print_usage() -> None:
    print("Usage: monrad-decode-header <header_file>")
    print("Example: monrad-decode-header 20230418_191621_header.txt")
    print()
    print(
        "One acquisition run may write several *_headerNNN.txt files sharing a\n"
        "yyyyMMdd_hhmmss stem; pass any one of them and the siblings are merged\n"
        "and the [GPS] UBX frame reassembled."
    )


def main() -> None:
    import sys

    argv = sys.argv[1:]
    if any(a in ("-h", "--help") for a in argv):
        _print_usage()
        sys.exit(0)
    if not argv:
        _print_usage()
        sys.exit(1)

    group = find_header_files(sys.argv[1])
    if len(group) > 1:
        print("Merging acquisition run files:")
        for path in group:
            print(f"  {path}")
    print_header_info(parse_header_group(group))


if __name__ == "__main__":
    main()
