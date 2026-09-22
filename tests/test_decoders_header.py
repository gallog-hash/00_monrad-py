"""Tests for the acquisition-run-aware header decoding in monrad.decoders.header.

One run writes one or more ``<stem>_header<NNN>.txt`` files and the DAQ can lay
the ``[GPS]`` UBX frame out in three different ways across them (see the module
docstring of monrad.decoders.header).  The fixtures below are byte-for-byte the
layouts observed in 01_data_2026.
"""

import itertools
import struct
from datetime import datetime, timedelta

import pytest

from monrad.decoders.header import (
    assemble_gps_frame,
    decode_ubx_tm2,
    find_header_files,
    parse_header,
    parse_header_group,
    parse_header_run,
    split_header_name,
    ubx_checksum,
    ubx_frame_length,
)
from monrad.timing import load_header_params

# UBX-TIM-TM2 frame from muon-telescope-17/20260910_094959_header030.txt.
FRAME = bytes.fromhex(
    "b5620d031c0000cd0b0083098309c9ac4b16dbf50800c9ac4b16a3f408001d0000005cb3"
)
# ...and from the split muon-probe-16/J11/20260910_094906 run, which the DAQ
# cut after the 0x0A byte of the `count` field: 9 bytes, then 27.
SPLIT_FRAME = bytes.fromhex(
    "b5620d031c0000cd0a0083098309aa714b1618f30400aa714b1621ee04001a0000004feb"
)
SPLIT_AT = 9

CONFIG_SECTION = "[J11]\nsaving time (min) = 300.000000\nMAROC threshold = 320\n\n"


def _make_frame(payload: bytes) -> bytes:
    body = bytes([0x0D, 0x03]) + struct.pack("<H", len(payload)) + payload
    return b"\xb5\x62" + body + bytes(ubx_checksum(body))


# A frame whose payload is packed with the bytes that mean something to the
# text layer — tab, newline, space, quote, backslash — so a breakpoint sweep
# actually exercises the escaping and the line parser, not just the assembly.
# (The real telescope frame happens to contain none of 0x09/0x20/0x22.)
HOSTILE_FRAME = _make_frame(
    bytes.fromhex("00cd0a008309830920225c16090a2000225c4b160a2022001a000000")
)


def _escape(data: bytes) -> str:
    """Escape bytes the way the DAQ writes GPS_String values."""
    out = []
    for b in data:
        if b == 0x5C:
            out.append("\\\\")
        elif 0x20 <= b <= 0x7E:
            out.append(chr(b))
        else:
            out.append(f"\\{b:02X}")
    return "".join(out)


def _gps_section(chunk: bytes) -> str:
    return f'[GPS]\nGPS_String_00 = "{_escape(chunk)}"\n'


def _write(path, text: str) -> None:
    path.write_text(text, encoding="latin-1")


def _write_run(tmp_path, chunks, *, stem: str = "20260910_094959"):
    """Lay a frame out across a run's header files, one chunk per file.

    Mirrors the DAQ: file 000 carries the module sections, and every file
    repeats its own ``[GPS]`` / ``GPS_String_00`` prefix around its slice.
    Returns the path of ``_header000.txt``.
    """
    for i, chunk in enumerate(chunks):
        body = (CONFIG_SECTION if i == 0 else "") + _gps_section(chunk)
        _write(tmp_path / f"{stem}_header{i:03d}.txt", body)
    return tmp_path / f"{stem}_header000.txt"


def _cuts(frame: bytes, n_files: int, step: int = 1):
    """Every way to cut ``frame`` into ``n_files`` consecutive chunks."""
    return itertools.combinations(range(0, len(frame) + 1, step), n_files - 1)


def _slice_at(frame: bytes, cuts) -> list[bytes]:
    bounds = [0, *cuts, len(frame)]
    return [frame[a:b] for a, b in zip(bounds, bounds[1:])]


class TestUbxFraming:
    def test_checksum_matches_real_frame(self):
        assert ubx_checksum(FRAME[2:-2]) == (FRAME[-2], FRAME[-1])

    def test_frame_length_accepts_complete_frame(self):
        assert ubx_frame_length(FRAME) == 36

    def test_frame_length_ignores_trailing_bytes(self):
        assert ubx_frame_length(FRAME + b"\x00\x01\x02") == 36

    @pytest.mark.parametrize(
        "data",
        [
            b"",
            b"\xb5\x62",
            FRAME[:SPLIT_AT],  # truncated
            FRAME[1:],  # no sync chars
            FRAME[:-1] + bytes([FRAME[-1] ^ 0xFF]),  # bad checksum
        ],
    )
    def test_frame_length_rejects_incomplete_or_corrupt(self, data):
        assert ubx_frame_length(data) is None


class TestAssembleGpsFrame:
    def test_single_complete_chunk(self):
        assert assemble_gps_frame([FRAME]) == FRAME

    def test_two_chunks_are_concatenated_in_order(self):
        chunks = [SPLIT_FRAME[:SPLIT_AT], SPLIT_FRAME[SPLIT_AT:]]
        assert assemble_gps_frame(chunks) == SPLIT_FRAME

    def test_complete_chunk_wins_over_concatenation(self):
        # A leading fragment that never completes must not corrupt a later
        # chunk that is a whole frame on its own.
        assert assemble_gps_frame([b"\x00\x01", FRAME]) == FRAME

    def test_unusable_chunks_fall_back_to_concatenation(self):
        # No valid frame anywhere: return what we have so the caller's
        # decode_ubx_tm2 can raise a specific error.
        assert assemble_gps_frame([b"\x01", b"\x02"]) == b"\x01\x02"

    def test_empty(self):
        assert assemble_gps_frame([]) == b""


class TestSplitHeaderName:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("20260910_094906_header000.txt", ("20260910_094906", 0)),
            ("20260910_094906_header001.txt", ("20260910_094906", 1)),
            ("20260909_110505_header030.txt", ("20260909_110505", 30)),
            ("20230418_191621_header.txt", ("20230418_191621", -1)),
            ("20260910_094906.bin", None),
            ("20260910_094906_GPS.bin", None),
        ],
    )
    def test_split(self, name, expected):
        assert split_header_name(name) == expected

    def test_unnumbered_sorts_first(self):
        unnumbered = split_header_name("x_header.txt")
        numbered = split_header_name("x_header000.txt")
        assert unnumbered is not None and numbered is not None
        assert unnumbered[1] < numbered[1]


class TestFindHeaderFiles:
    def test_orders_by_numeric_suffix_not_lexically(self, tmp_path):
        for idx in (30, 0, 1, 2):
            _write(tmp_path / f"20260910_094959_header{idx:03d}.txt", "[J11]\n")
        group = find_header_files(tmp_path / "20260910_094959_header030.txt")
        assert [p.name for p in group] == [
            "20260910_094959_header000.txt",
            "20260910_094959_header001.txt",
            "20260910_094959_header002.txt",
            "20260910_094959_header030.txt",
        ]

    def test_ignores_other_runs_and_non_headers(self, tmp_path):
        _write(tmp_path / "20260910_094906_header000.txt", "[J11]\n")
        _write(tmp_path / "20260910_094906_header001.txt", "[J11]\n")
        _write(tmp_path / "20260910_094959_header000.txt", "[J11]\n")
        (tmp_path / "20260910_094906.bin").write_bytes(b"")
        group = find_header_files(tmp_path / "20260910_094906_header000.txt")
        assert [p.name for p in group] == [
            "20260910_094906_header000.txt",
            "20260910_094906_header001.txt",
        ]

    def test_any_member_of_the_run_yields_the_whole_group(self, tmp_path):
        _write(tmp_path / "20260910_094906_header000.txt", "[J11]\n")
        _write(tmp_path / "20260910_094906_header001.txt", "[J11]\n")
        first = find_header_files(tmp_path / "20260910_094906_header000.txt")
        second = find_header_files(tmp_path / "20260910_094906_header001.txt")
        assert first == second

    def test_non_header_name_returns_itself(self, tmp_path):
        p = tmp_path / "notaheader.dat"
        assert find_header_files(p) == [p]

    def test_missing_file_returns_itself(self, tmp_path):
        p = tmp_path / "20260910_094906_header000.txt"
        assert find_header_files(p) == [p]


class TestGpsLayouts:
    """The three real-world layouts, end to end through load_header_params."""

    def _expect_utc0(self, frame: bytes) -> datetime:
        # load_header_params snaps the 100 Hz calibration pulse up to the next
        # whole second, where the PPS edge it anchors actually fires.
        rising = decode_ubx_tm2(frame)["timeR"]
        if rising.microsecond:
            return rising.replace(microsecond=0) + timedelta(seconds=1)
        return rising

    def test_whole_frame_in_header000(self, tmp_path):
        run = tmp_path / "20260910_094844_header000.txt"
        _write(run, CONFIG_SECTION + _gps_section(FRAME))

        mods = parse_header_run(run)
        assert mods["GPS"]["GPS_String_00"] == FRAME
        utc0, _ = load_header_params(run)
        assert utc0 == self._expect_utc0(FRAME)

    def test_frame_split_across_two_files_of_the_same_run(self, tmp_path):
        head = tmp_path / "20260910_094906_header000.txt"
        tail = tmp_path / "20260910_094906_header001.txt"
        _write(head, CONFIG_SECTION + _gps_section(SPLIT_FRAME[:SPLIT_AT]))
        # The continuation file repeats the [GPS] / GPS_String_00 prefix and
        # carries nothing else.
        _write(tail, _gps_section(SPLIT_FRAME[SPLIT_AT:]))

        # Neither file's own [GPS] section is a usable frame...
        head_only = parse_header(str(head))["GPS"]["GPS_String_00"]
        tail_only = parse_header(str(tail))["GPS"]["GPS_String_00"]
        assert head_only == SPLIT_FRAME[:SPLIT_AT]
        assert tail_only == SPLIT_FRAME[SPLIT_AT:]
        with pytest.raises(ValueError):
            decode_ubx_tm2(head_only)  # truncated payload
        with pytest.raises(ValueError):
            decode_ubx_tm2(tail_only)  # no sync chars

        # ...but the run's is, from either end.
        assert load_header_params(head) == load_header_params(tail)
        mods = parse_header_run(head)
        assert mods["GPS"]["GPS_String_00"] == SPLIT_FRAME
        # Config from header000 survives the merge.
        assert mods["J11"]["MAROC threshold"] == 320
        utc0, _ = load_header_params(head)
        assert utc0 == self._expect_utc0(SPLIT_FRAME)

    def test_frame_only_in_a_higher_numbered_sibling(self, tmp_path):
        head = tmp_path / "20260910_094959_header000.txt"
        gps = tmp_path / "20260910_094959_header030.txt"
        _write(head, CONFIG_SECTION)  # no [GPS] section at all
        _write(gps, _gps_section(FRAME))

        assert "GPS" not in parse_header(str(head))
        mods = parse_header_run(head)
        assert mods["GPS"]["GPS_String_00"] == FRAME
        assert mods["J11"]["MAROC threshold"] == 320
        # Pointing at either file resolves the same run.
        for path in (head, gps):
            utc0, _ = load_header_params(path)
            assert utc0 == self._expect_utc0(FRAME)

    def test_run_without_any_gps_raises_naming_every_file_inspected(self, tmp_path):
        head = tmp_path / "20260909_120348_header000.txt"
        _write(head, CONFIG_SECTION)
        with pytest.raises(ValueError, match="20260909_120348_header000.txt"):
            load_header_params(head)


class TestParseHeaderGroup:
    def test_later_files_win_for_non_gps_keys(self, tmp_path):
        a = tmp_path / "run_header000.txt"
        b = tmp_path / "run_header001.txt"
        _write(a, "[J11]\nMAROC threshold = 320\nVmon = 1.5\n")
        _write(b, "[J11]\nMAROC threshold = 470\n")
        mods = parse_header_group([a, b])
        assert mods["J11"]["MAROC threshold"] == 470
        assert mods["J11"]["Vmon"] == 1.5

    def test_gps_chunks_keyed_separately(self, tmp_path):
        a = tmp_path / "run_header000.txt"
        _write(
            a,
            "[GPS]\n"
            f'GPS_String_00 = "{_escape(FRAME)}"\n'
            f'GPS_String_01 = "{_escape(SPLIT_FRAME)}"\n',
        )
        mods = parse_header_group([a])
        assert mods["GPS"]["GPS_String_00"] == FRAME
        assert mods["GPS"]["GPS_String_01"] == SPLIT_FRAME

    def test_empty_group(self):
        assert parse_header_group([]) == {}


class TestEscapeDecoding:
    def test_real_hardware_leaves_tab_and_high_bytes_raw(self, tmp_path):
        # Hardware escapes control bytes as \XX but writes 0x09 and >=0x80 raw;
        # the synthetic generator escapes everything outside 0x20-0x7E.  Both
        # must decode to the same frame.
        raw = "".join(
            chr(b) if (b == 0x09 or b >= 0x20) and b != 0x5C else f"\\{b:02X}"
            for b in FRAME
        )
        p = tmp_path / "run_header000.txt"
        _write(p, f'[GPS]\nGPS_String_00 = "{raw}"\n')
        assert parse_header(str(p))["GPS"]["GPS_String_00"] == FRAME

    def test_missing_closing_quote_still_decodes(self, tmp_path):
        p = tmp_path / "run_header000.txt"
        _write(p, f'[GPS]\nGPS_String_00 = "{_escape(FRAME[:SPLIT_AT])}\n')
        assert parse_header(str(p))["GPS"]["GPS_String_00"] == FRAME[:SPLIT_AT]

    def test_literal_backslash_byte_roundtrips(self, tmp_path):
        # CK_A of the telescope frame is 0x5C; it must not eat the next byte.
        assert FRAME[-2] == 0x5C
        p = tmp_path / "run_header000.txt"
        _write(p, f'[GPS]\nGPS_String_00 = "{_escape(FRAME)}"\n')
        decoded = parse_header(str(p))["GPS"]["GPS_String_00"]
        assert decoded == FRAME
        assert struct.unpack_from("<H", decoded, 4)[0] == 28

    @pytest.mark.parametrize("byte", [0x09, 0x20, 0x22])
    def test_raw_payload_bytes_that_look_like_syntax_survive(self, tmp_path, byte):
        # 0x09/0x20/0x22 are written raw (not escaped) and sit at the chunk
        # edges when a split lands there; stripping the line would eat them.
        payload = bytes([byte]) + b"\xb5\x62" + bytes([byte])
        p = tmp_path / "run_header000.txt"
        _write(p, f'[GPS]\nGPS_String_00 = "{_escape(payload)}"\n')
        assert parse_header(str(p))["GPS"]["GPS_String_00"] == payload

    def test_trailing_raw_space_survives_an_unclosed_quote(self, tmp_path):
        # A chunk cut by the DAQ mid-line has no closing quote, so the trailing
        # 0x20 is payload rather than padding.
        payload = b"\xb5\x62\x20"
        p = tmp_path / "run_header000.txt"
        _write(p, f'[GPS]\nGPS_String_00 = "{_escape(payload)}\n')
        assert parse_header(str(p))["GPS"]["GPS_String_00"] == payload


class TestArbitraryBreakpoints:
    """The frame must reassemble no matter where the DAQ cut it, or how often.

    HOSTILE_FRAME carries 0x09/0x0A/0x20/0x22/0x5C in its payload so each cut
    also exercises the escaping and the line parser.
    """

    def _assert_run_reassembles(self, tmp_path, chunks, frame):
        head = _write_run(tmp_path, chunks)
        assert parse_header_run(head)["GPS"]["GPS_String_00"] == frame
        # ...and the config from file 000 is not lost in the merge.
        assert parse_header_run(head)["J11"]["MAROC threshold"] == 320

    @pytest.mark.parametrize("cut", range(len(HOSTILE_FRAME) + 1))
    def test_every_two_way_breakpoint(self, tmp_path, cut):
        chunks = _slice_at(HOSTILE_FRAME, (cut,))
        self._assert_run_reassembles(tmp_path, chunks, HOSTILE_FRAME)

    @pytest.mark.parametrize("cuts", _cuts(HOSTILE_FRAME, 3, step=4))
    def test_three_way_breakpoints(self, tmp_path, cuts):
        chunks = _slice_at(HOSTILE_FRAME, cuts)
        assert len(chunks) == 3
        self._assert_run_reassembles(tmp_path, chunks, HOSTILE_FRAME)

    @pytest.mark.parametrize("cuts", _cuts(HOSTILE_FRAME, 4, step=6))
    def test_four_way_breakpoints(self, tmp_path, cuts):
        chunks = _slice_at(HOSTILE_FRAME, cuts)
        assert len(chunks) == 4
        self._assert_run_reassembles(tmp_path, chunks, HOSTILE_FRAME)

    def test_one_byte_per_file(self, tmp_path):
        # The most extreme split possible: 36 files of one byte each.
        chunks = [HOSTILE_FRAME[i : i + 1] for i in range(len(HOSTILE_FRAME))]
        self._assert_run_reassembles(tmp_path, chunks, HOSTILE_FRAME)

    def test_three_way_split_end_to_end_through_load_header_params(self, tmp_path):
        head = _write_run(tmp_path, _slice_at(SPLIT_FRAME, (9, 20)))
        utc0, _ = load_header_params(head)
        rising = decode_ubx_tm2(SPLIT_FRAME)["timeR"]
        assert utc0 == rising.replace(microsecond=0) + timedelta(seconds=1)

    def test_empty_chunk_between_two_halves(self, tmp_path):
        # A file whose [GPS] section is present but empty must not break the run.
        chunks = [SPLIT_FRAME[:SPLIT_AT], b"", SPLIT_FRAME[SPLIT_AT:]]
        head = _write_run(tmp_path, chunks)
        assert parse_header_run(head)["GPS"]["GPS_String_00"] == SPLIT_FRAME


class TestAssemblyIsPositionIndependent:
    def test_leading_non_frame_chunk_does_not_defeat_reassembly(self):
        # The frame does not have to start at byte 0 of the joined buffer.
        chunks = [b"\x00\x11", FRAME[:9], FRAME[9:]]
        assert assemble_gps_frame(chunks) == FRAME

    def test_junk_at_both_ends(self):
        chunks = [b"\x00\x11", FRAME[:9], FRAME[9:], b"\xff\xee"]
        assert assemble_gps_frame(chunks) == FRAME

    def test_trailing_bytes_are_trimmed(self):
        assert assemble_gps_frame([FRAME[:9], FRAME[9:] + b"\xff"]) == FRAME

    def test_stale_fragment_before_a_complete_frame(self):
        assert assemble_gps_frame([FRAME[:9], FRAME]) == FRAME
