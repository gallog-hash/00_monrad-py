"""Tests for monrad.decoders.position.BinDecoder.or_visual's n_fibers_per_ribbon
parameter (docs/handoffs/2026-07-10-fibers-per-ribbon-pr-review-findings.md #5).
"""

import struct

from monrad.decoders.position import BinDecoder


def _write_bin(path, x_word: int, y_word: int) -> None:
    """Write a minimal 16-row, 1-column .bin file with one golden hit,
    repeated across the 16-row block (GEN=0) as the real DAQ format requires.
    """
    word = (x_word << 32) | y_word
    n_rows, n_cols = 16, 1
    with open(path, "wb") as f:
        f.write(struct.pack("<II", n_rows, n_cols))
        for _ in range(n_rows):
            f.write(struct.pack("<Q", word))


class TestOrVisualFibersPerRibbon:
    def test_default_n_matches_pos_half_bits(self, tmp_path, capsys):
        # ribbon bit 3, fiber bit 3 on X; ribbon/fiber bit 0 on Y.
        x_word = (1 << 3) | (1 << (10 + 3))
        y_word = 1 | (1 << 10)
        p = tmp_path / "probe.bin"
        _write_bin(p, x_word, y_word)

        BinDecoder(str(p)).or_visual()
        out = capsys.readouterr().out
        assert "X=33 (r=3, f=3)" in out

    def test_custom_n_reconstructs_different_channel(self, tmp_path, capsys):
        x_word = (1 << 3) | (1 << (10 + 3))
        y_word = 1 | (1 << 10)
        p = tmp_path / "probe.bin"
        _write_bin(p, x_word, y_word)

        BinDecoder(str(p)).or_visual(n_fibers_per_ribbon=5)
        out = capsys.readouterr().out
        # N=5: channel = N*r + f = 5*3 + 3 = 18, not the N=10 default's 33.
        assert "X=18 (r=3, f=3)" in out
        assert "X=33" not in out
