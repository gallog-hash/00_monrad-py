#!/usr/bin/env python3
"""
Decoder for BuS_Tracker GPS binary data files (*_GPS.bin).

File format:
  - bytes 0-3:  u32, number of rows
  - bytes 4-end: n_rows little-endian u64 values (one per row)
    Each u64 is structured as:
      bits  0-51 : clock counter (52 bits)
      bits 52-62 : GEN counter   (11 bits)
      bit  63    : FLAG           (1 bit)
"""

import struct
import numpy as np
import sys


class GPSDecoder:
    """Decoder for _GPS.bin files."""

    def __init__(self, filename: str):
        self.filename = filename

    def read(self) -> tuple[int, np.ndarray]:
        """
        Read file and return (n_rows, data).

        data shape: (n_rows,), dtype uint64
        """
        with open(self.filename, 'rb') as f:
            raw = f.read()

        if len(raw) < 4:
            raise ValueError(f"File too short: {len(raw)} bytes")

        n_rows = struct.unpack_from('<I', raw, 0)[0]

        expected = 4 + n_rows * 8
        if len(raw) != expected:
            raise ValueError(
                f"Size mismatch: expected {expected} bytes "
                f"({n_rows} rows × 8 bytes + 4-byte header), got {len(raw)}"
            )

        data = np.frombuffer(raw[4:], dtype='<u8')
        return n_rows, data

    @staticmethod
    def _parse_u64(value: int) -> dict:
        return {
            'CLK':  value        & 0xFFFFFFFFFFFFF,  # bits  0-51 (52 bits)
            'GEN': (value >> 52) & 0x7FF,             # bits 52-62 (11 bits)
            'FLAG':(value >> 63) & 0x1,               # bit  63    ( 1 bit)
        }

    def analyze(self):
        n_rows, data = self.read()
        print(f"File:    {self.filename}")
        print(f"Rows:    {n_rows}")
        for r in range(n_rows):
            p = self._parse_u64(int(data[r]))
            print(f"  row {r}:  CLK={p['CLK']}  GEN={p['GEN']}  FLAG={p['FLAG']}")

    def export_csv(self, output_file: str = None):
        """Export data as CSV with CLK, GEN, FLAG fields extracted from each u64."""
        if output_file is None:
            output_file = self.filename.replace('.bin', '_decoded.csv')

        n_rows, data = self.read()

        with open(output_file, 'w') as f:
            f.write("row_id,CLK,GEN,FLAG\n")
            for r in range(n_rows):
                p = self._parse_u64(int(data[r]))
                f.write(f"{r},{p['CLK']},{p['GEN']},{p['FLAG']}\n")

        print(f"Exported {n_rows} rows (CLK/GEN/FLAG) to {output_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python decode_gps.py <gps_bin_file> [--csv]")
        sys.exit(1)

    decoder = GPSDecoder(sys.argv[1])

    if '--csv' in sys.argv:
        idx = sys.argv.index('--csv')
        out = sys.argv[idx + 1] if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith('--') else None
        decoder.export_csv(out)
    else:
        decoder.analyze()


if __name__ == "__main__":
    main()
