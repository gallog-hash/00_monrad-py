#!/usr/bin/env python3
"""
Decoder for BuS_Tracker header.txt files.
Parses configuration parameters for detector modules.
"""

import re
import struct
from datetime import datetime, timedelta
from typing import Dict, List, Any

GPS_EPOCH = datetime(1980, 1, 6, 0, 0, 0)


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
        if s[i] == '\\' and i + 1 < len(s):
            nxt = s[i + 1]
            if nxt == '\\':
                result.append(0x5C)
                i += 2
            elif i + 3 <= len(s) and all(c in '0123456789ABCDEFabcdef' for c in s[i+1:i+3]):
                result.append(int(s[i+1:i+3], 16))
                i += 3
            else:
                result.append(ord(s[i]))
                i += 1
        else:
            result.append(ord(s[i]) & 0xFF)
            i += 1
    return bytes(result)


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
    length = struct.unpack_from('<H', data, 4)[0]

    payload = data[6:6 + length]
    if len(payload) < 28:
        raise ValueError(f"Payload too short for TM2: {len(payload)} bytes")

    ch, flags, count, wnR, wnF, towMsR, towSubMsR, towMsF, towSubMsF, accEst = \
        struct.unpack_from('<BBHHHIIIII', payload, 0)

    def gps_to_utc(week, tow_ms):
        return GPS_EPOCH + timedelta(weeks=week, milliseconds=tow_ms)

    return {
        'class':      f'0x{cls:02X}',
        'id':         f'0x{msg_id:02X}',
        'ch':         ch,
        'flags':      f'0x{flags:02X}',
        'count':      count,
        'wnR':        wnR,
        'towMsR':     towMsR,
        'towSubMsR':  towSubMsR,
        'timeR':      gps_to_utc(wnR, towMsR),
        'wnF':        wnF,
        'towMsF':     towMsF,
        'towSubMsF':  towSubMsF,
        'timeF':      gps_to_utc(wnF, towMsF),
        'accEst':     accEst,
    }


def parse_header(filename: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse a BuS_Tracker header file.

    Returns a dict keyed by module name ([J11], [GPS], …).
    GPS string values are decoded to bytes and stored as bytes objects.
    """
    modules = {}
    current_module = None

    # latin-1 preserves every byte value 0x00-0xFF unchanged
    with open(filename, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()

            module_match = re.match(r'\[(\w+)\]', line)
            if module_match:
                current_module = module_match.group(1)
                modules[current_module] = {}
                continue

            if not line or current_module is None:
                continue

            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]

                # GPS string fields: decode escaped binary payload.
                # Real hardware writes "GPS String" (space); synth writes
                # "GPS_String_00" (underscore).  Normalise to underscore so
                # downstream code can use a single startswith check.
                norm_key = key.replace(' ', '_')
                if current_module == 'GPS' and norm_key.startswith('GPS_String'):
                    value = _decode_escaped_bytes(value)
                    key = norm_key
                else:
                    try:
                        if '.' not in value and value.replace('-', '').isdigit():
                            value = int(value)
                        elif value.replace('.', '').replace('-', '').replace('e', '').replace('E', '').isdigit():
                            value = float(value)
                        elif '\t' in value or '\\0A' in value:
                            value = value.replace('\\0A', '')
                            value = [int(x) for x in value.split('\t') if x.strip()]
                    except ValueError:
                        pass

                modules[current_module][key] = value

    return modules


def print_header_info(modules: Dict[str, Dict[str, Any]]) -> None:
    """Pretty print the parsed header information."""
    for module_name, params in modules.items():
        print(f"\n{'='*60}")
        print(f"Module: {module_name}")
        print(f"{'='*60}")

        for key, value in params.items():
            if isinstance(value, bytes) and key.startswith('GPS_String'):
                print(f"{key} ({len(value)} bytes): {value.hex()}")
                try:
                    f = decode_ubx_tm2(value)
                    print(f"  class/id  : {f['class']} / {f['id']}")
                    print(f"  ch        : {f['ch']}   flags: {f['flags']}   count: {f['count']}")
                    print(f"  rising    : week {f['wnR']}  TOW {f['towMsR']} ms  sub {f['towSubMsR']} ps")
                    print(f"  rising UTC: {f['timeR']}")
                    print(f"  falling   : week {f['wnF']}  TOW {f['towMsF']} ms  sub {f['towSubMsF']} ps")
                    print(f"  falling UTC: {f['timeF']}")
                    print(f"  accEst    : {f['accEst']} ns")
                except ValueError as e:
                    print(f"  (decode error: {e})")
            elif isinstance(value, bytes):
                print(f"{key} ({len(value)} bytes): {value.hex()}")
            elif isinstance(value, list):
                print(f"{key}:")
                print(f"  Length: {len(value)}")
                print(f"  Values: {value[:10]}..." if len(value) > 10 else f"  Values: {value}")
            else:
                print(f"{key}: {value}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python decode_header.py <header_file>")
        print("Example: python decode_header.py 20230418_191621_header.txt")
        sys.exit(1)

    header_file = sys.argv[1]
    modules = parse_header(header_file)
    print_header_info(modules)
