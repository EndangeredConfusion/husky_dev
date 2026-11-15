"""Pure NMEA helpers for the u-blox ZED-X20D (no ROS), so they can be unit-tested.

The receiver's USB port carries NMEA text mixed with binary UBX packets.
NmeaExtractor pulls complete, checksum-valid NMEA sentences out of that byte
stream and skips everything else.

The X20D heading firmware (HDG 2.00) reports dual-antenna heading as the NMEA
THS sentence (true heading and status), which replaces HDT:

    $--THS,<heading deg true>,<mode>*hh

mode: A = autonomous (valid), E = estimated (dead reckoning), M = manual input,
S = simulator, V = not valid.
"""

import re
from typing import List, Optional, Tuple

# $ + talker (2) + type (3) + fields + * + 2 hex; printable ASCII only, so
# binary UBX bytes can never be part of a match.
_SENTENCE = re.compile(rb'\$[A-Z]{2}[A-Z]{3}[\x20-\x29\x2b-\x7e]*\*[0-9A-Fa-f]{2}')

# Keep at most this many unmatched bytes (a partial sentence at the end).
_MAX_TAIL = 256


def checksum_ok(sentence: str) -> bool:
    """Check the *hh checksum (XOR of the characters between '$' and '*')."""
    if not sentence.startswith('$') or len(sentence) < 4 or sentence[-3] != '*':
        return False
    body, given = sentence[1:-3], sentence[-2:]
    value = 0
    for char in body:
        value ^= ord(char)
    try:
        return value == int(given, 16)
    except ValueError:
        return False


def with_checksum(body: str) -> str:
    """Build '$<body>*hh' (used by tests and tools)."""
    value = 0
    for char in body:
        value ^= ord(char)
    return f'${body}*{value:02X}'


class NmeaExtractor:
    """Collect complete NMEA sentences from a mixed NMEA/UBX byte stream."""

    def __init__(self) -> None:
        self._buffer = b''
        self.bad_checksums = 0

    def feed(self, data: bytes) -> List[str]:
        """Add received bytes; return the complete, valid sentences found."""
        self._buffer += data
        sentences = []
        end = 0
        for match in _SENTENCE.finditer(self._buffer):
            text = match.group().decode('ascii')
            end = match.end()
            if checksum_ok(text):
                sentences.append(text)
            else:
                self.bad_checksums += 1
        tail = self._buffer[end:]
        # A sentence can only be in progress from the last '$' onward.
        start = tail.rfind(b'$')
        tail = tail[start:] if start >= 0 else b''
        self._buffer = tail[-_MAX_TAIL:]
        return sentences


def sentence_type(sentence: str) -> str:
    """'$GNTHS,...' -> 'THS'."""
    return sentence[3:6]


def parse_ths(sentence: str) -> Optional[Tuple[Optional[float], str]]:
    """Parse a THS sentence into (heading_deg or None, mode).

    Returns None if it is not a well-formed THS sentence.  heading_deg is
    None when the field is empty (no heading solution).
    """
    if not checksum_ok(sentence) or sentence_type(sentence) != 'THS':
        return None
    fields = sentence[1:-3].split(',')
    if len(fields) < 3:
        return None
    heading_field, mode = fields[1], fields[2]
    if not heading_field:
        return None, mode
    try:
        heading = float(heading_field)
    except ValueError:
        return None
    if not 0.0 <= heading < 360.0:
        return None
    return heading, mode
