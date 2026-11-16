"""Tests for the NMEA stream helpers used by the ZED-X20D reader (no ROS needed)."""

import pytest

from husky_localization.nmea_stream import (
    NmeaExtractor, checksum_ok, parse_ths, sentence_type, with_checksum)

GGA = with_checksum('GNGGA,123519.00,4243.84230,N,07340.83552,W,4,12,0.6,60.0,M,-32.0,M,1.0,0000')
THS = with_checksum('GNTHS,96.90,A')
UBX = bytes([0xB5, 0x62, 0x01, 0x07, 0x5C, 0x00]) + bytes(range(0, 256, 3)) + b'$\n\x00*'


def test_checksum():
    assert checksum_ok(GGA) and checksum_ok(THS)
    assert checksum_ok('$GPGLL,4916.45,N,12311.12,W,225444,A*31')  # textbook example
    assert not checksum_ok(THS[:-2] + '00')
    assert not checksum_ok('GNTHS,96.90,A*00')
    assert not checksum_ok('$*')


def test_extracts_sentences_between_binary_packets():
    stream = UBX + (GGA + '\r\n').encode() + UBX + (THS + '\r\n').encode() + UBX
    assert NmeaExtractor().feed(stream) == [GGA, THS]


def test_sentence_split_across_reads():
    ex = NmeaExtractor()
    data = (GGA + '\r\n' + THS + '\r\n').encode()
    out = []
    for i in range(0, len(data), 7):
        out += ex.feed(data[i:i + 7])
    assert out == [GGA, THS]


def test_bad_checksum_is_dropped_and_counted():
    ex = NmeaExtractor()
    broken = THS[:-2] + ('00' if THS[-2:] != '00' else '11')
    assert ex.feed((broken + '\r\n' + GGA + '\r\n').encode()) == [GGA]
    assert ex.bad_checksums == 1


def test_binary_with_dollar_signs_does_not_grow_buffer():
    ex = NmeaExtractor()
    for _ in range(1000):
        assert ex.feed(UBX) == []
    assert len(ex._buffer) <= 256
    assert ex.feed((THS + '\r\n').encode()) == [THS]


def test_sentence_type():
    assert sentence_type(THS) == 'THS'
    assert sentence_type(GGA) == 'GGA'


@pytest.mark.parametrize('body,expected', [
    ('GNTHS,96.90,A', (96.9, 'A')),
    ('GPTHS,0.00,A', (0.0, 'A')),
    ('GNTHS,359.99,E', (359.99, 'E')),
    ('GNTHS,,V', (None, 'V')),           # no heading solution yet
])
def test_parse_ths(body, expected):
    assert parse_ths(with_checksum(body)) == expected


@pytest.mark.parametrize('sentence', [
    with_checksum('GNTHS,abc,A'),       # not a number
    with_checksum('GNTHS,400.0,A'),     # out of range
    with_checksum('GNTHS,12.0'),        # missing mode
    GGA,                                # not THS
    THS[:-2] + '00',                    # bad checksum
])
def test_parse_ths_rejects(sentence):
    assert parse_ths(sentence) is None
