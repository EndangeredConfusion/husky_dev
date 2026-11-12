"""Tests for the pure GPS heading / antenna helpers used by gps_pose."""

import math

import pytest

from husky_localization.gps_pose_math import (
    antenna_to_base, enu_yaw_to_compass, heading_to_enu_yaw, odom_to_local,
    yaw_from_quaternion)


@pytest.mark.parametrize('compass_deg,yaw', [
    (0.0, math.pi / 2),     # facing north
    (90.0, 0.0),            # facing east
    (180.0, -math.pi / 2),  # facing south
    (270.0, -math.pi),      # facing west (wrapped to [-pi, pi))
])
def test_cw_from_north(compass_deg, yaw):
    assert heading_to_enu_yaw(math.radians(compass_deg), 'cw_from_north') == \
        pytest.approx(yaw)


def test_enu_yaw_passthrough_and_offset():
    assert heading_to_enu_yaw(0.3, 'enu_yaw') == pytest.approx(0.3)
    assert heading_to_enu_yaw(0.3, 'enu_yaw', offset_rad=0.2) == pytest.approx(0.5)


def test_bad_convention():
    with pytest.raises(ValueError):
        heading_to_enu_yaw(0.0, 'degrees')


def test_yaw_from_quaternion():
    half = 0.4
    assert yaw_from_quaternion(0.0, 0.0, math.sin(half), math.cos(half)) == \
        pytest.approx(2 * half)


def test_antenna_to_base():
    # antenna 0.5 m ahead of the centre, robot facing north
    assert antenna_to_base(10.0, 20.0, math.pi / 2, 0.5, 0.0) == \
        pytest.approx((10.0, 19.5))
    # antenna 0.2 m to the left, robot facing east
    assert antenna_to_base(10.0, 20.0, 0.0, 0.0, 0.2) == pytest.approx((10.0, 19.8))


@pytest.mark.parametrize('yaw,compass_deg', [
    (math.pi / 2, 0.0), (0.0, 90.0), (-math.pi / 2, 180.0), (math.pi, 270.0)])
def test_enu_yaw_to_compass(yaw, compass_deg):
    assert math.degrees(enu_yaw_to_compass(yaw)) == pytest.approx(compass_deg)


def test_compass_round_trip_through_gps_pose_conversion():
    for yaw in (-3.0, -1.0, 0.0, 0.7, 2.5):
        assert heading_to_enu_yaw(enu_yaw_to_compass(yaw), 'cw_from_north') == \
            pytest.approx(yaw)


def test_odom_to_local():
    # odom origin at (10, 20) facing north: driving 2 m "forward" goes north
    e, n, yaw = odom_to_local(2.0, 0.0, 0.0, 10.0, 20.0, math.pi / 2)
    assert (e, n, yaw) == pytest.approx((10.0, 22.0, math.pi / 2))
    # 1 m to the robot's left (odom +y) while facing north is west
    e, n, _ = odom_to_local(0.0, 1.0, 0.0, 10.0, 20.0, math.pi / 2)
    assert (e, n) == pytest.approx((9.0, 20.0))
