'''
Offline tests for husky_control.path_follower (no ROS needed).

A unicycle model is driven by StanleyFollower at 50 Hz, the same rate as
point_controller, and the tests check that the robot reaches the goal,
stays close to the path and stops.

Run with:  python3 -m pytest test/test_path_follower.py -v
'''

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from husky_control.path_follower import (  # noqa: E402
    StanleyFollower, angle_wrap, chaikin_smooth, remove_duplicates,
    yaw_from_quaternion)

DT = 0.02


def simulate(follower, x, y, theta, max_time=120.0):
    ''' unicycle integration; returns (trajectory (N,3), done) '''
    traj = [(x, y, theta)]
    for _ in range(int(max_time / DT)):
        v, omega, done = follower.step(x, y, theta)
        if done:
            return np.array(traj), True
        theta = angle_wrap(theta + omega * DT)
        x += v * np.cos(theta) * DT
        y += v * np.sin(theta) * DT
        traj.append((x, y, theta))
    return np.array(traj), False


def max_deviation(traj, follower_pts):
    ''' largest distance from any trajectory point to the sampled path '''
    d = traj[:, None, :2] - follower_pts[None, :, :]
    return float(np.sqrt(np.min(np.sum(d * d, axis=2), axis=1)).max())


def test_angle_wrap():
    assert angle_wrap(0.0) == pytest.approx(0.0)
    assert angle_wrap(3 * np.pi / 2) == pytest.approx(-np.pi / 2)
    assert angle_wrap(-3 * np.pi / 2) == pytest.approx(np.pi / 2)


def test_yaw_from_quaternion():
    for yaw in (-2.5, -0.3, 0.0, 1.0, 3.0):
        q = (0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2))
        assert yaw_from_quaternion(*q) == pytest.approx(yaw)


def test_remove_duplicates_and_chaikin_keep_ends():
    pts = remove_duplicates([[0, 0], [0, 0], [1, 0], [1, 0], [1, 1]])
    assert pts.tolist() == [[0, 0], [1, 0], [1, 1]]
    smooth = chaikin_smooth(pts)
    assert smooth[0].tolist() == [0, 0]
    assert smooth[-1].tolist() == [1, 1]


def test_rejects_degenerate_paths():
    f = StanleyFollower()
    assert not f.set_path([[1.0, 1.0]])            # one point, no pose
    assert not f.set_path([[1.0, 1.0], [1.0, 1.0]])
    assert not f.active
    assert f.step(0, 0, 0) == (0.0, 0.0, True)


def test_single_point_goal():
    f = StanleyFollower()
    assert f.set_path([[3.0, 0.0]], current_xy=(0.0, 0.0))
    traj, done = simulate(f, 0.0, 0.0, 0.0)
    assert done
    assert np.hypot(traj[-1, 0] - 3.0, traj[-1, 1]) < f.goal_tolerance


def test_straight_line_with_offset_start():
    f = StanleyFollower()
    f.set_path([[0.0, 0.0], [5.0, 0.0]])
    traj, done = simulate(f, 0.0, 0.5, 0.0)
    assert done
    assert np.hypot(traj[-1, 0] - 5.0, traj[-1, 1]) < f.goal_tolerance
    # cross-track error should have decayed well before the end
    assert abs(traj[len(traj) // 2, 1]) < 0.1


def test_curved_path_tracking():
    waypoints = [[0, 0], [2, 0], [3, 1], [3, 3], [1, 4]]
    f = StanleyFollower()
    f.set_path(waypoints)
    pts = f.path_pts.copy()
    traj, done = simulate(f, 0.0, 0.0, 0.0)
    assert done
    assert np.hypot(*(traj[-1, :2] - np.array([1, 4]))) < f.goal_tolerance
    assert max_deviation(traj, pts) < 0.2


def test_starts_facing_backwards():
    f = StanleyFollower()
    f.set_path([[0.0, 0.0], [4.0, 0.0]])
    traj, done = simulate(f, 0.0, 0.0, np.pi)
    assert done
    assert np.hypot(traj[-1, 0] - 4.0, traj[-1, 1]) < f.goal_tolerance


def test_commands_respect_limits():
    f = StanleyFollower(nominal_velocity=0.3, max_omega=1.0)
    f.set_path([[0, 0], [1, 2], [-1, 3]])
    x, y, th = 0.0, 0.0, -np.pi / 2
    for _ in range(2000):
        v, omega, done = f.step(x, y, th)
        if done:
            break
        assert 0.0 <= v <= 0.3 + 1e-9
        assert abs(omega) <= 1.0 + 1e-9
        th = angle_wrap(th + omega * DT)
        x += v * np.cos(th) * DT
        y += v * np.sin(th) * DT


def test_new_path_replaces_old():
    f = StanleyFollower()
    f.set_path([[0, 0], [5, 0]])
    simulate(f, 0.0, 0.0, 0.0, max_time=3.0)
    assert f.last_idx > 0
    f.set_path([[0, 0], [0, 3]])
    assert f.last_idx == 0
    traj, done = simulate(f, 0.0, 0.0, np.pi / 2)
    assert done
    assert np.hypot(traj[-1, 0], traj[-1, 1] - 3.0) < f.goal_tolerance


def test_overshoot_stops():
    ''' tiny tolerance + big speed: the robot misses the circle but must stop '''
    f = StanleyFollower(nominal_velocity=2.0, goal_tolerance=0.01)
    f.set_path([[0.0, 0.0], [2.0, 0.0]])
    traj, done = simulate(f, 0.0, 0.03, 0.0, max_time=10.0)
    assert done
    assert traj[-1, 0] < 2.5
