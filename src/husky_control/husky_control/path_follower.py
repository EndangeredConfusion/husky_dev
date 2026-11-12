'''
Pure path-following math for point_controller (no ROS imports).

Kept separate from the ROS node so it can be unit-tested offline with a
simple 2D robot model (see test/test_path_follower.py).
'''

import numpy as np
from scipy.interpolate import make_interp_spline


def angle_wrap(angle):
    ''' wrap an angle to [-pi, pi) '''
    return (angle + np.pi) % (2 * np.pi) - np.pi


def yaw_from_quaternion(x, y, z, w):
    ''' heading (rotation about z) from a quaternion '''
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def chaikin_smooth(points, iterations=3, keep_ends=True):
    ''' chaikin smoothing algorithm to help the spline cut corners '''
    pts = np.asarray(points, dtype=float)
    for _ in range(iterations):
        new_pts = []
        if keep_ends:
            new_pts.append(pts[0])
        for i in range(len(pts) - 1):
            p0 = pts[i]
            p1 = pts[i + 1]
            new_pts.extend([0.75 * p0 + 0.25 * p1, 0.25 * p0 + 0.75 * p1])
        if keep_ends:
            new_pts.append(pts[-1])
        pts = np.array(new_pts)
    return pts


def fit_continuous_path(path):
    ''' cubic spline through the (smoothed) points, parameterised by t in [0, 1] '''
    path_array = np.asarray(path, dtype=float)
    k = min(3, len(path_array) - 1)
    t = np.linspace(0, 1, len(path_array))
    return make_interp_spline(t, path_array, k=k)


def remove_duplicates(points, eps=1e-6):
    ''' drop consecutive repeated waypoints (they break the tangent) '''
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) == 0:
        return pts
    keep = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - keep[-1]) > eps:
            keep.append(p)
    return np.array(keep)


class StanleyFollower:
    '''
    Stanley control along a smoothed spline, adapted to a differential drive:
        omega = k_heading * heading_error - atan2(k_cross * e, v)
    where e > 0 means the robot is to the LEFT of the path.
    https://ai.stanford.edu/~gabeh/papers/hoffmann_stanley_control07.pdf
    '''

    def __init__(self, nominal_velocity=0.3, k_heading=1.0, k_cross=1.0,
                 max_omega=1.0, goal_tolerance=0.15, n_samples=300,
                 search_window=0.25):
        self.nominal_velocity = nominal_velocity
        self.k_heading = k_heading
        self.k_cross = k_cross
        self.max_omega = max_omega
        self.goal_tolerance = goal_tolerance
        self.n_samples = n_samples
        # fraction of the path ahead of the last closest point we search in
        self.search_window = search_window
        self.clear()

    def clear(self):
        self.path_obj = None
        self.path_deriv = None
        self.path_pts = None
        self.t_vals = None
        self.goal = None
        self.last_idx = 0

    @property
    def active(self):
        return self.path_obj is not None

    def set_path(self, points, current_xy=None):
        '''
        points: (N, 2) waypoints in metres, same frame as the localization.
        current_xy: robot position; prepended when only one waypoint is given
        so a single "go to (x, y)" message works.
        '''
        pts = remove_duplicates(points)
        if current_xy is not None and len(pts) == 1:
            pts = remove_duplicates(np.vstack([np.asarray(current_xy, float), pts]))
        if len(pts) < 2:
            self.clear()
            return False
        smoothed = chaikin_smooth(pts)
        self.path_obj = fit_continuous_path(smoothed)
        self.path_deriv = self.path_obj.derivative(1)
        self.t_vals = np.linspace(0, 1, self.n_samples)
        self.path_pts = self.path_obj(self.t_vals)
        self.goal = pts[-1]
        self.last_idx = 0
        return True

    def step(self, x, y, theta):
        ''' returns (v, omega, done) '''
        if not self.active:
            return 0.0, 0.0, True

        if np.hypot(x - self.goal[0], y - self.goal[1]) < self.goal_tolerance:
            self.clear()
            return 0.0, 0.0, True

        # closest point, searched only forward from the last one so the robot
        # never snaps back to an earlier part of a curved path
        lo = self.last_idx
        hi = min(len(self.t_vals), lo + max(2, int(self.search_window * self.n_samples)))
        deltas = self.path_pts[lo:hi] - np.array((x, y))
        min_idx = lo + int(np.argmin(np.sum(deltas * deltas, axis=1)))
        self.last_idx = min_idx
        t_star = self.t_vals[min_idx]
        px, py = self.path_pts[min_idx]

        tangent = self.path_deriv(t_star)
        tangent = tangent / (np.linalg.norm(tangent) + 1e-9)
        left_normal = np.array([-tangent[1], tangent[0]])
        cross_track_e = float(np.dot(np.array([x - px, y - py]), left_normal))

        # overshot the end of the path (missed the tolerance circle): stop
        # rather than drive on forever
        if min_idx == len(self.t_vals) - 1 and np.dot(
                np.array([x, y]) - self.goal, tangent) > 0:
            self.clear()
            return 0.0, 0.0, True

        path_theta = np.arctan2(tangent[1], tangent[0])
        heading_error = angle_wrap(path_theta - theta)

        v = self.nominal_velocity
        # facing the wrong way: creep forward while turning (keeps EKF alive)
        if abs(heading_error) > np.pi / 2:
            v = self.nominal_velocity / 2

        omega = (self.k_heading * heading_error
                 - np.arctan2(self.k_cross * cross_track_e, abs(v) + 1e-6))
        omega = float(np.clip(omega, -self.max_omega, self.max_omega))
        return v, omega, False
