#!/usr/bin/env python3
"""
Stanley path-following controller for a differential-drive robot (Clearpath Husky A200).

Inputs
  points_topic        path waypoints; message type set by `points_msg_type`:
                        int32    std_msgs/Int32MultiArray    flat [x0, y0, x1, y1, ...]
                        float64  std_msgs/Float64MultiArray  flat [x0, y0, x1, y1, ...]
                        path     nav_msgs/Path
                      flat arrays are converted to metres as  value * points_scale + points_origin
  localization_topic  robot pose; message type set by `pose_source`:
                        ekf_state  husky_interfaces/EkfState (msg.data.position_m.x/y, msg.data.theta_rad)
                        odometry   nav_msgs/Odometry
Outputs
  cmd_vel_topic       geometry_msgs/Twist or TwistStamped (`cmd_vel_type`: auto | twist | twist_stamped)
  ~/smoothed_path     nav_msgs/Path of the curve being tracked, for RViz

Behaviour
  - A new path replaces the current one. Re-sending the identical path while it is active is ignored.
  - The path subscription accepts both latched (transient_local) and normal (volatile) publishers.
    With a latched publisher, a path published before this node started is still received.
  - The robot stops (zero velocity) at the goal, when the pose goes stale, and on shutdown (Ctrl-C / SIGTERM).
  - If the robot faces away from where it should go by more than `rotate_enter_angle`, it turns in place first.
  - A negative `nominal_velocity` drives the path in reverse.

Conventions: REP-103 (x forward, theta counter-clockwise positive, angular.z > 0 turns left).
"""

import math
import signal
import time
from dataclasses import dataclass, fields

import numpy as np
from scipy.interpolate import make_interp_spline

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist, TwistStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Int32MultiArray, Float64MultiArray

try:  # Jazzy and later
    from rclpy.event_handler import SubscriptionEventCallbacks
except ImportError:  # Humble
    from rclpy.qos_event import SubscriptionEventCallbacks


# ----------------------------------------------------------------------------
# Geometry helpers (no ROS dependencies, so they can be unit-tested directly)
# ----------------------------------------------------------------------------

def angle_wrap(angle):
    """Wrap to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def dedupe(points, eps=1e-9):
    """Drop consecutive duplicate points (they break the spline fit)."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) < 2:
        return pts
    keep = np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > eps]
    return pts[keep]


def chaikin_smooth(points, iterations=3):
    """Chaikin corner cutting, keeping both end points."""
    pts = np.asarray(points, dtype=float)
    for _ in range(iterations):
        q = 0.75 * pts[:-1] + 0.25 * pts[1:]
        r = 0.25 * pts[:-1] + 0.75 * pts[1:]
        mid = np.empty((2 * len(q), 2))
        mid[0::2], mid[1::2] = q, r
        pts = np.vstack([pts[0], mid, pts[-1]])
    return pts


@dataclass
class PreparedPath:
    raw: np.ndarray        # waypoints as received, in metres (N, 2)
    s: np.ndarray          # arc length of each sample along the curve (M,)
    pts: np.ndarray        # sampled curve (M, 2)
    tangents: np.ndarray   # unit tangent at each sample (M, 2), computed once per path
    curvature: np.ndarray  # |curvature| at each sample (1/m), for slowing down in tight turns
    length: float          # total arc length (m)
    max_corner_cut: float  # largest distance from any waypoint to the curve (m)


def resample_polyline(points, max_spacing):
    """Insert points so no segment is longer than max_spacing; original points are kept."""
    out = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / max_spacing)))
        out.extend(a + (b - a) * (j / n) for j in range(1, n + 1))
    return np.array(out)


def prepare_path(waypoints, smoothing_iterations=3, degree=3, spacing=0.02, resample_spacing=0.5):
    """Smooth, fit and sample a path once, so the control loop only does array lookups.
    Returns None if the waypoints don't form a usable path.

    Chaikin cuts each corner by a fraction of the neighbouring segments, so long segments get
    big cuts. Resampling to at most `resample_spacing` first keeps the cut small (~0.1-0.2 m for
    a 90 degree corner at 0.5 m) regardless of how far apart the waypoints are."""
    raw = dedupe(waypoints)
    if len(raw) < 2 or not np.all(np.isfinite(raw)):
        return None

    if smoothing_iterations > 0:
        dense = resample_polyline(raw, resample_spacing) if resample_spacing > 0 else raw
        pts = dedupe(chaikin_smooth(dense, smoothing_iterations))
    else:
        pts = raw
    chord = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    if chord[-1] < 1e-6:
        return None

    k = int(min(max(degree, 1), len(pts) - 1))  # a cubic needs at least 4 points
    spl = make_interp_spline(chord, pts, k=k)

    u = np.linspace(0.0, chord[-1], max(2, int(math.ceil(chord[-1] / spacing)) + 1))
    curve = spl(u)
    d = spl.derivative(1)(u)
    tangents = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]

    heading = np.unwrap(np.arctan2(tangents[:, 1], tangents[:, 0]))
    ds = np.maximum(np.gradient(s), 1e-6)
    kappa = np.abs(np.gradient(heading) / ds)
    win = max(1, int(round(0.1 / spacing)))  # average over ~10 cm so a single sample can't spike it
    kappa = np.convolve(kappa, np.ones(win) / win, mode='same')

    diff = raw[:, None, :] - curve[None, :, :]
    max_cut = float(np.sqrt(np.min(np.einsum('nmk,nmk->nm', diff, diff), axis=1)).max())

    return PreparedPath(raw, s, curve, tangents, kappa, float(s[-1]), max_cut)


# ----------------------------------------------------------------------------
# Controller (no ROS dependencies)
# ----------------------------------------------------------------------------

@dataclass
class TrackerParams:
    nominal_velocity: float = 0.3                     # m/s, negative = drive in reverse
    k_heading: float = 1.0                            # omega = k_heading * steering error
    k_cross: float = 1.0                              # weight of the cross-track correction
    k_soft: float = 1.0                               # Stanley gain inside atan(k_soft * e / v)
    max_angular_velocity: float = 1.0                 # rad/s
    max_linear_accel: float = 0.5                     # m/s^2, for smooth starts and stops
    goal_tolerance: float = 0.15                      # m, done when this close to the end
    goal_pass_radius: float = 0.30                    # m, also done if it drives past the end within this
    slowdown_radius: float = 0.5                      # m, slow down over the last part of the path
    min_speed: float = 0.05                           # m/s, never creep slower than this while driving
    search_window: float = 1.0                        # m ahead of progress to look for the closest point
    search_back: float = 0.25                         # m behind progress to look for the closest point
    rotate_enter_angle: float = math.radians(60.0)    # turn in place above this steering error
    rotate_exit_angle: float = math.radians(15.0)     # resume driving below this
    min_rotate_speed: float = 0.3                     # rad/s, enough to overcome friction when turning in place
    rotate_delay: float = 0.15                        # s the error must stay above rotate_enter_angle first
    curvature_speed_factor: float = 0.7               # in turns, v <= factor * max_angular_velocity / curvature
    curvature_lookahead: float = 0.6                  # m ahead (plus 1 s of travel) to check for tight turns


def move_toward(current, target, max_step):
    return current + max(-max_step, min(max_step, target - current))


class PathTracker:
    """Stanley-style tracking for a differential-drive (unicycle) robot.

    The steering error combines heading error and cross-track error the way Stanley does:
        err = (path_heading - robot_heading) - k_cross * atan(k_soft * e / v)
    where e > 0 means the robot is left of the path. Then omega = k_heading * err.
    """

    def __init__(self, path: PreparedPath, params: TrackerParams, v0: float = 0.0):
        if params.nominal_velocity == 0.0:
            raise ValueError('nominal_velocity must be non-zero')
        self.path, self.p = path, params
        self.progress = 0.0     # arc length reached so far; only moves forward
        self.rotating = False
        self.rotate_dir = 1.0
        self.v_cmd = v0         # last commanded linear velocity, for the acceleration limit
        self.over_time = 0.0    # how long the error has been above rotate_enter_angle
        self.mode = 'tracking'  # tracking | rotating | to_goal | done

    def step(self, x, y, theta, dt):
        """Returns (v, omega, done)."""
        p, P = self.p, self.path
        reverse = p.nominal_velocity < 0
        speed = abs(p.nominal_velocity)
        # Driving backwards is driving forwards with the heading flipped; omega is unchanged.
        th = angle_wrap(theta + math.pi) if reverse else theta
        pos = np.array([x, y], dtype=float)

        # Closest sample, searched only around current progress, so a path that passes near
        # itself can't make the robot jump ahead or back.
        lo = int(np.searchsorted(P.s, self.progress - p.search_back, side='left'))
        hi = int(np.searchsorted(P.s, self.progress + p.search_window, side='right'))
        hi = max(hi, lo + 1)
        d = P.pts[lo:hi] - pos
        i = lo + int(np.argmin(np.einsum('ij,ij->i', d, d)))
        self.progress = max(self.progress, float(P.s[i]))

        goal = P.pts[-1]
        dist_goal = float(np.linalg.norm(goal - pos))
        if dist_goal < p.goal_tolerance:
            return self._finish()

        at_end = i == len(P.s) - 1
        if at_end:
            # The closest point is the end of the path, i.e. the robot is level with or past it.
            passed = float(np.dot(pos - goal, P.tangents[-1])) > 0.0
            if passed and dist_goal < p.goal_pass_radius:
                return self._finish()
            # Ended up off to the side: drive straight to the goal point instead of circling.
            err = angle_wrap(math.atan2(goal[1] - y, goal[0] - x) - th)
            self.mode = 'to_goal'
        else:
            t = P.tangents[i]
            e = float(np.dot(pos - P.pts[i], np.array([-t[1], t[0]])))  # > 0: robot is left of path
            heading_err = angle_wrap(math.atan2(t[1], t[0]) - th)
            err = angle_wrap(heading_err - p.k_cross * math.atan2(p.k_soft * e, speed + 1e-6))
            self.mode = 'tracking'

        # Turn in place when facing the wrong way. Hysteresis and a fixed turn direction stop it
        # flipping back and forth when the error is near +/-180 degrees.
        # A large error must persist for rotate_delay, so one noisy pose can't trigger it.
        self.over_time = self.over_time + dt if abs(err) > p.rotate_enter_angle else 0.0
        big_from_start = self.progress == 0.0 and self.v_cmd == 0.0 and abs(err) > p.rotate_enter_angle
        if not self.rotating and (self.over_time >= p.rotate_delay or big_from_start):
            self.rotating = True
            self.rotate_dir = 1.0 if err >= 0 else -1.0
        elif self.rotating and abs(err) < p.rotate_exit_angle:
            self.rotating = False
            self.over_time = 0.0

        if self.rotating:
            v_target = 0.0
            omega = self.rotate_dir * min(max(p.k_heading * abs(err), p.min_rotate_speed), p.max_angular_velocity)
            self.mode = 'rotating'
        else:
            remaining = max(P.length - self.progress, 0.0) if not at_end else dist_goal
            approach = min(1.0, max(remaining, dist_goal) / p.slowdown_radius) if p.slowdown_radius > 0 else 1.0
            v_target = speed * max(math.cos(err), 0.0) * approach
            # slow down ahead of tight turns so the turn rate limit can keep up
            j0 = i
            j1 = int(np.searchsorted(P.s, P.s[i] + p.curvature_lookahead + speed * 1.0, side='right'))
            kappa = float(P.curvature[j0:max(j1, j0 + 1)].max())
            if kappa > 1e-6:
                v_target = min(v_target, p.curvature_speed_factor * p.max_angular_velocity / kappa)
            v_target = max(v_target, p.min_speed)
            omega = min(max(p.k_heading * err, -p.max_angular_velocity), p.max_angular_velocity)

        if reverse:
            v_target = -v_target
        self.v_cmd = move_toward(self.v_cmd, v_target, p.max_linear_accel * dt)
        return self.v_cmd, omega, False

    def _finish(self):
        self.mode = 'done'
        self.v_cmd = 0.0
        return 0.0, 0.0, True


# ----------------------------------------------------------------------------
# ROS node
# ----------------------------------------------------------------------------

class PointController(Node):
    def __init__(self):
        super().__init__('point_controller')

        # topics and message types
        self.declare_parameter('points_topic', 'path_points')
        self.declare_parameter('points_msg_type', 'int32')        # int32 | float64 | path
        self.declare_parameter('points_scale', 1.0)                # metres per unit (e.g. A* cell size)
        self.declare_parameter('points_origin_x', 0.0)             # metres
        self.declare_parameter('points_origin_y', 0.0)
        self.declare_parameter('localization_topic', 'uwb_local/ekf')
        self.declare_parameter('pose_source', 'ekf_state')         # ekf_state | odometry
        self.declare_parameter('cmd_vel_topic', '/a200_1201/cmd_vel')
        self.declare_parameter('cmd_vel_type', 'auto')             # auto | twist | twist_stamped
        self.declare_parameter('frame_id', '')                     # expected frame of path and pose; '' = don't check
        self.declare_parameter('base_frame_id', 'base_link')       # header frame for TwistStamped
        # path shaping
        self.declare_parameter('smoothing_iterations', 3)          # 0 = pass through the waypoints exactly
        self.declare_parameter('spline_degree', 3)                 # 1 = straight lines between waypoints
        self.declare_parameter('resample_spacing', 0.5)            # m, bounds how much smoothing cuts corners
        self.declare_parameter('max_corner_cut', 0.3)              # m, warn if smoothing cuts a corner by more
        # timing and safety
        self.declare_parameter('control_rate', 50.0)               # Hz
        self.declare_parameter('pose_timeout', 0.5)                # s, stop if no pose for this long
        # controller gains and limits (see TrackerParams for meanings)
        defaults = TrackerParams()
        for f in fields(TrackerParams):
            value = getattr(defaults, f.name)
            if f.name in ('rotate_enter_angle', 'rotate_exit_angle'):
                self.declare_parameter(f.name + '_deg', math.degrees(value))
            else:
                self.declare_parameter(f.name, float(value))

        gp = lambda n: self.get_parameter(n).value
        self.points_type = str(gp('points_msg_type')).lower()
        self.points_scale = float(gp('points_scale'))
        self.points_origin = np.array([float(gp('points_origin_x')), float(gp('points_origin_y'))])
        self.pose_source = str(gp('pose_source')).lower()
        self.cmd_topic = str(gp('cmd_vel_topic'))
        self.cmd_type = str(gp('cmd_vel_type')).lower()
        self.frame_id = str(gp('frame_id'))
        self.base_frame_id = str(gp('base_frame_id'))
        self.smoothing_iterations = int(gp('smoothing_iterations'))
        self.spline_degree = int(gp('spline_degree'))
        self.resample_spacing = float(gp('resample_spacing'))
        self.max_corner_cut = float(gp('max_corner_cut'))
        self.pose_timeout = float(gp('pose_timeout'))
        rate = float(gp('control_rate'))

        kw = {}
        for f in fields(TrackerParams):
            if f.name in ('rotate_enter_angle', 'rotate_exit_angle'):
                kw[f.name] = math.radians(float(gp(f.name + '_deg')))
            else:
                kw[f.name] = float(gp(f.name))
        self.params = TrackerParams(**kw)
        self._validate(rate)

        # state
        self.pose = None
        self.pose_time = None
        self.pose_stale = False
        self.tracker = None
        self.active_key = None
        self.cmd_pub = None
        self.last_tick = None
        self._warned = set()

        # localization
        if self.pose_source == 'odometry':
            self.create_subscription(Odometry, str(gp('localization_topic')), self._on_odometry, 10)
        elif self.pose_source == 'ekf_state':
            try:
                from husky_interfaces.msg import EkfState  # only needed in this mode
            except ImportError as exc:
                raise RuntimeError("pose_source is 'ekf_state' but husky_interfaces isn't found. Source the "
                                   "workspace that builds it, or use pose_source:=odometry.") from exc
            self.create_subscription(EkfState, str(gp('localization_topic')), self._on_ekf_state, 10)
        else:
            raise ValueError(f"pose_source must be 'ekf_state' or 'odometry', got '{self.pose_source}'")

        # path: one latched and one normal subscription, so either kind of publisher works.
        # (A transient_local subscription can't connect to a volatile publisher, and a volatile
        # one never receives a message sent before it started.) Duplicates are ignored in _on_points.
        msg_type = {'int32': Int32MultiArray, 'float64': Float64MultiArray, 'path': Path}.get(self.points_type)
        if msg_type is None:
            raise ValueError(f"points_msg_type must be int32, float64 or path, got '{self.points_type}'")
        points_topic = str(gp('points_topic'))
        latched = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                             reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        normal = QoSProfile(depth=10, history=HistoryPolicy.KEEP_LAST,
                            reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
        quiet = SubscriptionEventCallbacks(incompatible_qos=lambda event: None)  # expected for one of the two
        self.create_subscription(msg_type, points_topic, self._on_points, latched, event_callbacks=quiet)
        self.create_subscription(msg_type, points_topic, self._on_points, normal, event_callbacks=quiet)

        # outputs
        if self.cmd_type in ('twist', 'twist_stamped'):
            self._make_cmd_publisher(self.cmd_type)
        elif self.cmd_type != 'auto':
            raise ValueError(f"cmd_vel_type must be auto, twist or twist_stamped, got '{self.cmd_type}'")
        self.path_pub = self.create_publisher(
            Path, '~/smoothed_path',
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))

        self.timer = self.create_timer(1.0 / rate, self._on_timer)
        self.create_timer(2.0, self._check_other_publishers)
        self.get_logger().info(
            f"listening for paths on '{points_topic}' ({self.points_type}), pose on "
            f"'{gp('localization_topic')}' ({self.pose_source}), commanding '{self.cmd_topic}'")

    # --- setup helpers -------------------------------------------------------

    def _validate(self, rate):
        p = self.params
        problems = []
        if p.nominal_velocity == 0.0: problems.append('nominal_velocity must be non-zero')
        if p.goal_tolerance <= 0.0: problems.append('goal_tolerance must be > 0')
        if p.goal_pass_radius < p.goal_tolerance: problems.append('goal_pass_radius must be >= goal_tolerance')
        if p.max_angular_velocity <= 0.0: problems.append('max_angular_velocity must be > 0')
        if p.max_linear_accel <= 0.0: problems.append('max_linear_accel must be > 0')
        if p.search_window <= 0.0: problems.append('search_window must be > 0')
        if p.rotate_exit_angle >= p.rotate_enter_angle: problems.append('rotate_exit_angle must be < rotate_enter_angle')
        if rate <= 0.0: problems.append('control_rate must be > 0')
        if self.pose_timeout <= 0.0: problems.append('pose_timeout must be > 0')
        if self.points_scale <= 0.0: problems.append('points_scale must be > 0')
        if problems:
            raise ValueError('invalid parameters: ' + '; '.join(problems))

    def _make_cmd_publisher(self, kind):
        self.cmd_stamped = kind == 'twist_stamped'
        self.cmd_pub = self.create_publisher(TwistStamped if self.cmd_stamped else Twist, self.cmd_topic, 10)
        self.get_logger().info(f"publishing {'TwistStamped' if self.cmd_stamped else 'Twist'} on '{self.cmd_topic}'")

    def _resolve_cmd_type(self):
        """For cmd_vel_type=auto: match whatever the robot's driver subscribes with.
        ROS 2 Jazzy Clearpath software uses TwistStamped; Humble uses Twist."""
        subs = self.get_subscriptions_info_by_topic(self.cmd_topic)
        types = {s.topic_type for s in subs}
        if 'geometry_msgs/msg/TwistStamped' in types:
            self._make_cmd_publisher('twist_stamped')
        elif 'geometry_msgs/msg/Twist' in types:
            self._make_cmd_publisher('twist')
        else:
            self._warn_once('no_cmd_sub',
                            f"nothing is subscribed to '{self.cmd_topic}' yet, so the robot can't move. "
                            "Check cmd_vel_topic and that the robot driver is running.")
            return False
        return True

    def _check_other_publishers(self):
        """Two nodes commanding the same cmd_vel fight each other (e.g. a second copy of this node
        left running in another terminal). Warn about it."""
        if self.cmd_pub is None:
            return
        me = (self.get_namespace().rstrip('/'), self.get_name())
        others = sorted({f"{i.node_namespace.rstrip('/')}/{i.node_name}"
                         for i in self.get_publishers_info_by_topic(self.cmd_topic)
                         if (i.node_namespace.rstrip('/'), i.node_name) != me})
        key = 'others:' + ','.join(others)
        if others and key not in self._warned:
            self._warned.add(key)
            self.get_logger().warn(f"other nodes also publish to '{self.cmd_topic}': {', '.join(others)}. "
                                   'If one is another controller, the commands will fight.')

    def _warn_once(self, key, text, level='warn'):
        if key not in self._warned:
            self._warned.add(key)
            getattr(self.get_logger(), level)(text)

    def _check_frame(self, what, frame):
        if self.frame_id and frame and frame != self.frame_id:
            self._warn_once(f'frame_{what}', f"{what} is in frame '{frame}' but frame_id is '{self.frame_id}'. "
                                             "The path and the pose must be in the same frame.")

    # --- callbacks -----------------------------------------------------------

    def _set_pose(self, x, y, theta):
        if not all(math.isfinite(v) for v in (x, y, theta)):
            self._warn_once('nan_pose', 'received a non-finite pose (NaN/inf); ignoring it')
            return
        self.pose = (x, y, theta)
        self.pose_time = self.get_clock().now()

    def _on_ekf_state(self, msg):
        try:
            d = msg.data
            x, y, theta = float(d.position_m.x), float(d.position_m.y), float(d.theta_rad)
        except AttributeError:
            self._warn_once('ekf_fields',
                            'EkfState does not have msg.data.position_m.x/y and msg.data.theta_rad. Its fields are '
                            f'{msg.get_fields_and_field_types()}. Edit _on_ekf_state to match.', level='error')
            return
        self._set_pose(x, y, theta)

    def _on_odometry(self, msg):
        self._check_frame('pose', msg.header.frame_id)
        p = msg.pose.pose
        self._set_pose(float(p.position.x), float(p.position.y), yaw_from_quaternion(p.orientation))

    def _on_points(self, msg):
        if self.points_type == 'path':
            self._check_frame('path', msg.header.frame_id)
            pts = np.array([[ps.pose.position.x, ps.pose.position.y] for ps in msg.poses], dtype=float).reshape(-1, 2)
        else:
            data = np.asarray(msg.data, dtype=float)
            if len(data) % 2:
                self.get_logger().warn(f'odd-length points array ({len(data)} values); dropping the last value')
                data = data[:-1]
            pts = data.reshape(-1, 2) * self.points_scale + self.points_origin

        key = pts.round(6).tobytes()
        if self.tracker is not None and key == self.active_key:
            return  # same path again (or the second copy from the other subscription)

        path = prepare_path(pts, self.smoothing_iterations, self.spline_degree,
                            resample_spacing=self.resample_spacing)
        if path is None:
            self.get_logger().warn(f'ignoring path: need at least 2 distinct, finite points (got {len(pts)})')
            return

        v0 = self.tracker.v_cmd if self.tracker is not None else 0.0
        self.tracker = PathTracker(path, self.params, v0=v0)
        self.active_key = key
        self.get_logger().info(f'new path: {len(path.raw)} waypoints, {path.length:.2f} m, '
                               f'max corner cut {path.max_corner_cut:.2f} m')
        if path.max_corner_cut > self.max_corner_cut:
            self.get_logger().warn(
                f'smoothing moves the path up to {path.max_corner_cut:.2f} m away from a waypoint. If that '
                'clips an obstacle, lower resample_spacing or smoothing_iterations (0 = through the waypoints).')
        self._publish_path(path)

    def _on_timer(self):
        now = self.get_clock().now()
        dt = 0.0 if self.last_tick is None else (now - self.last_tick).nanoseconds * 1e-9
        dt = min(max(dt, 0.0), 0.2)
        self.last_tick = now

        if self.tracker is None:
            return

        if self.pose_time is None or (now - self.pose_time).nanoseconds * 1e-9 > self.pose_timeout:
            if not self.pose_stale:
                self.pose_stale = True
                self.get_logger().warn(f'no pose for over {self.pose_timeout:.2f} s; stopping until it comes back')
                self.stop_robot()
            return
        if self.pose_stale:
            self.pose_stale = False
            self.get_logger().info('pose is back; resuming the path')

        v, omega, done = self.tracker.step(*self.pose, dt)
        self._publish_cmd(v, omega)
        if done:
            self.get_logger().info('reached the end of the path')
            self.tracker = None
            self.active_key = None

    # --- output --------------------------------------------------------------

    def _publish_cmd(self, v, omega):
        if self.cmd_pub is None and not self._resolve_cmd_type():
            return
        twist = Twist()
        twist.linear.x = float(v)
        twist.angular.z = float(omega)
        if self.cmd_stamped:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.base_frame_id
            msg.twist = twist
            self.cmd_pub.publish(msg)
        else:
            self.cmd_pub.publish(twist)

    def stop_robot(self, repeat=1):
        if self.tracker is not None:
            self.tracker.v_cmd = 0.0
        for _ in range(repeat):
            self._publish_cmd(0.0, 0.0)

    def _publish_path(self, path):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id or 'map'
        step = max(1, len(path.pts) // 500)  # RViz doesn't need every 2 cm sample
        for x, y in path.pts[::step].tolist() + [path.pts[-1].tolist()]:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x, ps.pose.position.y = float(x), float(y)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.path_pub.publish(msg)


def _sigterm_to_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


def main(args=None):
    # Handle Ctrl-C ourselves so the context is still valid when we send the final stop command.
    try:
        from rclpy.signals import SignalHandlerOptions
        rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    except ImportError:
        rclpy.init(args=args)
    signal.signal(signal.SIGTERM, _sigterm_to_keyboard_interrupt)  # ros2 launch escalates to SIGTERM

    node = None
    try:
        node = PointController()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                node.get_logger().info('shutting down; sending stop command')
                node.stop_robot(repeat=3)
                time.sleep(0.1)  # give DDS a moment to deliver it
            except Exception as exc:  # never let cleanup hide the original error
                print(f'point_controller: could not send stop command: {exc}')
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
