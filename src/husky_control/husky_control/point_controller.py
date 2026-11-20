#!/usr/bin/env python3
'''
expected: Float64MultiArray [x0, y0, x1, y1, ...] in metres on 'points_topic'
          (same frame as the localization topic)
output:   velocity command on 'cmd_vel_topic' that drives the robot along a
          smoothed spline through the points, then stops at the last point
notes:
- publish each path ONCE; a new message replaces the current path.
- a single point [x, y] means "go straight to (x, y)".
- the control math lives in path_follower.py (tested offline, no ROS).
- Gazebo / Jazzy Clearpath expects TwistStamped (use_stamped:=true).
  Confirm with: ros2 topic info /a200_1201/cmd_vel -v
'''

import numpy as np
import time
import math
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from geometry_msgs.msg import Twist, TwistStamped
from std_msgs.msg import Float64MultiArray

from husky_interfaces.msg import EkfState

from husky_control.path_follower import StanleyFollower


class PointController(Node):
    def __init__(self):
        super().__init__('point_controller')

        # husky_rl/launch/rl.launch.py sets this to /rl_path
        self.declare_parameter('points_topic', '/rl_path')
        self.declare_parameter('cmd_vel_topic', '/a200_1201/cmd_vel')
        self.declare_parameter('use_stamped', True)  # Gazebo / Jazzy Clearpath uses TwistStamped
        self.declare_parameter('localization_topic', 'uwb_local/ekf')
        self.declare_parameter('nominal_velocity', 0.3)
        self.declare_parameter('k_heading', 1.0)
        self.declare_parameter('k_cross', 1.0)
        self.declare_parameter('max_omega', 1.0)
        self.declare_parameter('goal_tolerance', 0.15)
        self.declare_parameter('pose_timeout_s', 1.0)

        def p(name):
            return self.get_parameter(name).value

        self.points_topic = str(p('points_topic'))
        self.cmd_topic = str(p('cmd_vel_topic'))
        self.use_stamped = bool(p('use_stamped'))
        self.localization_topic = str(p('localization_topic'))

        self.follower = StanleyFollower(
            nominal_velocity=float(p('nominal_velocity')),
            k_heading=float(p('k_heading')),
            k_cross=float(p('k_cross')),
            max_omega=float(p('max_omega')),
            goal_tolerance=float(p('goal_tolerance')))

        self.state = None
        self.pose_received_at = None
        self.pose_stamp = None
        self.pose_timeout_s = float(p('pose_timeout_s'))
        if not math.isfinite(self.pose_timeout_s) or self.pose_timeout_s <= 0:
            raise ValueError('pose_timeout_s must be positive and finite')
        self.localization_subscriber = self.create_subscription(
            EkfState, self.localization_topic, self.localization_callback, 10)
        self.points_subscriber = self.create_subscription(
            Float64MultiArray, self.points_topic, self.point_callback, 10)

        msg_type = TwistStamped if self.use_stamped else Twist
        self.control_publisher = self.create_publisher(msg_type, self.cmd_topic, 10)
        self.timer = self.create_timer(0.02, self.control_publish_callback)  # 50 Hz

        self.get_logger().info(
            f'path: {self.points_topic}  pose: {self.localization_topic}  '
            f'cmd: {self.cmd_topic} ({msg_type.__name__})')

    def point_callback(self, msg: Float64MultiArray):
        ''' replaces the current path with the published waypoints '''
        if not self.pose_fresh():
            self.stop_path('path rejected: no fresh pose')
            return
        data = np.asarray(msg.data, dtype=float)
        if len(data) < 2 or len(data) % 2 or not np.isfinite(data).all():
            self.stop_path('invalid path')
            self.get_logger().error(
                f'path needs an even number of values (x0,y0,x1,y1,...), got {len(data)}')
            return
        points = data.reshape(-1, 2)
        current_xy = None if self.state is None else self.state[:2]
        if self.follower.set_path(points, current_xy=current_xy):
            self.get_logger().info(
                f'new path: {len(points)} waypoints, goal {points[-1].tolist()}')
        else:
            self.get_logger().error(
                'path rejected: need 2+ distinct waypoints '
                '(or 1 waypoint after a pose has been received)')

    def localization_callback(self, msg: EkfState):
        ''' updates internal state from the localization topic '''
        state = (msg.position_m.x, msg.position_m.y, msg.theta_rad)
        stamp = msg.timestamp.sec + msg.timestamp.nanosec * 1e-9
        age = self.get_clock().now().nanoseconds * 1e-9 - stamp
        if not all(math.isfinite(value) for value in state) or not 0 <= age <= self.pose_timeout_s:
            self.state = None
            self.pose_received_at = None
            self.stop_path('invalid pose')
            return
        self.state = state
        self.pose_stamp = stamp
        self.pose_received_at = time.monotonic()

    def pose_fresh(self):
        return (self.state is not None and self.pose_received_at is not None
                and self.pose_stamp is not None
                and time.monotonic() - self.pose_received_at <= self.pose_timeout_s
                and 0 <= self.get_clock().now().nanoseconds * 1e-9 - self.pose_stamp <= self.pose_timeout_s)

    def stop_path(self, reason):
        was_active = self.follower.active
        self.follower.clear()
        if was_active:
            self.publish_cmd(0.0, 0.0)
            self.get_logger().warning(reason + '; stopped, a new path is required')

    def control_publish_callback(self):
        if not self.follower.active:
            return
        if not self.pose_fresh():
            self.stop_path('pose timed out')
            return

        v, omega, done = self.follower.step(*self.state)
        self.publish_cmd(v, omega)
        if done:
            self.get_logger().info('goal reached, stopping')

    def publish_cmd(self, v, omega):
        if self.use_stamped:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_link'
            twist = msg.twist
        else:
            msg = Twist()
            twist = msg
        twist.linear.x = float(v)
        twist.angular.z = float(omega)
        self.control_publisher.publish(msg)


def main():
    rclpy.init()
    node = PointController()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
