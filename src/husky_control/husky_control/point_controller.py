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
import rclpy
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
        self.declare_parameter('use_stamped', True)
        self.declare_parameter('localization_topic', 'uwb_local/ekf')
        self.declare_parameter('nominal_velocity', 0.3)
        self.declare_parameter('k_heading', 1.0)
        self.declare_parameter('k_cross', 1.0)
        self.declare_parameter('max_omega', 1.0)
        self.declare_parameter('goal_tolerance', 0.15)

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
        data = np.asarray(msg.data, dtype=float)
        if len(data) < 2 or len(data) % 2:
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
        self.state = (msg.position_m.x, msg.position_m.y, msg.theta_rad)

    def control_publish_callback(self):
        if not self.follower.active:
            return
        if self.state is None:
            self.get_logger().warn(
                f'have a path but no pose on {self.localization_topic}',
                throttle_duration_sec=2.0)
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
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
