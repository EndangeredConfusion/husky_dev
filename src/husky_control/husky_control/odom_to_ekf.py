#!/usr/bin/env python3
'''
Simulation bridge: nav_msgs/Odometry  ->  husky_interfaces/EkfState

Gazebo has no UWB anchors, so the real EKF never publishes. This node
republishes the simulator's odometry in the EkfState format that
point_controller (and the rest of the pipeline) already listens to.

  in : odom_topic          (default /a200_1201/platform/odom/filtered)
  out: ekf_topic           (default uwb_local/ekf)

Find the right odom topic with:  ros2 topic list | grep odom
Do NOT run this together with the real ekf node; both publish ekf_topic.
'''

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from nav_msgs.msg import Odometry
from husky_interfaces.msg import EkfState

from husky_control.path_follower import yaw_from_quaternion

# rows/cols of x, y, yaw in the 6x6 Odometry pose covariance
_IDX = (0, 1, 5)


class OdomToEkf(Node):
    def __init__(self):
        super().__init__('odom_to_ekf')
        self.declare_parameter('odom_topic', '/a200_1201/platform/odom/filtered')
        self.declare_parameter('ekf_topic', 'uwb_local/ekf')
        odom_topic = str(self.get_parameter('odom_topic').value)
        ekf_topic = str(self.get_parameter('ekf_topic').value)

        self.pub = self.create_publisher(EkfState, ekf_topic, 10)
        # sensor-data QoS (best effort) also matches reliable publishers
        self.sub = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, qos_profile_sensor_data)
        self.count = 0
        self.get_logger().info(f'{odom_topic} -> {ekf_topic}')
        self.create_timer(5.0, self.check_alive)

    def odom_callback(self, msg: Odometry):
        pose = msg.pose.pose
        out = EkfState()
        out.timestamp = msg.header.stamp
        out.position_m.x = pose.position.x
        out.position_m.y = pose.position.y
        out.position_m.z = pose.position.z
        q = pose.orientation
        out.theta_rad = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        cov = msg.pose.covariance
        out.covariance = [float(cov[6 * r + c]) for r in _IDX for c in _IDX]
        self.pub.publish(out)
        self.count += 1

    def check_alive(self):
        if self.count == 0:
            self.get_logger().warn(
                'no odometry received yet; check odom_topic with '
                '"ros2 topic list | grep odom"')


def main():
    rclpy.init()
    node = OdomToEkf()
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
