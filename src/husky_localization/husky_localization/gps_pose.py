#!/usr/bin/env python3
'''
GPS -> husky_interfaces/EkfState in the '86 Field local frame.

  in : gps_fix_topic  sensor_msgs/NavSatFix            (default /gps/fix)
       heading_topic  geometry_msgs/QuaternionStamped  (default /heading)
  out: pose_topic     husky_interfaces/EkfState        (default /gps/local_pose)

Position is converted with husky_rl.cell_table (same frame and formula as
config/field86_cells.yaml: x = east, y = north, metres, origin Star_1), so the
A* path from astar_node and this pose share one frame and point_controller
needs no TF.  theta_rad is ENU yaw (0 = east, counter-clockwise).

Nothing is published until a heading has arrived within heading_timeout_s,
because the controller needs theta.

VERIFY ON HARDWARE: heading_convention and heading_offset_rad depend on the
driver and on how the two antennas are mounted.
'''

import math

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import QuaternionStamped
from sensor_msgs.msg import NavSatFix
from husky_interfaces.msg import EkfState

from husky_rl.cell_table import CellTable, default_table_path, fix_accepted

from husky_localization.gps_pose_math import (
    antenna_to_base, heading_to_enu_yaw, yaw_from_quaternion)


class GpsPose(Node):
    def __init__(self):
        super().__init__('gps_pose')
        self.declare_parameter('cell_table', '')
        self.declare_parameter('gps_fix_topic', '/gps/fix')
        self.declare_parameter('heading_topic', '/heading')
        self.declare_parameter('pose_topic', '/gps/local_pose')
        self.declare_parameter('heading_convention', 'cw_from_north')
        self.declare_parameter('heading_offset_rad', 0.0)
        self.declare_parameter('heading_timeout_s', 1.0)
        self.declare_parameter('antenna_offset_x_m', 0.0)
        self.declare_parameter('antenna_offset_y_m', 0.0)
        self.declare_parameter('require_rtk', False)

        def p(name):
            return self.get_parameter(name).value

        table_path = str(p('cell_table')) or default_table_path()
        self.frame = CellTable.load(table_path).frame
        self.convention = str(p('heading_convention'))
        self.heading_offset = float(p('heading_offset_rad'))
        self.heading_timeout = float(p('heading_timeout_s'))
        self.ant_x = float(p('antenna_offset_x_m'))
        self.ant_y = float(p('antenna_offset_y_m'))
        self.require_rtk = bool(p('require_rtk'))
        heading_to_enu_yaw(0.0, self.convention)  # validate the parameter early

        self.yaw = None
        self.yaw_time = None
        fix_topic = str(p('gps_fix_topic'))
        heading_topic = str(p('heading_topic'))
        pose_topic = str(p('pose_topic'))
        self.pub = self.create_publisher(EkfState, pose_topic, 10)
        self.create_subscription(
            NavSatFix, fix_topic, self.fix_callback, qos_profile_sensor_data)
        self.create_subscription(
            QuaternionStamped, heading_topic, self.heading_callback,
            qos_profile_sensor_data)
        self.get_logger().info(
            f'{fix_topic} + {heading_topic} -> {pose_topic}; '
            f'origin ({self.frame.lat0}, {self.frame.lon0}), '
            f'heading_convention={self.convention}')

    def heading_callback(self, msg: QuaternionStamped):
        q = msg.quaternion
        raw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        self.yaw = heading_to_enu_yaw(raw, self.convention, self.heading_offset)
        self.yaw_time = self.get_clock().now()

    def fix_callback(self, msg: NavSatFix):
        ok, reason = fix_accepted(
            msg.status.status, msg.latitude, msg.longitude, require_rtk=self.require_rtk)
        if not ok:
            self.get_logger().warn(f'fix rejected: {reason}', throttle_duration_sec=2.0)
            return
        if self.yaw is None or (
                (self.get_clock().now() - self.yaw_time).nanoseconds * 1e-9
                > self.heading_timeout):
            self.get_logger().warn('no recent heading; pose not published',
                                   throttle_duration_sec=2.0)
            return

        east, north = self.frame.to_local(msg.latitude, msg.longitude)
        east, north = antenna_to_base(east, north, self.yaw, self.ant_x, self.ant_y)

        out = EkfState()
        out.timestamp = msg.header.stamp
        out.position_m.x = east
        out.position_m.y = north
        out.position_m.z = msg.altitude if math.isfinite(msg.altitude) else 0.0
        out.theta_rad = self.yaw
        cov = msg.position_covariance  # ENU 3x3; keep x/y, heading variance unknown
        out.covariance = [float(cov[0]), float(cov[1]), 0.0,
                          float(cov[3]), float(cov[4]), 0.0,
                          0.0, 0.0, 0.0]
        self.pub.publish(out)


def main():
    rclpy.init()
    node = GpsPose()
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
