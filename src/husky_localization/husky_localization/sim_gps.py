#!/usr/bin/env python3
'''
Simulation only: fake GPS on the '86 Field.

  out: gps_fix_topic  sensor_msgs/NavSatFix            (default /gps/fix)
       heading_topic  geometry_msgs/QuaternionStamped  (default /heading)

Gazebo has no GPS on the robot, so this publishes the same messages the real
receiver will, which lets the whole RL pipeline run in simulation:
  discretizer (cells) -> rl_policy -> astar_node (path in metres)
  gps_pose (EkfState) -> point_controller -> cmd_vel -> Gazebo

pose_source:
  'gazebo' (default) -- the robot's TRUE pose, read from Gazebo's
      /world/<gz_world>/dynamic_pose/info (entity <gz_model>).  Like a real
      RTK GPS, it does not drift.  The world rpi_86_field_1.sdf is built in
      the cell-table frame (origin Star_1, +x east, +y north), so Gazebo world
      metres are table metres and no spawn pose is needed.
  'odom' -- simulator odometry placed at spawn_x / spawn_y / spawn_yaw.
      Wheel odometry of a skid-steer Husky gets turns wrong (wheel slip),
      so the pipeline's idea of the heading drifts away from the real one.

The heading is published as a compass angle (0 = north, clockwise) packed
into the quaternion's yaw, matching gps_pose's default
heading_convention=cw_from_north.

Do NOT run this on the real robot; it would fight the real GPS on /gps/fix.
'''

import math
import random
import threading

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import QuaternionStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix, NavSatStatus

from husky_rl.cell_table import CellTable, default_table_path

from husky_localization.gps_pose_math import (
    enu_yaw_to_compass, odom_to_local, yaw_from_quaternion)


class SimGps(Node):
    def __init__(self):
        super().__init__('sim_gps')
        self.declare_parameter('cell_table', '')
        self.declare_parameter('pose_source', 'gazebo')
        self.declare_parameter('gz_world', 'rpi_86_field_1')
        self.declare_parameter('gz_model', 'a200_0000/robot')
        self.declare_parameter('odom_topic', '/a200_1201/platform/odom/filtered')
        # odom mode only: Gazebo spawn pose in world (= cell-table) metres / ENU rad
        self.declare_parameter('spawn_x', 0.0)
        self.declare_parameter('spawn_y', 0.0)
        self.declare_parameter('spawn_yaw', 0.0)
        self.declare_parameter('gps_fix_topic', '/gps/fix')
        self.declare_parameter('heading_topic', '/heading')
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('noise_std_m', 0.0)
        self.declare_parameter('heading_noise_std_rad', 0.0)
        self.declare_parameter('fix_status', NavSatStatus.STATUS_GBAS_FIX)

        def p(name):
            return self.get_parameter(name).value

        table = CellTable.load(str(p('cell_table')) or default_table_path())
        self.table = table
        self.frame = table.frame
        self.noise = float(p('noise_std_m'))
        self.heading_noise = float(p('heading_noise_std_rad'))
        self.status = int(p('fix_status'))
        self.source = str(p('pose_source'))

        self.fix_pub = self.create_publisher(NavSatFix, str(p('gps_fix_topic')), 10)
        self.heading_pub = self.create_publisher(QuaternionStamped, str(p('heading_topic')), 10)

        self._lock = threading.Lock()
        self._pose = None      # latest (east, north, yaw) in the local frame
        self.count = 0
        self._logged_start = False

        if self.source == 'gazebo':
            self._start_gazebo(str(p('gz_world')), str(p('gz_model')))
        elif self.source == 'odom':
            self.start_e = float(p('spawn_x'))
            self.start_n = float(p('spawn_y'))
            self.start_yaw = float(p('spawn_yaw'))
            odom_topic = str(p('odom_topic'))
            self.create_subscription(Odometry, odom_topic, self.odom_callback,
                                     qos_profile_sensor_data)
            self.source_desc = (
                f'{odom_topic} placed at spawn ({self.start_e:.2f}, {self.start_n:.2f}) m, '
                f'yaw {math.degrees(self.start_yaw):.1f} deg (must match the Gazebo spawn; '
                'wheel odometry drifts on turns)')
        else:
            raise ValueError(f"pose_source must be 'gazebo' or 'odom', got {self.source!r}")

        self.create_timer(1.0 / float(p('rate_hz')), self.publish_fix)
        self.create_timer(5.0, self.check_alive)
        self.get_logger().info(f'fake GPS from {self.source_desc}')

    # ── pose sources ─────────────────────────────────────────────────────────

    def _start_gazebo(self, world, model):
        # Imported here so the odom mode works without Gazebo's Python bindings.
        from gz.msgs10.pose_v_pb2 import Pose_V
        from gz.transport13 import Node as GzNode

        self.gz_model = model
        self._gz_names = set()
        topic = f'/world/{world}/dynamic_pose/info'

        def callback(msg):  # runs in a Gazebo transport thread
            for pose in msg.pose:
                if pose.name == model:
                    q = pose.orientation
                    with self._lock:
                        self._pose = (pose.position.x, pose.position.y,
                                      yaw_from_quaternion(q.x, q.y, q.z, q.w))
                    return
            with self._lock:
                self._gz_names.update(p.name for p in msg.pose)

        self._gz_node = GzNode()  # keep a reference or the subscription ends
        if not self._gz_node.subscribe(Pose_V, topic, callback):
            raise RuntimeError(f'could not subscribe to Gazebo topic {topic}')
        self.source_desc = f'Gazebo true pose of "{model}" on {topic}'

    def odom_callback(self, msg: Odometry):
        pose = msg.pose.pose
        q = pose.orientation
        local = odom_to_local(
            pose.position.x, pose.position.y, yaw_from_quaternion(q.x, q.y, q.z, q.w),
            self.start_e, self.start_n, self.start_yaw)
        with self._lock:
            self._pose = local

    # ── output ───────────────────────────────────────────────────────────────

    def publish_fix(self):
        with self._lock:
            pose = self._pose
        if pose is None:
            return
        east, north, yaw = pose
        if not self._logged_start:
            cell, dist = self.table.nearest_cell(east, north)
            self.get_logger().info(
                f'robot at ({east:.2f}, {north:.2f}) m, yaw {math.degrees(yaw):.1f} deg; '
                f'nearest cell {cell} ({dist:.2f} m)')
            self._logged_start = True
        self.count += 1
        if self.noise > 0.0:
            east += random.gauss(0.0, self.noise)
            north += random.gauss(0.0, self.noise)
        if self.heading_noise > 0.0:
            yaw += random.gauss(0.0, self.heading_noise)
        lat, lon = self.frame.to_latlon(east, north)

        fix = NavSatFix()
        fix.header.stamp = self.get_clock().now().to_msg()
        fix.header.frame_id = 'gps_link'
        fix.status.status = self.status
        fix.status.service = NavSatStatus.SERVICE_GPS
        fix.latitude, fix.longitude, fix.altitude = lat, lon, 0.0
        var = max(self.noise, 0.01) ** 2
        fix.position_covariance = [var, 0.0, 0.0, 0.0, var, 0.0, 0.0, 0.0, 4 * var]
        fix.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
        self.fix_pub.publish(fix)

        heading = QuaternionStamped()
        heading.header = fix.header
        compass = enu_yaw_to_compass(yaw)
        heading.quaternion.z = math.sin(compass / 2.0)
        heading.quaternion.w = math.cos(compass / 2.0)
        self.heading_pub.publish(heading)

    def check_alive(self):
        if self.count > 0:
            return
        if self.source == 'gazebo':
            with self._lock:
                names = sorted(self._gz_names)
            hint = (f'; models seen: {names[:10]} (set gz_model)' if names
                    else '; is Gazebo running with this world (gz_world)?')
            self.get_logger().warn(f'no Gazebo pose for "{self.gz_model}" yet{hint}')
        else:
            self.get_logger().warn(
                'no odometry received yet; check odom_topic with '
                '"ros2 topic list | grep odom"')


def main():
    rclpy.init()
    node = SimGps()
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
