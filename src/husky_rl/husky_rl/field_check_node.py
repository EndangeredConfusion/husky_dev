#!/usr/bin/env python3
"""Field walk-through check for the '86 Field cell table.

Prints (every ``print_period_s``) fix quality, local position, nearest cell
and its distance, the last reached cell (same rule as the discretizer), the
distance to the walkways and the raw heading, and logs every fix to a CSV.
Read-only: publishes nothing, so it is safe to run with the motors off.

    ros2 launch husky_rl field_check.launch.py
"""

from __future__ import annotations

import csv
import datetime
import math
import os

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import QuaternionStamped
from sensor_msgs.msg import NavSatFix

from husky_rl.cell_table import CellTable, CellTracker, default_table_path, fix_accepted

STATUS_NAMES = {-1: 'NO_FIX', 0: 'FIX', 1: 'SBAS', 2: 'GBAS/RTK'}

CSV_FIELDS = [
    'stamp_s', 'lat', 'lon', 'alt', 'status', 'accepted', 'std_east_m', 'std_north_m',
    'east_m', 'north_m', 'nearest_cell', 'nearest_dist_m', 'reached_cell',
    'walkway_dist_m', 'off_walkway', 'raw_heading_deg',
]


def _cell_str(cell) -> str:
    return '' if cell is None else f'{cell[0]},{cell[1]}'


class FieldCheckNode(Node):
    """Print and log GPS fixes against the cell table."""

    def __init__(self) -> None:
        super().__init__('field_check')
        self.declare_parameter('cell_table', '')
        self.declare_parameter('gps_fix_topic', '/gps/fix')
        self.declare_parameter('heading_topic', '/heading')
        self.declare_parameter('csv_path', '')
        self.declare_parameter('print_period_s', 1.0)
        self.declare_parameter('reached_radius_m', 1.5)
        self.declare_parameter('off_walkway_m', 6.0)
        self.declare_parameter('bootstrap_nearest', True)

        def p(name):
            return self.get_parameter(name).value

        table_path = str(p('cell_table')) or default_table_path()
        self.table = CellTable.load(table_path)
        self.tracker = CellTracker(
            self.table, float(p('reached_radius_m')), float(p('off_walkway_m')),
            bool(p('bootstrap_nearest')))

        csv_path = str(p('csv_path'))
        if not csv_path:
            stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_path = os.path.expanduser(f'~/field_check_{stamp}.csv')
        self.csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        self.csv = csv.DictWriter(self.csv_file, fieldnames=CSV_FIELDS)
        self.csv.writeheader()

        self.raw_heading_deg = math.nan
        self.latest = None
        self.count = 0
        fix_topic = str(p('gps_fix_topic'))
        heading_topic = str(p('heading_topic'))
        self.create_subscription(
            NavSatFix, fix_topic, self.fix_callback, qos_profile_sensor_data)
        self.create_subscription(
            QuaternionStamped, heading_topic, self.heading_callback,
            qos_profile_sensor_data)
        self.create_timer(float(p('print_period_s')), self.print_status)
        self.get_logger().info(
            f'field check: {fix_topic} (+ {heading_topic}), table {table_path}, '
            f'logging to {csv_path}')

    def heading_callback(self, msg: QuaternionStamped) -> None:
        q = msg.quaternion
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.raw_heading_deg = math.degrees(yaw)

    def fix_callback(self, msg: NavSatFix) -> None:
        self.count += 1
        ok, reason = fix_accepted(msg.status.status, msg.latitude, msg.longitude)
        cov = msg.position_covariance
        row = {
            'stamp_s': f'{msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9:.3f}',
            'lat': f'{msg.latitude:.8f}', 'lon': f'{msg.longitude:.8f}',
            'alt': f'{msg.altitude:.3f}', 'status': msg.status.status,
            'accepted': int(ok),
            'std_east_m': f'{math.sqrt(max(cov[0], 0.0)):.3f}',
            'std_north_m': f'{math.sqrt(max(cov[4], 0.0)):.3f}',
            'raw_heading_deg': f'{self.raw_heading_deg:.2f}',
        }
        if ok:
            east, north = self.table.frame.to_local(msg.latitude, msg.longitude)
            upd = self.tracker.update(east, north)
            row.update({
                'east_m': f'{east:.3f}', 'north_m': f'{north:.3f}',
                'nearest_cell': _cell_str(upd.nearest),
                'nearest_dist_m': f'{upd.nearest_dist_m:.3f}',
                'reached_cell': _cell_str(upd.cell),
                'walkway_dist_m': f'{upd.walkway_dist_m:.3f}',
                'off_walkway': int(upd.off_walkway),
            })
            self.latest = (msg, row, '')
        else:
            self.latest = (msg, row, reason)
        self.csv.writerow(row)
        self.csv_file.flush()

    def print_status(self) -> None:
        if self.latest is None:
            self.get_logger().warn('no GPS fix received yet', throttle_duration_sec=5.0)
            return
        msg, row, reason = self.latest
        status = STATUS_NAMES.get(msg.status.status, str(msg.status.status))
        head = (f'fix={status} std=({row["std_east_m"]},{row["std_north_m"]}) m '
                f'heading_raw={row["raw_heading_deg"]} deg  n={self.count}')
        if reason:
            print(f'{head}  REJECTED: {reason}', flush=True)
            return
        print(f'{head}  pos=({row["east_m"]},{row["north_m"]}) m  '
              f'nearest={row["nearest_cell"]} @ {row["nearest_dist_m"]} m  '
              f'reached={row["reached_cell"] or "-"}  '
              f'walkway={row["walkway_dist_m"]} m'
              + ('  OFF WALKWAY' if row['off_walkway'] else ''), flush=True)

    def destroy_node(self) -> None:
        self.csv_file.close()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FieldCheckNode()
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
