#!/usr/bin/env python3
"""Convert GPS fixes into cells for the trained DQN policy.

Uses the '86 Field cell table (config/field86_cells.yaml) instead of a
uniform grid: a cell is "reached" within ``reached_radius_m`` of its centre,
and the last reached cell is held (and published) until the next one is
reached, because the policy counts cell transitions and training assumed the
agent is always at a cell centre.
"""

from __future__ import annotations

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data

from sensor_msgs.msg import NavSatFix
from husky_interfaces.msg import GridCell

from husky_rl.astar_node import GRID_HEIGHT, GRID_WIDTH, OPEN_CELLS
from husky_rl.cell_table import CellTable, CellTracker, default_table_path, fix_accepted


class DiscretizerNode(Node):
    """Bridge ``NavSatFix`` GPS fixes to the 17x11 policy grid."""

    def __init__(self) -> None:
        super().__init__('discretizer_node')

        self.declare_parameter('cell_table', '') # field86_cells.yaml
        self.declare_parameter('gps_fix_topic', '/gps/fix') # GPS input 
        self.declare_parameter('cell_topic', '/agent_grid_cell') 
        self.declare_parameter('reached_radius_m', 1.5)
        self.declare_parameter('off_walkway_m', 6.0)    # adjustable
        self.declare_parameter('bootstrap_nearest', True)   # very first cell is the nearest one
        self.declare_parameter('require_rtk', False)    # acccept only RTK fixes
        self.declare_parameter('max_horizontal_std_m', 0.0)     # reject fixes whose reported accuracy is worse than this
        self.declare_parameter('publish_only_on_change', True)
        # Re-send the held cell so nodes that start late (rl_policy takes ~2 s to load the DQN) still get it while the robot stands still.
        # rl_policy only counts changes, so repeats do not affect it.
        self.declare_parameter('republish_period_s', 1.0)   # Re-send the held cell every second

        def p(name):
            return self.get_parameter(name).value

        table_path = str(p('cell_table')) or default_table_path()
        fix_topic = str(p('gps_fix_topic'))
        cell_topic = str(p('cell_topic'))
        self._require_rtk = bool(p('require_rtk'))
        self._max_std = float(p('max_horizontal_std_m'))
        self._publish_only_on_change = bool(p('publish_only_on_change'))

        # Fail loudly: a wrong table would silently feed the policy bad cells.
        table = CellTable.load(table_path)
        table.check_against(OPEN_CELLS, GRID_WIDTH, GRID_HEIGHT)
        for cell, err in table.mismatches:
            self.get_logger().warning(
                f'cell {cell}: lat/lon and east_m/north_m differ by {err:.2f} m; '
                'using lat/lon.')
        self._tracker = CellTracker(
            table,
            reached_radius_m=float(p('reached_radius_m')),
            off_walkway_m=float(p('off_walkway_m')),
            bootstrap_nearest=bool(p('bootstrap_nearest')))
        self._table = table
        self._last_reject_reason = ''
        self._was_off_walkway = False

        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._cell_publisher = self.create_publisher(GridCell, cell_topic, reliable_qos)
        # sensor-data QoS (best effort) also matches reliable GPS publishers
        self._fix_subscription = self.create_subscription(
            NavSatFix, fix_topic, self._fix_callback, qos_profile_sensor_data)
        republish_period = float(p('republish_period_s'))
        if republish_period > 0.0:
            self.create_timer(republish_period, self._republish)

        self.get_logger().info(
            f'Discretizer ready: {fix_topic} -> {cell_topic}; '
            f'table={table_path} ({len(table.centers)} cells), '
            f'reached_radius_m={self._tracker.reached_radius_m}, '
            f'off_walkway_m={self._tracker.off_walkway_m}, '
            f'bootstrap_nearest={self._tracker.bootstrap_nearest}, '
            f'require_rtk={self._require_rtk}')

    def _fix_callback(self, msg: NavSatFix) -> None:
        ok, reason = fix_accepted(
            msg.status.status, msg.latitude, msg.longitude,
            require_rtk=self._require_rtk,
            covariance=msg.position_covariance,
            covariance_type=msg.position_covariance_type,
            max_horizontal_std_m=self._max_std)
        if not ok:
            if reason != self._last_reject_reason:
                self.get_logger().warning(f'GPS fix rejected: {reason}')
                self._last_reject_reason = reason
            return
        self._last_reject_reason = ''

        east, north = self._table.frame.to_local(msg.latitude, msg.longitude)
        upd = self._tracker.update(east, north)

        if upd.off_walkway:
            self.get_logger().warning(
                f'Off the walkways: ({east:.1f}, {north:.1f}) m is '
                f'{upd.walkway_dist_m:.1f} m from the nearest walk; no new cell.',
                throttle_duration_sec=2.0)
            self._was_off_walkway = True
            return
        if self._was_off_walkway:
            self.get_logger().info('Back on the walkways.')
            self._was_off_walkway = False

        if upd.cell is None:
            return
        if upd.changed:
            if upd.bootstrapped:
                self.get_logger().info(
                    f'Start cell {upd.cell} (nearest, {upd.nearest_dist_m:.1f} m away).')
            else:
                self.get_logger().info(
                    f'Reached cell {upd.cell} ({upd.nearest_dist_m:.2f} m from centre).')
            if upd.skipped_from is not None:
                self.get_logger().warning(
                    f'Cell {upd.cell} is not adjacent to the previous cell '
                    f'{upd.skipped_from}; at least one cell was skipped.')
        elif self._publish_only_on_change:
            return
        self._publish(upd.cell)

    def _republish(self) -> None:
        if self._tracker.cell is not None:
            self._publish(self._tracker.cell)

    def _publish(self, cell) -> None:
        cell_msg = GridCell()
        cell_msg.x, cell_msg.y = cell
        self._cell_publisher.publish(cell_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DiscretizerNode()
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
