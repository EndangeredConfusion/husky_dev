#!/usr/bin/env python3
"""Convert continuous EKF positions into cells for the trained DQN policy."""

from __future__ import annotations

from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from husky_interfaces.msg import EkfState, GridCell

from husky_rl.grid_geometry import GridGeometry


class DiscretizerNode(Node):
    """Bridge ``EkfState`` world coordinates to the 17x11 policy grid."""

    def __init__(self) -> None:
        super().__init__('discretizer_node')

        self.declare_parameter('localization_topic', 'uwb_local/ekf')
        self.declare_parameter('cell_topic', '/agent_grid_cell')
        self.declare_parameter('grid_width', 17)
        self.declare_parameter('grid_height', 11)
        self.declare_parameter('cell_size_m', 0.0)
        self.declare_parameter('origin_x_m', 0.0)
        self.declare_parameter('origin_y_m', 0.0)
        self.declare_parameter('invert_y_axis', False)
        self.declare_parameter('publish_only_on_change', True)

        localization_topic = str(self.get_parameter('localization_topic').value)
        cell_topic = str(self.get_parameter('cell_topic').value)
        self._publish_only_on_change = bool(
            self.get_parameter('publish_only_on_change').value)
        self._last_cell: Optional[Tuple[int, int]] = None
        self._last_rejected_cell: Optional[Tuple[float, float]] = None

        try:
            self._geometry: Optional[GridGeometry] = GridGeometry(
                width=int(self.get_parameter('grid_width').value),
                height=int(self.get_parameter('grid_height').value),
                cell_size_m=float(self.get_parameter('cell_size_m').value),
                origin_x_m=float(self.get_parameter('origin_x_m').value),
                origin_y_m=float(self.get_parameter('origin_y_m').value),
                invert_y_axis=bool(self.get_parameter('invert_y_axis').value),
            )
        except ValueError as exc:
            self._geometry = None
            self.get_logger().error(
                f'Discretizer disabled: {exc}. Set the measured cell_size_m '
                'and grid origin before operating the pipeline.')

        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._cell_publisher = self.create_publisher(
            GridCell, cell_topic, reliable_qos)
        self._localization_subscription = self.create_subscription(
            EkfState, localization_topic, self._localization_callback, reliable_qos)

        if self._geometry is not None:
            self.get_logger().info(
                f'Discretizer ready: {localization_topic} -> {cell_topic}; '
                f'grid={self._geometry.width}x{self._geometry.height}, '
                f'cell_size_m={self._geometry.cell_size_m}, '
                f'origin=({self._geometry.origin_x_m}, '
                f'{self._geometry.origin_y_m}), '
                f'invert_y_axis={self._geometry.invert_y_axis}')

    def _localization_callback(self, msg: EkfState) -> None:
        if self._geometry is None:
            return

        x_m = float(msg.position_m.x)
        y_m = float(msg.position_m.y)
        cell = self._geometry.world_to_grid(x_m, y_m)
        if cell is None:
            rejected_position = (x_m, y_m)
            if rejected_position != self._last_rejected_cell:
                self.get_logger().warning(
                    f'Localization position ({x_m:.3f}, {y_m:.3f}) is outside '
                    'the configured grid; no cell published.')
                self._last_rejected_cell = rejected_position
            return

        self._last_rejected_cell = None
        if self._publish_only_on_change and cell == self._last_cell:
            return

        cell_msg = GridCell()
        cell_msg.x, cell_msg.y = cell
        self._cell_publisher.publish(cell_msg)
        self._last_cell = cell


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DiscretizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
