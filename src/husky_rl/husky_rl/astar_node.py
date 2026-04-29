#!/usr/bin/env python3
"""
A* Node
=======
Pipeline position:
    RL Policy --/rl_goal--> [A* Node] --/rl_path--> Controller
    Discretizer --/agent_grid_cell--> [A* Node]

Triggered by every new /rl_goal.  Runs A* on the hardcoded 17x11 grid
(identical to the training environment) and publishes the resulting
waypoint sequence as Int32MultiArray([x0,y0, x1,y1, ...]) to the controller.

TODO: grid -> world coordinate conversion.
      When cell_size_m is known (measured in lab), multiply each grid
      coordinate by cell_size_m before publishing so the controller's
      Stanley algorithm operates in metres.
      Until then, coordinates are published as raw grid integers.
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Int32MultiArray
from husky_interfaces.msg import GridCell

from pathfinding.core.grid import Grid as PFGrid
from pathfinding.finder.a_star import AStarFinder
from pathfinding.core.diagonal_movement import DiagonalMovement


# ── Map (must match adaptive_crl.py _gen_grid exactly) ───────────────────────

GRID_WIDTH  = 17
GRID_HEIGHT = 11

OPEN_CELLS = {
    # top path
    (1,1),(2,1),(3,1),(4,1),(5,1),(6,1),(7,1),(8,1),
    (9,1),(10,1),(11,1),(12,1),(13,1),(14,1),(15,1),
    # left vertical
    (1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8),(1,9),
    # middle branch
    (2,5),(3,5),(3,4),(4,3),(3,3),(4,2),(5,3),
    (5,4),(5,5),(5,6),(5,7),(6,7),(7,7),(7,8),
    # bottom horizontal
    (2,9),(3,9),(4,9),(5,9),(6,9),(7,9),(8,9),(9,9),(10,9),
    (11,9),(12,9),(13,9),(14,9),(15,9),
    # lower stair/diagonal
    (10,6),(10,7),(10,8),(11,6),(12,6),(13,6),(14,6),(15,6),
    (12,2),(12,3),(12,4),(12,5),(15,2),(15,3),(15,4),(15,5),
    # goal cells
    (1,1),(15,1),(1,9),(15,9),
}

def _build_matrix() -> np.ndarray:
    """Build the 0/1 walkability matrix once at startup."""
    mat = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.int32)
    for (x, y) in OPEN_CELLS:
        mat[y, x] = 1
    return mat

WALKABILITY_MATRIX = _build_matrix()


# ── Node ─────────────────────────────────────────────────────────────────────

class AStarNode(Node):
    def __init__(self):
        super().__init__('astar_node')

        # Parameters
        self.declare_parameter('goal_topic',  '/rl_goal')
        self.declare_parameter('cell_topic',  '/agent_grid_cell')
        self.declare_parameter('path_topic',  '/rl_path')
        # TODO: set cell_size_m to the measured lab value (metres per grid cell)
        # to convert grid coordinates to physical coordinates for the controller.
        self.declare_parameter('cell_size_m', 0.0)

        goal_topic    = self.get_parameter('goal_topic').value
        cell_topic    = self.get_parameter('cell_topic').value
        path_topic    = self.get_parameter('path_topic').value
        self._cell_size_m = float(self.get_parameter('cell_size_m').value)

        # Internal state
        self._current_cell: tuple[int, int] | None = None
        self._current_goal: tuple[int, int] | None = None

        # QoS
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.cell_sub = self.create_subscription(
            GridCell, cell_topic, self._cell_cb, sensor_qos)
        self.goal_sub = self.create_subscription(
            GridCell, goal_topic, self._goal_cb, reliable_qos)

        self.path_pub = self.create_publisher(
            Int32MultiArray, path_topic, reliable_qos)

        scale_note = (f'cell_size_m={self._cell_size_m} m'
                      if self._cell_size_m > 0
                      else 'cell_size_m not set — publishing raw grid integers (TODO)')
        self.get_logger().info(
            f'A* node ready.\n'
            f'  sub: {goal_topic} | {cell_topic}\n'
            f'  pub: {path_topic}\n'
            f'  {scale_note}'
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _cell_cb(self, msg: GridCell):
        first_fix = self._current_cell is None
        self._current_cell = (msg.x, msg.y)
        # If a goal was already received before the first cell arrived, run now.
        if first_fix and self._current_goal is not None:
            self._run_and_publish(self._current_cell, self._current_goal)

    def _goal_cb(self, msg: GridCell):
        goal = (msg.x, msg.y)

        if goal == self._current_goal:
            return  # same goal republished, skip

        self._current_goal = goal

        if self._current_cell is None:
            self.get_logger().warn('Goal received but no current cell yet — waiting.')
            return

        self._run_and_publish(self._current_cell, goal)

    # ── A* + publish ──────────────────────────────────────────────────────────

    def _run_and_publish(self, start: tuple[int, int], goal: tuple[int, int]):
        grid = PFGrid(matrix=WALKABILITY_MATRIX.tolist())
        start_node = grid.node(start[0], start[1])
        goal_node  = grid.node(goal[0],  goal[1])

        finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
        path, _ = finder.find_path(start_node, goal_node, grid)

        if not path:
            self.get_logger().error(
                f'A* found no path from {start} to {goal}.')
            return

        # Flatten path to [x0,y0, x1,y1, ...] as raw grid integers.
        # TODO: once cell_size_m is measured, coordinate scaling to metres
        # requires switching to Float64MultiArray — coordinate with controller.
        flat: list[int] = []
        for node in path:
            flat.append(node.x)
            flat.append(node.y)

        msg = Int32MultiArray()
        msg.data = flat
        self.path_pub.publish(msg)

        self.get_logger().info(
            f'A* {start} -> {goal}: {len(path)} waypoints published.')


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = AStarNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
