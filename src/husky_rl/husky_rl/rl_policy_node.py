#!/usr/bin/env python3
"""
RL Policy Node
==============
Pipeline position:
    Discretizer -> [RL Policy] -> A* Node -> Controller
                        ^
                  Lambda Node

Subscribes to the agent's current grid cell and the lambda array.
Every 5 new grid-cell transitions, queries the DQN and publishes
one of the 4 fixed goal cells to the controller.
"""

from __future__ import annotations

import os
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from ament_index_python.packages import get_package_share_directory

from std_msgs.msg import Int32, Float32MultiArray
from husky_interfaces.msg import GridCell

from stable_baselines3 import DQN


# ── Constants (must match training environment exactly) ──────────────────────

GRID_WIDTH  = 17
GRID_HEIGHT = 11

# Action index i -> GOAL_POSITIONS[i].  Order must match training.
GOAL_POSITIONS = [
    (1,  1),   # 0 top-left
    (15, 1),   # 1 top-right
    (1,  9),   # 2 bottom-left
    (15, 9),   # 3 bottom-right
]
NUM_GOALS = len(GOAL_POSITIONS)

RESELECT_EVERY_N_CELLS = 5

DEFAULT_LAMBDA = np.ones(NUM_GOALS, dtype=np.float32)


# ── Node ─────────────────────────────────────────────────────────────────────

class RLPolicyNode(Node):
    def __init__(self):
        super().__init__('rl_policy_node')

        # Parameters
        _default_model = os.path.join(
            get_package_share_directory('husky_rl'),
            'models', 'dqn_eightsixmulti_lambda.zip')
        self.declare_parameter('model_path', _default_model)
        self.declare_parameter('cell_topic',       '/agent_grid_cell')
        self.declare_parameter('lambda_topic',     '/lambda_values')
        self.declare_parameter('goal_topic',       '/rl_goal')
        self.declare_parameter('goal_index_topic', '/rl_goal_index')

        model_path       = self.get_parameter('model_path').value
        cell_topic       = self.get_parameter('cell_topic').value
        lambda_topic     = self.get_parameter('lambda_topic').value
        goal_topic       = self.get_parameter('goal_topic').value
        goal_index_topic = self.get_parameter('goal_index_topic').value

        # Load model
        self.get_logger().info(f'Loading DQN from: {model_path}')
        try:
            self.model = DQN.load(model_path, device='cpu')
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            raise
        self.get_logger().info('DQN loaded.')

        # Internal state
        self.latest_lambda: np.ndarray     = DEFAULT_LAMBDA.copy()
        self.has_lambda:    bool           = False
        self.last_cell:     tuple | None   = None
        self.cell_counter:  int            = 0   # counts new-cell transitions
        self.current_goal:  int | None     = None
        self._goal_triggered: bool         = False  # prevents goal-cell spam

        # QoS
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        # Transient-local so we get the last lambda even if lambda_node
        # started before us.
        lambda_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # Subscriptions
        self.cell_sub = self.create_subscription(
            GridCell, cell_topic, self._cell_cb, sensor_qos)
        self.lambda_sub = self.create_subscription(
            Float32MultiArray, lambda_topic, self._lambda_cb, lambda_qos)

        # Publishers
        self.goal_pub = self.create_publisher(GridCell, goal_topic, reliable_qos)
        self.goal_idx_pub = self.create_publisher(Int32, goal_index_topic, reliable_qos)

        self.get_logger().info(
            f'RL policy node ready.\n'
            f'  sub: {cell_topic} | {lambda_topic}\n'
            f'  pub: {goal_topic} | {goal_index_topic}\n'
            f'  reselect every {RESELECT_EVERY_N_CELLS} new cells\n'
            f'  using default lambda {DEFAULT_LAMBDA.tolist()} until first message'
        )

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _lambda_cb(self, msg: Float32MultiArray):
        data = np.array(msg.data, dtype=np.float32)
        if data.shape != (NUM_GOALS,):
            self.get_logger().warn(
                f'Bad lambda shape {data.shape}, expected ({NUM_GOALS},) — ignored.')
            return
        self.latest_lambda = np.clip(data, 0.0, 10.0)
        self.has_lambda = True

    def _cell_cb(self, msg: GridCell):
        cell = (msg.x, msg.y)

        if not (0 <= cell[0] < GRID_WIDTH and 0 <= cell[1] < GRID_HEIGHT):
            self.get_logger().warn(f'Out-of-bounds cell {cell} — ignored.')
            return

        is_new = (self.last_cell is None) or (cell != self.last_cell)
        if is_new:
            if self.last_cell is not None:
                self.cell_counter += 1
            self.last_cell = cell
            self._goal_triggered = False  # reset whenever we enter a new cell

        cold_start      = self.current_goal is None
        periodic        = (is_new
                           and self.cell_counter > 0
                           and self.cell_counter % RESELECT_EVERY_N_CELLS == 0)
        on_current_goal = (not self._goal_triggered
                           and self.current_goal is not None
                           and cell == GOAL_POSITIONS[self.current_goal])

        if cold_start or periodic or on_current_goal:
            if on_current_goal:
                self._goal_triggered = True
            self._select_and_publish(cell)

    # ── Inference + publish ──────────────────────────────────────────────────

    def _select_and_publish(self, cell: tuple[int, int]):
        obs = {
            'agent_pos':    np.array(cell, dtype=np.float32),
            'lambda_values': self.latest_lambda.astype(np.float32),
        }
        try:
            action, _ = self.model.predict(obs, deterministic=True)
        except Exception as e:
            self.get_logger().error(f'model.predict failed: {e}')
            return

        action = int(action.item()) if isinstance(action, np.ndarray) else int(action)

        if not (0 <= action < NUM_GOALS):
            self.get_logger().error(f'Invalid action {action} from policy.')
            return

        self.current_goal = action
        gx, gy = GOAL_POSITIONS[action]

        goal_msg = GridCell()
        goal_msg.x = gx
        goal_msg.y = gy
        self.goal_pub.publish(goal_msg)

        idx_msg = Int32()
        idx_msg.data = action
        self.goal_idx_pub.publish(idx_msg)

        self.get_logger().info(
            f'[cells={self.cell_counter}] pos={cell} '
            f'lambda={self.latest_lambda.tolist()} '
            f'-> goal_idx={action} cell=({gx},{gy})'
            + ('' if self.has_lambda else '  [default lambda]')
        )


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = RLPolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
