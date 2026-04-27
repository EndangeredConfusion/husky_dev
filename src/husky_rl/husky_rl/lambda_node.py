#!/usr/bin/env python3
"""
Lambda Node
===========
Pipeline position:
    Discretizer -> [Lambda Node] -> RL Policy

Two modes (set at launch via parameter):

  adaptive := false  — Fixed mode.
      Requires parameter fixed_lambda (list of 4 floats, e.g. [1,2,2,9]).
      Publishes that array once on startup (transient-local, so late subscribers
      still receive it) and never changes it.

  adaptive := true   — Adaptive mode (Algo 2 from adaptive_crl.py).
      Subscribes to /agent_grid_cell.  Divides the 17x11 grid into 4 quadrants
      matching the 4 goal regions.  Every T0=75 new cell transitions, computes
      the visit ratio per region and updates lambda:
          lambda = clip(lambda - eta * (visit_ratio - ci), 0, 10)
      Publishes updated lambda after every T0 window.
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from std_msgs.msg import Float32MultiArray
from husky_interfaces.msg import GridCell


# ── Constants (must match adaptive_crl.py exactly) ───────────────────────────

NUM_GOALS = 4

# Algo 2 hyper-parameters
ETA_LAMBDA = 0.1
CI         = 0.25
T0_CELLS   = 75   # new grid-cell transitions per update window

LAMBDA_MIN = 0.0
LAMBDA_MAX = 10.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_region_index(x: int, y: int) -> int:
    """Quadrant assignment matching adaptive_crl.py SingleAgentWrapper."""
    if x <= 8 and y <= 5:
        return 0   # top-left
    elif x >= 9 and y <= 5:
        return 1   # top-right
    elif x <= 8 and y >= 6:
        return 2   # bottom-left
    else:
        return 3   # bottom-right


# ── Node ─────────────────────────────────────────────────────────────────────

class LambdaNode(Node):
    def __init__(self):
        super().__init__('lambda_node')

        # Parameters
        self.declare_parameter('adaptive',     False)
        self.declare_parameter('fixed_lambda', rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter('cell_topic',   '/agent_grid_cell')
        self.declare_parameter('lambda_topic', '/lambda_values')

        self._adaptive   = self.get_parameter('adaptive').value
        cell_topic       = self.get_parameter('cell_topic').value
        lambda_topic     = self.get_parameter('lambda_topic').value

        # Transient-local publisher so RL policy node gets the value
        # regardless of startup order.
        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.lambda_pub = self.create_publisher(Float32MultiArray, lambda_topic, pub_qos)

        if not self._adaptive:
            # ── Fixed mode ────────────────────────────────────────────────
            try:
                fixed = list(self.get_parameter('fixed_lambda').value)
            except Exception:
                self.get_logger().fatal(
                    'adaptive:=false but fixed_lambda was not provided. '
                    'Pass e.g. --ros-args -p fixed_lambda:="[1,2,2,9]"'
                )
                raise RuntimeError('fixed_lambda parameter required in fixed mode')

            if len(fixed) != NUM_GOALS:
                raise ValueError(
                    f'fixed_lambda must have {NUM_GOALS} values, got {len(fixed)}')

            self._fixed_values = np.clip(
                np.array(fixed, dtype=np.float32), LAMBDA_MIN, LAMBDA_MAX)
            self._publish(self._fixed_values)
            self.get_logger().info(
                f'Lambda node (fixed): published {self._fixed_values.tolist()}')

        else:
            # ── Adaptive mode (Algo 2) ────────────────────────────────────
            self._lambda_values   = np.ones(NUM_GOALS, dtype=np.float32)
            self._region_counts   = np.zeros(NUM_GOALS, dtype=np.float32)
            self._window_counter  = 0   # new-cell transitions in current window
            self._last_cell: tuple | None = None

            sensor_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
            )
            self.cell_sub = self.create_subscription(
                GridCell, cell_topic, self._cell_cb, sensor_qos)

            # Publish initial lambda so RL policy has something on startup.
            self._publish(self._lambda_values)
            self.get_logger().info(
                f'Lambda node (adaptive): started with {self._lambda_values.tolist()}, '
                f'T0={T0_CELLS} cells, eta={ETA_LAMBDA}, ci={CI}')

    # ── Adaptive callback ─────────────────────────────────────────────────────

    def _cell_cb(self, msg: GridCell):
        cell = (msg.x, msg.y)
        if cell == self._last_cell:
            return

        self._last_cell = cell
        self._window_counter += 1
        self._region_counts[get_region_index(msg.x, msg.y)] += 1

        if self._window_counter >= T0_CELLS:
            visit_ratio = self._region_counts / float(T0_CELLS)
            self._lambda_values = np.clip(
                self._lambda_values - ETA_LAMBDA * (visit_ratio - CI),
                LAMBDA_MIN,
                LAMBDA_MAX,
            ).astype(np.float32)

            self._publish(self._lambda_values)
            self.get_logger().info(
                f'[T0 window] visit_ratio={np.round(visit_ratio,3).tolist()} '
                f'-> lambda={np.round(self._lambda_values,3).tolist()}'
            )

            self._region_counts  = np.zeros(NUM_GOALS, dtype=np.float32)
            self._window_counter = 0

    # ── Publish helper ────────────────────────────────────────────────────────

    def _publish(self, values: np.ndarray):
        msg = Float32MultiArray()
        msg.data = values.astype(np.float32).tolist()
        self.lambda_pub.publish(msg)


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = LambdaNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
