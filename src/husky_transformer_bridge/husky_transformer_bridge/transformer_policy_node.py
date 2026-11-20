"""Candidate-only ROS adapter. No execution feedback and no history commits."""
import json
import math
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import String, Int32, Float32MultiArray
from husky_interfaces.msg import GridCell
from husky_transformer_policy.protocol import request


class TransformerPolicy(Node):
    def __init__(self):
        super().__init__('transformer_policy')
        self.declare_parameter('socket_path', '/tmp/husky_policy_candidates.sock')
        self.declare_parameter('observation_timeout_s', 2.0)
        self.path = self.get_parameter('socket_path').value
        self.state = None
        self.received = 0.0
        self.lam = None
        self.generation = 0
        self.last_key = None
        self.future = None
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.task = None
        self.task_lambda = None
        self.goal = self.create_publisher(GridCell, '/rl_goal', 10)
        self.index = self.create_publisher(Int32, '/rl_goal_index', 10)
        self.decision = self.create_publisher(String, '/transformer/decision', 10)
        self.status = self.create_publisher(String, '/transformer/status', 10)
        self.create_subscription(String, '/agent_grid_cell/state', self.on_state, 10)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Float32MultiArray, '/lambda_values', self.on_lambda, qos)
        self.create_timer(0.1, self.tick)

    def on_lambda(self, msg):
        values = tuple(float(v) for v in msg.data)
        valid = len(values) == 4 and all(math.isfinite(v) and 0 <= v <= 10 for v in values)
        values = values if valid else None
        if values != self.lam:
            self.generation += 1
            self.lam = values

    def on_state(self, msg):
        try:
            s = json.loads(msg.data)
            cell = s['cell']
            if (type(s['valid']) is not bool or type(s['state_seq']) is not int
                    or not isinstance(cell, list) or len(cell) != 2
                    or any(type(v) is not int for v in cell)
                    or not (0 <= cell[0] <= 16 and 0 <= cell[1] <= 10)
                    or not math.isfinite(s['source_stamp'])):
                raise ValueError('Invalid observation')
            if not s['valid']:
                self.generation += 1
            self.state = s
            self.received = time.monotonic()
        except (ValueError, TypeError, KeyError):
            self.state = None
            self.generation += 1

    def fresh(self):
        if self.state is None or not self.state['valid'] or self.lam is None:
            return False
        timeout = float(self.get_parameter('observation_timeout_s').value)
        age = self.get_clock().now().nanoseconds * 1e-9 - self.state['source_stamp']
        return 0 <= age <= timeout and time.monotonic() - self.received <= timeout

    def key(self):
        return (self.generation, self.lam, self.state['state_seq'], tuple(self.state['cell']))

    def infer(self, state, lam):
        # The stage 1-5 adapter deliberately does not fabricate execution rewards.
        if self.task is None or lam != self.task_lambda:
            self.task = request(self.path, 'init_task', lambda_values=list(lam))
            self.task_lambda = lam
        task = self.task
        result = request(self.path, 'infer', epoch=task['epoch'],
                         history_version=task['history_version'],
                         state_seq=state['state_seq'], agent_pos=state['cell'])
        request(self.path, 'cancel', epoch=result['epoch'], ticket=result['ticket'],
                history_version=result['history_version'])
        return result

    def tick(self):
        if self.future is not None:
            if not self.future.done():
                return
            try:
                result = self.future.result()
                if self.fresh() and self.pending_key == self.key():
                    cell = result['goal_grid_xy']
                    decision = dict(decision_id=str(uuid.uuid4()),
                                    state_seq=self.state['state_seq'], start=self.state['cell'],
                                    goal=cell, goal_index=result['goal_index'],
                                    source_stamp=self.state['source_stamp'],
                                    history_version=result['history_version'],
                                    mode='candidate_only', logits=result['logits'])
                    self.decision.publish(String(data=json.dumps(decision)))
                    self.goal.publish(GridCell(x=cell[0], y=cell[1]))
                    self.index.publish(Int32(data=result['goal_index']))
                    self.last_key = self.pending_key
                    self.status.publish(String(data=json.dumps(dict(ok=True, **decision))))
            except Exception as exc:
                self.task = None
                self.get_logger().warning(str(exc), throttle_duration_sec=5.0)
            finally:
                self.future = None
            return
        if self.fresh() and self.key() != self.last_key:
            self.pending_key = self.key()
            self.future = self.pool.submit(self.infer, dict(self.state), self.lam)


def main(args=None):
    rclpy.init(args=args)
    node = TransformerPolicy()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pool.shutdown(wait=True, cancel_futures=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
