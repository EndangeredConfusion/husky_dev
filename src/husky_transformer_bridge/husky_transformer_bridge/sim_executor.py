"""Gazebo-only primitive executor. No real-robot topic or launch entry point."""
import json
import math
import os
import time
import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from husky_rl.astar_node import OPEN_CELLS, WALKABILITY_MATRIX
from husky_rl.cell_table import CellTable, default_table_path
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from pathfinding.core.diagonal_movement import DiagonalMovement
from husky_transformer_bridge.macro_execution import MacroExecution


def wrap(a):
    return (a + math.pi) % (2*math.pi) - math.pi


class SimExecutor(Node):
    def __init__(self):
        if not 1 <= int(os.environ.get('ROS_DOMAIN_ID', '0')) <= 232 or not os.environ.get('GZ_PARTITION', '').startswith('husky_transformer_stage67'):
            raise RuntimeError('Simulation requires isolated ROS domain and Gazebo partition')
        super().__init__('transformer_sim_executor')
        self.table = CellTable.load(default_table_path())
        self.pose = self.offset = None
        self.pose_time = 0.0
        self.pose_seq = 0
        self.last_pose_stamp = None
        self.cell, self.direction = (4, 3), 0
        self.engine = None
        self.command = self.primitive = None
        self.paused = True
        self.finished_ids = set()
        self.heartbeat_time = 0.0
        self.interrupted_macro = False
        self.velocity = self.create_publisher(Twist, '/transformer_sim/cmd_vel', 10)
        self.events = self.create_publisher(String, '/transformer_sim/feedback', 10)
        self.observations = self.create_publisher(String, '/transformer_sim/state', 10)
        self.create_subscription(Odometry, '/transformer_sim/odom', self.on_pose, 10)
        self.create_subscription(String, '/transformer_sim/command', self.on_command, 10)
        self.create_subscription(String, '/transformer_sim/heartbeat', self.on_heartbeat, 10)
        self.create_timer(.05, self.tick)

    def send(self, status, **extra):
        identity = {} if self.command is None else {
            k: self.command[k] for k in ('decision_id', 'epoch', 'history_version', 'state_seq', 'action')}
        self.events.publish(String(data=json.dumps(dict(identity, status=status, **extra))))

    def stop(self):
        self.velocity.publish(Twist())

    def on_heartbeat(self, msg):
        self.heartbeat_time = time.monotonic()

    def fail(self, reason):
        self.stop()
        if self.command is not None:
            self.interrupted_macro = True
        self.send('failed', reason=reason)
        self.command = self.primitive = None
        self.paused = True

    def on_pose(self, msg):
        stamp = msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec
        if self.last_pose_stamp is not None and stamp <= self.last_pose_stamp:
            return
        self.last_pose_stamp = stamp
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
        if not all(math.isfinite(v) for v in (p.x, p.y, yaw)):
            self.fail('invalid simulation pose')
            return
        if self.offset is None:
            cx, cy = self.table.center(self.cell)
            self.offset = (cx-p.x, cy-p.y)
        self.pose = (p.x+self.offset[0], p.y+self.offset[1], yaw)
        self.pose_time = time.monotonic()
        self.pose_seq += 1

    def plan(self, start, goal):
        grid = Grid(matrix=WALKABILITY_MATRIX.tolist())
        path, _ = AStarFinder(diagonal_movement=DiagonalMovement.never).find_path(
            grid.node(*start), grid.node(*goal), grid)
        return [(n.x, n.y) for n in path]

    def on_command(self, msg):
        try:
            d = json.loads(msg.data)
            if d['op'] == 'stop':
                self.fail('explicit stop')
                self.send('stopped')
                return
            if d['op'] == 'initialize':
                if self.command is not None:
                    raise ValueError('Stop active macro before initialization')
                if self.interrupted_macro:
                    raise ValueError('Interrupted macro: restart simulation episode and executor before reset')
                if self.pose is None or time.monotonic()-self.pose_time > 1:
                    raise ValueError('Initialization requires fresh simulation pose')
                self.engine = MacroExecution(self.plan, OPEN_CELLS, d['lambda_values'])
                self.finished_ids.clear()
                self.paused = False
                self.send('initialized')
                return
            if d['op'] != 'execute' or self.paused or self.engine is None:
                raise ValueError('Executor is not initialized')
            ident = d['decision_id']
            if ident in self.finished_ids or (self.command and ident == self.command['decision_id']):
                return
            if self.command is not None or tuple(d['start']) != self.cell:
                raise ValueError('Busy executor or stale start cell')
            if self.pose is None or time.monotonic()-self.pose_time > 1:
                raise ValueError('No fresh simulation pose')
            self.engine.begin(d['action'], self.cell, self.direction)
            self.command = d
            self.send('accepted')
        except (ValueError, TypeError, KeyError, RuntimeError) as exc:
            self.fail(str(exc))

    def tick(self):
        fresh = self.pose is not None and time.monotonic()-self.pose_time <= 1
        self.observations.publish(String(data=json.dumps(dict(
            valid=fresh and not self.paused, cell=self.cell, direction=self.direction,
            busy=self.command is not None, pose_seq=self.pose_seq))))
        if not fresh:
            if self.command is not None:
                self.fail('simulation localization timeout')
            return
        if self.command is None:
            return
        if time.monotonic() - self.heartbeat_time > 2.0:
            self.fail('coordinator heartbeat timeout')
            return
        if self.primitive is None:
            try:
                self.primitive = self.engine.next_primitive()
            except Exception as exc:
                self.fail(str(exc))
                return
            self.primitive_started = time.monotonic()
            self.primitive_pose_seq = self.pose_seq
            self.turn_yaw = wrap(self.pose[2] - math.pi/2)
        if time.monotonic()-self.primitive_started > 120:
            self.fail('primitive execution timeout')
            return
        p = self.primitive
        x, y, yaw = self.pose
        done = False
        cmd = Twist()
        if p['primitive_action'] == 1:
            error = wrap(self.turn_yaw-yaw)
            done = abs(error) < .06
            cmd.angular.z = max(-1., min(1., 1.5*error))
        elif p['blocked']:
            # Virtual training-grid wall: hold position; require a subsequent pose.
            tx, ty = self.table.center(self.cell)
            done = self.pose_seq > self.primitive_pose_seq and math.hypot(tx-x, ty-y) < .2
        else:
            tx, ty = self.table.center(p['end'])
            distance = math.hypot(tx-x, ty-y)
            error = wrap(math.atan2(ty-y, tx-x)-yaw)
            done = distance < .12
            cmd.angular.z = max(-1., min(1., 2*error))
            cmd.linear.x = min(1., distance) if abs(error) < .2 else 0.
        if not done:
            self.velocity.publish(cmd)
            return
        self.stop()
        self.cell, self.direction = p['end'], p['direction']
        macro_done = self.engine.confirm(self.cell, self.direction)
        self.send('primitive_completed', primitive=self.engine.trace[-1])
        self.primitive = None
        if macro_done:
            self.send('completed', **self.engine.feedback())
            self.finished_ids.add(self.command['decision_id'])
            if len(self.finished_ids) > 1024:
                self.finished_ids = {self.command['decision_id']}
            self.command = None
            if self.engine.episode_steps >= self.engine.max_steps:
                self.paused = True


def main():
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = SimExecutor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.command = node.primitive = None
        node.paused = True
        node.stop()
        rclpy.spin_once(node, timeout_sec=.05)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
