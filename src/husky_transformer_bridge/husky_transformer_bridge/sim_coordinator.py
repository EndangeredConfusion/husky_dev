"""Simulation-only policy/execution/history coordinator; explicit fault recovery."""
import json
import os
import time
import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String
from husky_transformer_policy.protocol import request
from husky_transformer_bridge.execution_session import ExecutionSession


class SimCoordinator(Node):
    def __init__(self):
        if not 1 <= int(os.environ.get('ROS_DOMAIN_ID', '0')) <= 232 or not os.environ.get('GZ_PARTITION', '').startswith('husky_transformer_stage67'):
            raise RuntimeError('Simulation requires isolated ROS domain and Gazebo partition')
        super().__init__('transformer_sim_coordinator')
        self.declare_parameter('socket_path', '/tmp/husky_policy_execution.sock')
        self.declare_parameter('max_decisions', 6)
        self.declare_parameter('lambda_values', [7.0, 2.0, 2.0, 2.0])
        self.socket = self.get_parameter('socket_path').value
        self.lam = list(self.get_parameter('lambda_values').value)
        self.limit = int(self.get_parameter('max_decisions').value)
        self.session = ExecutionSession(self.rpc)
        self.state = None
        self.state_time = 0.0
        self.initialized = False
        self.started = False
        self.stopped = False
        self.fault = None
        self.committed = 0
        self.sequence = 0
        self.last_health = 0.0
        self.command_time = 0.0
        self.commands = self.create_publisher(String, '/transformer_sim/command', 10)
        self.audit = self.create_publisher(String, '/transformer_sim/audit', 10)
        self.heartbeat = self.create_publisher(String, '/transformer_sim/heartbeat', 10)
        self.create_subscription(String, '/transformer_sim/state', self.on_state, 10)
        self.create_subscription(String, '/transformer_sim/feedback', self.on_feedback, 10)
        self.create_subscription(String, '/transformer_sim/control', self.on_control, 10)
        self.create_timer(.1, self.tick)

    def rpc(self, op, **payload):
        return request(self.socket, op, timeout=1.0, **payload)

    def emit(self, event, **extra):
        record = dict(event=event, committed=self.committed, **extra)
        self.audit.publish(String(data=json.dumps(record)))
        self.get_logger().info(json.dumps(record))

    def command(self, op, **extra):
        self.commands.publish(String(data=json.dumps(dict(op=op, **extra))))

    def fail(self, reason):
        self.command('stop')
        if self.fault is not None:
            return
        self.fault = str(reason)
        self.initialized = False
        try:
            self.session.abort(reason)
        except Exception:
            pass  # Uncertain commits remain preserved for explicit retry.
        self.emit('paused', reason=self.fault)

    def on_state(self, msg):
        try:
            self.state = json.loads(msg.data)
            self.state_time = time.monotonic()
        except (ValueError, TypeError):
            self.fail('malformed simulation state')

    def on_feedback(self, msg):
        try:
            f = json.loads(msg.data)
            status = f['status']
            if status == 'stopped':
                self.stopped = True
                return
            if status == 'initialized':
                self.initialized = True
                return
            if self.fault is not None:
                self.emit('feedback_ignored_while_paused', status=status)
                return
            if status == 'accepted':
                self.session.accepted(f['decision_id'])
            elif status == 'completed':
                duplicate = f['decision_id'] in self.session.completed
                result = self.session.complete(f)
                if not duplicate:
                    self.committed += 1
                    self.emit('history_committed', feedback=f, worker=result)
                if f['terminated'] or f['truncated']:
                    self.fail('episode ended; explicit new simulation episode required')
            elif status == 'failed':
                self.fail(f.get('reason', 'execution failed'))
        except Exception as exc:
            self.fail(str(exc))

    def on_control(self, msg):
        try:
            q = json.loads(msg.data)
            if q['op'] == 'stop':
                self.fail('operator stop')
            elif q['op'] == 'retry_commit':
                result = self.session.retry_commit()
                self.committed += 1
                self.emit('history_commit_recovered', worker=result)
                # Executor was stopped: do not automatically resume movement.
            elif q['op'] == 'reset':
                if not self.stopped or self.state is None or self.state['busy']:
                    raise ValueError('Wait for executor stop acknowledgment before reset')
                if self.session.commit_request is not None:
                    raise ValueError('Resolve uncertain commit before reset, or explicitly confirm worker loss')
                self.session.reset_after_worker_loss()
                self.lam = q.get('lambda_values', self.lam)
                self.session.initialize(self.lam)
                self.initialized = False
                self.command('initialize', lambda_values=self.lam)
                self.fault = None
                self.stopped = False
                self.committed = 0
                self.emit('task_reset', worker=self.session.task)
            else:
                raise ValueError('Unknown control operation')
        except Exception as exc:
            self.emit('control_rejected', reason=str(exc))

    def tick(self):
        self.heartbeat.publish(String(data='alive'))
        if self.fault is not None:
            return
        if not self.started:
            if self.commands.get_subscription_count() == 0:
                return
            try:
                self.session.initialize(self.lam)
                self.command('initialize', lambda_values=self.lam)
                self.started = True
                self.emit('task_initialized', worker=self.session.task)
            except Exception as exc:
                self.get_logger().warning(str(exc), throttle_duration_sec=5.)
            return
        if not self.initialized:
            return
        if self.state is None or time.monotonic()-self.state_time > 1 or not self.state['valid']:
            self.fail('simulation state invalid or stale')
            return
        try:
            if time.monotonic()-self.last_health > .5:
                status = self.rpc('status')
                self.last_health = time.monotonic()
                if status['epoch'] != self.session.task['epoch'] or status['paused']:
                    raise RuntimeError('Worker task changed or paused')
            if self.session.pending is not None:
                if self.session.pending['status'] == 'proposed' and time.monotonic()-self.command_time > 2:
                    raise RuntimeError('Execution acceptance timed out')
                return
            if self.committed >= self.limit:
                self.command('stop')
                self.fault = 'decision limit reached'
                self.emit('run_complete', worker=self.session.task)
                return
            if self.state['busy']:
                return
            self.sequence += 1
            d = self.session.infer(self.state['cell'], self.sequence)
            self.command('execute', **d)
            self.command_time = time.monotonic()
            self.emit('decision', decision=d)
        except Exception as exc:
            self.fail(str(exc))


def main():
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = SimCoordinator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.fault = 'shutdown'
        node.command('stop')
        rclpy.spin_once(node, timeout_sec=.05)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
