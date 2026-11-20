"""Single-owner execution/history transactions, independent of ROS.

The caller serializes operations and stops its executor whenever this session
pauses. A transport error never silently resets a task or fabricates feedback.
"""
import copy
import math
import uuid
from collections import OrderedDict


class ExecutionSession:
    def __init__(self, rpc):
        self.rpc = rpc
        self.task = None
        self.pending = None
        self.commit_request = None
        self.feedback = None
        self.completed = OrderedDict()
        self.paused = True
        self.reason = 'not initialized'

    def initialize(self, lambda_values):
        if self.pending is not None:
            raise RuntimeError('Stop and abort the active execution before changing task')
        self.paused = True
        self.task = self.rpc('init_task', lambda_values=list(lambda_values))
        self.commit_request = None
        self.feedback = None
        self.completed.clear()
        self.paused = False
        self.reason = ''
        return copy.deepcopy(self.task)

    def infer(self, cell, state_seq):
        if self.paused or self.task is None or self.pending is not None:
            raise RuntimeError('Session is not ready for a new decision')
        try:
            result = self.rpc('infer', epoch=self.task['epoch'],
                              history_version=self.task['history_version'],
                              agent_pos=list(cell), state_seq=state_seq)
        except Exception:
            self.paused, self.reason = True, 'inference transport or worker failure'
            raise
        self.pending = dict(decision_id=str(uuid.uuid4()), epoch=result['epoch'],
                            history_version=result['history_version'], ticket=result['ticket'],
                            state_seq=state_seq, start=list(cell), goal=result['goal_grid_xy'],
                            action=result['goal_index'], status='proposed')
        return copy.deepcopy(self.pending)

    def accepted(self, decision_id):
        if self.paused or self.pending is None or decision_id != self.pending['decision_id']:
            raise ValueError('Unexpected execution acknowledgment')
        self.pending['status'] = 'executing'

    def complete(self, feedback):
        """Commit exactly one completed simulated macro, retaining retry identity."""
        ident = feedback.get('decision_id')
        if ident in self.completed:
            original, result = self.completed[ident]
            if original != feedback:
                raise ValueError('Conflicting duplicate execution feedback')
            return copy.deepcopy(result)
        if self.paused or self.pending is None or ident != self.pending['decision_id']:
            raise ValueError('No matching active execution')
        p = self.pending
        if p['status'] != 'executing' or feedback.get('status') != 'completed':
            raise ValueError('Only acknowledged completed execution can update history')
        for key in ('epoch', 'history_version', 'state_seq', 'action'):
            if feedback.get(key) != p[key]:
                raise ValueError('Feedback identity mismatch: ' + key)
        steps = feedback.get('primitive_steps')
        if type(steps) is not int or not 1 <= steps <= 5:
            raise ValueError('Expected one to five executed primitive steps')
        reward = feedback.get('reward')
        if type(reward) not in (int, float) or not math.isfinite(reward):
            raise ValueError('Expected finite macro reward')
        if any(type(feedback.get(k)) is not bool for k in ('terminated', 'truncated')):
            raise ValueError('Explicit end flags required')
        if steps < 5 and not feedback['terminated']:
            raise ValueError('Incomplete nonterminal macro must not commit')
        self.feedback = copy.deepcopy(feedback)
        self.commit_request = dict(epoch=p['epoch'], history_version=p['history_version'],
                                   state_seq=p['state_seq'], ticket=p['ticket'],
                                   action=feedback['action'], reward=reward,
                                   terminated=feedback['terminated'], truncated=feedback['truncated'])
        return self._commit()

    def _commit(self):
        try:
            result = self.rpc('commit', **self.commit_request)
        except Exception:
            self.paused, self.reason = True, 'commit outcome uncertain; retry exact transaction'
            raise
        self.task = result
        ident = self.pending['decision_id']
        self.completed[ident] = (copy.deepcopy(self.feedback), copy.deepcopy(result))
        while len(self.completed) > 256:
            self.completed.popitem(last=False)
        self.pending = None
        self.commit_request = None
        self.feedback = None
        self.paused, self.reason = False, ''
        return copy.deepcopy(result)

    def retry_commit(self):
        if self.commit_request is None or not self.paused:
            raise RuntimeError('No uncertain commit to retry')
        # Same payload/ticket: the worker's ledger resolves lost acknowledgments.
        return self._commit()

    def abort(self, reason):
        self.paused, self.reason = True, str(reason)
        if self.commit_request is not None:
            raise RuntimeError('Resolve uncertain commit before abort/reset')
        if self.task is not None:
            self.rpc('abort', epoch=self.task['epoch'])
        self.pending = None

    def reset_after_worker_loss(self):
        """Explicit recovery only after the external executor is confirmed stopped."""
        self.paused = True
        self.reason = 'worker lost; task and simulation episode must be reinitialized'
        self.pending = self.task = self.commit_request = self.feedback = None
        self.completed.clear()
