"""Transactional raw-input history, independent of GPU inference and ROS."""
import copy
import math
import uuid


class HistoryManager:
    def __init__(self, context_type, np):
        self.Context, self.np = context_type, np
        self.reset()

    def reset(self):
        self.epoch = str(uuid.uuid4())
        self.ctx = self.Context(2400)
        self.lam = None
        self.next_episode = True
        self.paused = False
        self.tickets = {}
        self.commits = {}
        return self.status()

    def status(self):
        return dict(epoch=self.epoch, history_version=self.ctx.decisions,
                    history_size=self.ctx.size, task_ready=self.lam is not None and not self.paused,
                    paused=self.paused, next_episode_start=self.next_episode)

    def configure(self, values):
        if not isinstance(values, list) or len(values) != 4 or any(
                type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 10 for x in values):
            raise ValueError('Expected four finite raw lambda values in [0,10]')
        self.reset()
        self.lam = self.np.asarray(values, dtype=self.np.float32)
        return self.status()

    def require_epoch(self, q):
        if q.get('epoch') != self.epoch:
            raise ValueError('Stale task epoch')

    def require_current(self, q):
        self.require_epoch(q)
        if self.lam is None or self.paused:
            raise ValueError('Task not ready; initialize a new task after uncertain execution')
        if type(q.get('history_version')) is not int or q['history_version'] != self.ctx.decisions:
            raise ValueError('Stale or invalid history version')

    def prepare(self, q):
        self.require_current(q)
        pos = q['agent_pos']
        if not isinstance(pos, list) or len(pos) != 2 or any(type(x) is not int for x in pos):
            raise ValueError('Expected raw integer grid position')
        if not (0 <= pos[0] <= 16 and 0 <= pos[1] <= 10):
            raise ValueError('Position outside grid')
        if q.get('state_seq') is not None and (type(q['state_seq']) is not int or q['state_seq'] < 0):
            raise ValueError('Invalid observation sequence')
        obs = dict(agent_pos=self.np.asarray(pos, dtype=self.np.float32), lambda_values=self.lam.copy())
        snapshot = copy.deepcopy(self.ctx)
        snapshot.observe(obs, self.next_episode)
        return snapshot, obs

    def issue(self, obs, state_seq):
        ticket = str(uuid.uuid4())
        self.tickets[ticket] = (copy.deepcopy(obs), self.next_episode, state_seq)
        while len(self.tickets) > 32:
            self.tickets.pop(next(iter(self.tickets)))
        return ticket

    def commit(self, q):
        self.require_epoch(q)
        ticket = q['ticket']
        if ticket in self.commits:
            original, result = self.commits[ticket]
            if q != original:
                raise ValueError('Conflicting duplicate feedback')
            return copy.deepcopy(result)  # Original acknowledgment, not current status.
        self.require_current(q)
        if ticket not in self.tickets:
            raise ValueError('Unknown, cancelled or expired decision ticket')
        action, reward = q['action'], q['reward']
        if type(action) is not int or not 0 <= action < 4:
            raise ValueError('Invalid actual action')
        if type(reward) not in (int, float) or not math.isfinite(reward):
            raise ValueError('Invalid raw reward')
        if abs(reward / 10.) > self.np.finfo(self.np.float32).max:
            raise ValueError('Reward exceeds representable model input')
        if any(type(q.get(k)) is not bool for k in ('terminated', 'truncated')):
            raise ValueError('Explicit boolean end flags required')
        obs, episode_start, seq = self.tickets[ticket]
        if 'state_seq' in q and q['state_seq'] != seq:
            raise ValueError('Feedback observation sequence mismatch')
        candidate = copy.deepcopy(self.ctx)
        candidate.observe(obs, episode_start)
        candidate.commit(action, reward, q['terminated'], q['truncated'])
        self.ctx = candidate
        self.next_episode = q['terminated'] or q['truncated']
        self.tickets.clear()
        result = self.status()
        self.commits[ticket] = (copy.deepcopy(q), copy.deepcopy(result))
        while len(self.commits) > 256:
            self.commits.pop(next(iter(self.commits)))
        return result

    def cancel(self, q):
        """Discard a candidate which was never executed; no synthetic transition."""
        self.require_current(q)
        if q['ticket'] not in self.tickets:
            raise ValueError('Unknown candidate')
        del self.tickets[q['ticket']]
        return self.status()

    def abort(self, q):
        """Execution outcome is uncertain: block inference until explicit task reset."""
        self.require_epoch(q)
        self.tickets.clear()
        self.paused = True
        return self.status()
