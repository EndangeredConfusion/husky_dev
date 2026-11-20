import copy
import unittest
from husky_transformer_bridge.execution_session import ExecutionSession


class Worker:
    def __init__(self):
        self.version = 0
        self.ledger = {}
        self.requests = []
        self.lose_ack = False

    def __call__(self, op, **q):
        self.requests.append((op, copy.deepcopy(q)))
        if op == 'init_task':
            self.version = 0
            return dict(epoch='task', history_version=0)
        if op == 'infer':
            return dict(epoch='task', history_version=self.version, ticket='ticket',
                        goal_index=2, goal_grid_xy=[1, 9])
        if op == 'commit':
            if q['ticket'] not in self.ledger:
                self.version += 1
                self.ledger[q['ticket']] = dict(epoch='task', history_version=self.version)
            if self.lose_ack:
                self.lose_ack = False
                raise TimeoutError('ack lost after server committed')
            return self.ledger[q['ticket']]
        if op == 'abort':
            return {}
        raise AssertionError(op)


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.worker = Worker()
        self.session = ExecutionSession(self.worker)
        self.session.initialize([7, 2, 2, 2])
        self.decision = self.session.infer([4, 3], 7)
        self.feedback = dict(self.decision, status='completed', primitive_steps=5,
                             reward=1.25, terminated=False, truncated=False)

    def test_requires_execution_ack(self):
        with self.assertRaises(ValueError):
            self.session.complete(self.feedback)
        self.assertEqual(self.worker.version, 0)

    def test_commit_and_duplicate_are_exactly_once(self):
        self.session.accepted(self.decision['decision_id'])
        result = self.session.complete(self.feedback)
        self.assertEqual(self.session.complete(self.feedback), result)
        self.assertEqual(self.worker.version, 1)
        with self.assertRaises(ValueError):
            self.session.complete(dict(self.feedback, reward=2.0))

    def test_lost_ack_retries_identical_commit(self):
        self.session.accepted(self.decision['decision_id'])
        self.worker.lose_ack = True
        with self.assertRaises(TimeoutError):
            self.session.complete(self.feedback)
        self.assertTrue(self.session.paused)
        with self.assertRaises(RuntimeError):
            self.session.infer([4, 2], 8)
        self.session.retry_commit()
        commits = [q for op, q in self.worker.requests if op == 'commit']
        self.assertEqual(commits[0], commits[1])
        self.assertEqual(self.worker.version, 1)

    def test_rejects_stale_and_partial_feedback(self):
        self.session.accepted(self.decision['decision_id'])
        for update in ({'epoch': 'old'}, {'state_seq': 6}, {'action': 1},
                       {'primitive_steps': 3}, {'reward': float('nan')}):
            with self.assertRaises(ValueError):
                self.session.complete(dict(self.feedback, **update))
        self.assertEqual(self.worker.version, 0)

    def test_abort_blocks_old_feedback_and_requires_explicit_init(self):
        with self.assertRaises(RuntimeError):
            self.session.initialize([1, 1, 1, 1])
        self.session.abort('pose lost')
        with self.assertRaises(ValueError):
            self.session.complete(self.feedback)
        with self.assertRaises(RuntimeError):
            self.session.infer([4, 3], 8)
        self.session.initialize([1, 1, 1, 1])
        self.assertFalse(self.session.paused)


if __name__ == '__main__':
    unittest.main()
