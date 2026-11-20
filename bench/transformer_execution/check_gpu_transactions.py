"""Fault-inject real worker transactions using an already completed Gazebo trace.

Run only after the simulation coordinator is stopped. This is recorded-feedback
validation, not additional physical execution. Uses a dedicated task epoch.
"""
import argparse
import copy
import json
from pathlib import Path
from husky_transformer_bridge.execution_session import ExecutionSession
from husky_transformer_policy.protocol import request


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--socket', default='/tmp/husky_policy_execution.sock')
    p.add_argument('--simulation-log', type=Path, required=True)
    args = p.parse_args()
    events = [json.loads(line[line.index('{'):]) for line in args.simulation_log.read_text().splitlines()
              if '{' in line]
    first = next(e for e in events if e['event'] == 'history_committed')
    recorded = first['feedback']
    drop_ack = [True]
    commits = []

    def rpc(op, **payload):
        result = request(args.socket, op, **payload)
        if op == 'commit':
            commits.append(copy.deepcopy(payload))
            if drop_ack[0]:
                drop_ack[0] = False
                raise TimeoutError('Injected lost acknowledgment AFTER real GPU worker commit')
        return result

    s = ExecutionSession(rpc)
    s.initialize([7, 2, 2, 2])
    decision = s.infer(recorded['trace'][0]['start'], 1)
    assert decision['action'] == recorded['action'], 'Replay action must match actually executed trace'
    s.accepted(decision['decision_id'])
    feedback = dict(recorded, **{k: decision[k] for k in
                    ('decision_id', 'epoch', 'history_version', 'state_seq', 'action')})
    try:
        s.complete(feedback)
    except TimeoutError:
        pass
    else:
        raise AssertionError('Lost acknowledgment was not injected')
    assert s.paused and request(args.socket, 'status')['history_version'] == 1
    recovered = s.retry_commit()
    assert recovered['history_version'] == 1 and commits[0] == commits[1]
    assert s.complete(feedback) == recovered
    try:
        s.complete(dict(feedback, reward=feedback['reward']+1))
    except ValueError:
        pass
    else:
        raise AssertionError('Conflicting duplicate accepted')
    assert request(args.socket, 'status')['history_version'] == 1
    pending = s.infer(recorded['end_cell'], 2)
    s.accepted(pending['decision_id'])
    s.abort('injected localization loss before macro completion')
    status = request(args.socket, 'status')
    assert status['paused'] and status['history_version'] == 1
    old_epoch = status['epoch']
    s.initialize([1, 7, 2, 2])
    assert s.task['epoch'] != old_epoch and s.task['history_version'] == 0
    try:
        s.complete(feedback)
    except ValueError:
        pass
    else:
        raise AssertionError('Previous task feedback accepted')
    print(json.dumps(dict(passed=True, checks=['lost_commit_ack_exact_retry',
        'duplicate_feedback_once', 'conflicting_duplicate_rejected',
        'abort_preserves_committed_history', 'task_switch_rejects_old_feedback'])))


if __name__ == '__main__':
    main()
