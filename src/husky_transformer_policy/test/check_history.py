"""Check full recorded history and selected BF16 outputs against original code."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from husky_transformer_policy.policy_worker import Policy
from husky_transformer_policy.transition_assembler import RecordedTransitions

p = argparse.ArgumentParser()
p.add_argument('--model-dir', type=Path, required=True)
p.add_argument('--recording', type=Path, required=True)
a = p.parse_args()
data = np.load(a.recording, allow_pickle=False)
records = RecordedTransitions(data, 6000)
w = Policy(a.model_dir)
epoch = w.handle(dict(op='init_task', lambda_values=records.lambda_values))['epoch']
reference = w.Context(2400)
selected = {0, 29, 299, 300, 599, 600, 2399, 2400, 2999, 5999}
gpu_checks = []
recent_commits = []
for i in range(6000):
    episode_start = bool(data['episode_decision_id'][i] == 0)
    if episode_start: reference.begin_episode()
    reference.observe(dict(agent_pos=data['obs_position'][i], lambda_values=data['obs_lambda'][i]), episode_start)
    q = records.observation(i, epoch)
    snapshot, obs = w.prepare(q)
    assert w.next_episode == episode_start
    if i in selected:
        actual, last = w.pack([snapshot], 'cpu')
        expected, ref_last = w.pack([reference], 'cpu')
        for key in expected: assert w.torch.equal(actual[key], expected[key]), (i, key)
        assert w.torch.equal(last, ref_last)
        result = w.handle(dict(op='infer', **q))
        batch, last = w.pack([reference], 'cuda')
        with w.torch.inference_mode(), w.torch.autocast('cuda', dtype=w.torch.bfloat16):
            direct = w.model(batch)[0,last[0]].float().cpu().tolist()
        assert result['logits'] == direct, i
        gpu_checks.append(dict(index=i, context_length=reference.size+1, exact_logits=True))
    ticket = w.issue(obs, i)
    feedback = dict(op='commit', **records.feedback(i, epoch, ticket))
    reply = w.handle(feedback)
    assert w.handle(feedback) == reply
    if recent_commits:
        previous, ack = recent_commits[-1]
        assert w.handle(previous) == ack
    recent_commits = [(feedback, copy.deepcopy(reply))]
    reference.commit(feedback['action'], feedback['reward'], feedback['terminated'], feedback['truncated'])
    assert w.ctx.pending is None and w.ctx.decisions == reference.decisions
    for key in ('obs', 'actions', 'feedback'):
        np.testing.assert_array_equal(getattr(w.ctx, key), getattr(reference, key))
    assert w.ctx.size == reference.size == min(i+1, 2399)

def reject(q):
    before = (w.ctx.decisions, w.ctx.size, w.ctx.obs.copy(), w.ctx.feedback.copy())
    try: w.handle(q)
    except (ValueError, KeyError): pass
    else: raise AssertionError('Invalid transaction accepted')
    assert (w.ctx.decisions, w.ctx.size) == before[:2]
    np.testing.assert_array_equal(w.ctx.obs, before[2])
    np.testing.assert_array_equal(w.ctx.feedback, before[3])

reject(dict(recent_commits[0][0], reward=123.))
q = dict(op='infer', epoch=epoch, history_version=6000, agent_pos=[4,3], state_seq=6000)
snapshot, obs = w.prepare(q); ticket = w.issue(obs, 6000)
feedback = dict(op='commit', epoch=epoch, history_version=6000, ticket=ticket,
                state_seq=6000, action=0, reward=0., terminated=False, truncated=False)
reject(dict(feedback, state_seq=6001)); reject(dict(feedback, reward=float('nan')))
reject(dict(feedback, history_version=5999))
w.handle(dict(op='cancel', epoch=epoch, history_version=6000, ticket=ticket))
reject(feedback)
w.handle(dict(op='abort', epoch=epoch)); reject(q)
new = w.handle(dict(op='init_task', lambda_values=[1,1,1,1]))
assert new['epoch'] != epoch and new['history_size'] == 0
reject(q)
# A fresh history owner models worker restart: no old task/ticket survives.
from husky_transformer_policy.history_manager import HistoryManager
fresh = HistoryManager(w.Context, np)
assert fresh.epoch != w.epoch and not fresh.status()['task_ready']
try: fresh.prepare(q)
except ValueError: pass
else: raise AssertionError('Restart accepted old task')
summary = dict(status='PASS',records=6000,episode_ends=int(np.sum(data['terminated'][:6000]|data['truncated'][:6000])),
               exact_history_arrays=True, selected_gpu_checks=gpu_checks,
               duplicates_conflicts_cancel_abort_reset_restart='PASS')
Path('history_result.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary))
