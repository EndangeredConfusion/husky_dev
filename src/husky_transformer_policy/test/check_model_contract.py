"""Run with GPU venv: python test_model_contract.py --model-dir ../model."""
import argparse
import json
from pathlib import Path
from husky_transformer_policy.policy_worker import Policy

p=argparse.ArgumentParser();p.add_argument('--model-dir',type=Path,required=True)
a=p.parse_args();w=Policy(a.model_dir)
s=w.handle(dict(op='init_task',lambda_values=[7,2,2,2]))
q=dict(op='infer',epoch=s['epoch'],history_version=0,agent_pos=[4,3],state_seq=1)
x=w.handle(q);y=w.handle(q)
assert x['goal_index']==y['goal_index']==0 and x['logits']==y['logits']
assert w.ctx.size==w.ctx.decisions==0 and w.ctx.pending is None
# Independent direct online-input path, bypassing IPC and worker inference method.
c=w.Context(2400)
c.observe(dict(agent_pos=w.np.array([4,3]),lambda_values=w.np.array([7,2,2,2])),True)
batch,last=w.pack([c],'cuda')
w.np.testing.assert_allclose(batch['observations'][0,0].cpu().numpy(),
    [.25,.3,.7,.2,.2,.2,1,1,1],rtol=1e-6)
with w.torch.inference_mode(),w.torch.autocast('cuda',dtype=w.torch.bfloat16):
    direct=w.model(batch)[0,last[0]].float().cpu().tolist()
assert direct==x['logits']
commit=dict(op='commit',epoch=s['epoch'],history_version=0,ticket=x['ticket'],
            action=2,reward=5.,terminated=False,truncated=True)
first=w.handle(commit);second=w.handle(commit)
assert first==second and w.ctx.decisions==1 and w.ctx.size==1
assert w.ctx.actions[0]==2  # Actual action, not predicted action 0.
w.np.testing.assert_allclose(w.ctx.feedback[0],[.5,0,1])
bad_requests=[q,dict(q,epoch='stale'),dict(q,history_version=1,agent_pos=[float('nan'),3])]
for bad in bad_requests:
    try:w.handle(bad)
    except ValueError:pass
    else:raise AssertionError('Invalid input accepted')
next_result=w.handle(dict(q,history_version=1))
assert next_result['history_size']==1
old=s['epoch'];s=w.handle(dict(op='init_task',lambda_values=[1,1,1,1]))
assert s['epoch']!=old and s['history_size']==0 and s['history_version']==0
result=dict(status='PASS',raw_normalization='PASS',direct_bf16_equal=True,
    repeat_inference_pure=True,actual_action_commit=True,commit_idempotent=True,
    stale_invalid_inputs_rejected=True,task_reset=True,logits=x['logits'])
Path('model_contract_result.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
