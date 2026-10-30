"""Single-owner GPU history worker. Never publishes robot commands."""
import argparse
import copy
import hashlib
import math
import os
from pathlib import Path
import socket
import signal
import sys
import time
import uuid
from husky_transformer_policy.protocol import VERSION, receive, send

class Policy:
    def __init__(self, model_dir):
        import numpy as np
        import torch
        sys.path.insert(0, str(model_dir))
        from ad_online import OnlineContext, pack_contexts
        from ad_transformer import ADTransformer, ModelConfig
        self.np, self.torch = np, torch
        self.Context, self.pack = OnlineContext, pack_contexts
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('CUDA BF16 required')
        path = model_dir/'model.pt'
        self.digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if self.digest != '012765a00264a5b64cc14ef2cd3227a2d2f4d8d03463de4f79a1969c87a53f50':
            raise ValueError('Unexpected checkpoint')
        with torch.serialization.safe_globals([np.core.multiarray.scalar, np.dtype,
                type(np.dtype('float64')), type(np.dtype('float32'))]):
            saved = torch.load(path, map_location='cpu', weights_only=True)
        self.model = ADTransformer(ModelConfig(**saved['config'])).cuda().eval()
        self.model.load_state_dict(saved['model'], strict=True)
        self.model.requires_grad_(False)
        self.reset()
        # Prepare CUDA kernels before accepting requests; leave task/history empty.
        self.handle(dict(op='init_task',lambda_values=[7,2,2,2]))
        for _ in range(3):
            self.handle(dict(op='infer',epoch=self.epoch,history_version=0,agent_pos=[4,3]))
        self.reset()

    def reset(self):
        self.epoch = str(uuid.uuid4())
        self.ctx = self.Context(2400)
        self.lam = None
        self.next_episode = True
        self.last_commit = None
        self.tickets = {}
        return self.status()

    def status(self):
        return dict(epoch=self.epoch, history_version=self.ctx.decisions,
                    history_size=self.ctx.size, task_ready=self.lam is not None,
                    model_hash=self.digest, precision='bf16', context=2400)

    def handle(self, q):
        op = q['op']
        if op == 'status': return self.status()
        if op == 'init_task':
            lam = q['lambda_values']
            if not isinstance(lam, list) or len(lam) != 4 or any(
                    type(x) not in (int,float) or not math.isfinite(x) or not 0<=x<=10 for x in lam):
                raise ValueError('Expected four finite raw lambda values in [0,10]')
            self.reset()
            self.lam = self.np.asarray(lam, dtype=self.np.float32)
            return self.status()
        if q.get('epoch') != self.epoch: raise ValueError('Stale task epoch')
        if op == 'reset': return self.reset()
        if self.lam is None: raise ValueError('Task not configured')
        if op == 'commit' and self.last_commit is not None and q == self.last_commit[0]:
            return self.last_commit[1]
        if q.get('history_version') != self.ctx.decisions:
            raise ValueError('Stale history version')
        if op == 'infer':
            pos = q['agent_pos']
            if not isinstance(pos,list) or len(pos)!=2 or any(type(x) is not int for x in pos):
                raise ValueError('Expected raw integer grid position')
            if not (0<=pos[0]<=16 and 0<=pos[1]<=10): raise ValueError('Position outside grid')
            ctx = copy.deepcopy(self.ctx)
            obs = dict(agent_pos=self.np.asarray(pos,dtype=self.np.float32),lambda_values=self.lam.copy())
            ctx.observe(obs, self.next_episode)
            start = time.perf_counter()
            batch,last = self.pack([ctx], 'cuda')
            with self.torch.inference_mode(), self.torch.autocast('cuda',dtype=self.torch.bfloat16):
                scores = self.model(batch)[0,last[0]].float().cpu()
            if not self.torch.isfinite(scores).all(): raise ValueError('Nonfinite logits')
            action = int(scores.argmax())
            ticket = str(uuid.uuid4())
            self.tickets[ticket] = obs
            while len(self.tickets)>32: self.tickets.pop(next(iter(self.tickets)))
            return dict(**self.status(), ticket=ticket, state_seq=q.get('state_seq'),
                        goal_index=action, goal_grid_xy=[(1,1),(15,1),(1,9),(15,9)][action],
                        logits=scores.tolist(), inference_ms=(time.perf_counter()-start)*1000)
        if op == 'commit':
            ticket=q['ticket']; action=q['action']; reward=q['reward']
            if ticket not in self.tickets: raise ValueError('Unknown or expired decision ticket')
            if type(action) is not int or not 0<=action<4: raise ValueError('Invalid actual action')
            if type(reward) not in (int,float) or not math.isfinite(reward): raise ValueError('Invalid raw reward')
            if abs(reward/10.)>self.np.finfo(self.np.float32).max:
                raise ValueError('Reward exceeds representable model input')
            if any(type(q.get(k)) is not bool for k in ['terminated','truncated']):
                raise ValueError('Explicit boolean end flags required')
            self.ctx.observe(self.tickets[ticket],self.next_episode)
            self.ctx.commit(action,reward,q['terminated'],q['truncated'])
            self.next_episode=q['terminated'] or q['truncated']
            self.tickets.clear()
            result=self.status()
            self.last_commit=(copy.deepcopy(q),result)
            return result
        raise ValueError('Unknown operation')

def main():
    def terminate(signum,frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,terminate)
    signal.signal(signal.SIGINT,terminate)
    p=argparse.ArgumentParser()
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--socket',type=Path,required=True)
    a=p.parse_args()
    policy=Policy(a.model_dir)
    os.umask(0o077)
    with socket.socket(socket.AF_UNIX) as server:
        server.bind(str(a.socket))  # Refuse to overwrite an existing socket.
        try:
            server.listen(4)
            print('READY',flush=True)
            while True:
                conn,_=server.accept()
                with conn:
                    conn.settimeout(1.0)
                    q={}
                    try:
                        q=receive(conn)
                        if q.get('version')!=VERSION: raise ValueError('Protocol version mismatch')
                        if not isinstance(q.get('request_id'),str): raise ValueError('Missing request ID')
                        body={k:v for k,v in q.items() if k not in ('version','request_id')}
                        result=policy.handle(body)
                        reply=dict(ok=True,result=result)
                    except Exception as exc:
                        reply=dict(ok=False,error=type(exc).__name__+': '+str(exc))
                    try: send(conn,dict(version=VERSION,request_id=q.get('request_id'),**reply))
                    except (OSError,ValueError): pass
        finally: a.socket.unlink(missing_ok=True)

if __name__=='__main__':
    try: main()
    except KeyboardInterrupt: pass
