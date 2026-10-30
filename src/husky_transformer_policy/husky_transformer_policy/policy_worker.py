"""Single-owner GPU history worker. Never publishes robot commands."""
import argparse
import hashlib
import os
from pathlib import Path
import socket
import signal
import sys
import time
from husky_transformer_policy.protocol import VERSION, receive, send

from husky_transformer_policy.history_manager import HistoryManager


class Policy(HistoryManager):
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
        super().__init__(OnlineContext, np)
        # Prepare CUDA kernels before accepting requests; leave task/history empty.
        self.handle(dict(op='init_task',lambda_values=[7,2,2,2]))
        for _ in range(3):
            self.handle(dict(op='infer',epoch=self.epoch,history_version=0,agent_pos=[4,3]))
        self.reset()

    def status(self):
        return dict(**super().status(), model_hash=self.digest, precision='bf16', context=2400,
                    peak_cuda_allocated_bytes=self.torch.cuda.max_memory_allocated(),
                    peak_cuda_reserved_bytes=self.torch.cuda.max_memory_reserved())

    def handle(self, q):
        op = q['op']
        if op == 'status': return self.status()
        if op == 'init_task': return self.configure(q['lambda_values'])
        self.require_epoch(q)
        if op == 'reset': return self.reset()
        if op == 'commit': return self.commit(q)
        if op == 'cancel': return self.cancel(q)
        if op == 'abort': return self.abort(q)
        if op != 'infer': raise ValueError('Unknown operation')
        start = time.perf_counter()
        snapshot, obs = self.prepare(q)
        batch, last = self.pack([snapshot], 'cuda')
        with self.torch.inference_mode(), self.torch.autocast('cuda', dtype=self.torch.bfloat16):
            scores = self.model(batch)[0,last[0]].float().cpu()
        if not self.torch.isfinite(scores).all(): raise ValueError('Nonfinite logits')
        action = int(scores.argmax())
        ticket = self.issue(obs, q.get('state_seq'))
        return dict(**self.status(), ticket=ticket, state_seq=q.get('state_seq'),
                    goal_index=action, goal_grid_xy=[(1,1),(15,1),(1,9),(15,9)][action],
                    logits=scores.tolist(), inference_ms=(time.perf_counter()-start)*1000)

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
