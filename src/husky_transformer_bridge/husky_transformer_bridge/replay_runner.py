"""Paced recorded-trajectory replay, isolated ROS domain, no actuation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from husky_transformer_policy.transition_assembler import RecordedTransitions


class ReplayClient(Node):
    def __init__(self):
        super().__init__('phase4_replay', namespace='/transformer/shadow')
        self.pub = self.create_publisher(String, 'request', 10)
        self.sub = self.create_subscription(String, 'response', self.receive, 10)
        self.waiting = None
        self.reply = None
        self.unexpected_responses = 0

    def receive(self, msg):
        value = json.loads(msg.data)
        if value.get('client_id') == self.waiting:
            if self.reply is not None: self.unexpected_responses += 1
            self.reply = value
        else:
            self.unexpected_responses += 1

    def call(self, op, **payload):
        self.waiting, self.reply = str(uuid.uuid4()), None
        self.pub.publish(String(data=json.dumps(dict(client_id=self.waiting, op=op, **payload), allow_nan=False)))
        end = time.monotonic()+2
        while self.reply is None and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=.01)
        if self.reply is None: raise TimeoutError('ROS RPC deadline exceeded')
        if not self.reply.get('ok'): raise RuntimeError(self.reply.get('error'))
        return self.reply['result']

    def audit_graph(self):
        names = {name for name, _ in self.get_topic_names_and_types()}
        forbidden = names.intersection({'/cmd_vel','/platform/cmd_vel','/rl_goal','/rl_path','/rl_goal_index'})
        if forbidden: raise RuntimeError('Unexpected execution topics in isolated domain: '+repr(forbidden))
        nodes = self.get_node_names_and_namespaces()
        unknown = [(n,ns) for n,ns in nodes if n not in ('phase4_replay','policy_bridge')]
        if unknown: raise RuntimeError('Unexpected node in replay domain: '+repr(unknown))
        return dict(topics=sorted(names), nodes=nodes, forbidden_topics=[])


def resources():
    available = next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()
                     if x.startswith('MemAvailable:'))
    temps = []
    for p in Path('/sys/class/thermal').glob('thermal_zone*/temp'):
        try:
            t = int(p.read_text())/1000
            if 0 < t < 150: temps.append(t)
        except (OSError,ValueError): pass
    workers = []
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            cmd = p.read_bytes().replace(b'\0', b' ').decode(errors='replace')
            if ' -m husky_transformer_policy.policy_worker ' in cmd:
                status = (p.parent/'status').read_text().splitlines()
                rss = next(int(x.split()[1])*1024 for x in status if x.startswith('VmRSS:'))
                workers.append(dict(pid=int(p.parent.name),rss_bytes=rss))
        except (OSError,StopIteration): pass
    temp = max(temps,default=0)
    if available < 512*1024**2 or temp > 85: raise RuntimeError('Resource guard triggered')
    return dict(available_bytes=available, max_soc_c=temp, workers=workers)


def stats(values):
    x = np.asarray(values)
    return dict(p50_ms=float(np.percentile(x,50)),p95_ms=float(np.percentile(x,95)),
                p99_ms=float(np.percentile(x,99)),max_ms=float(x.max()),mean_ms=float(x.mean()))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--recording',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--count',type=int,default=6000)
    p.add_argument('--hz',type=float,default=10.)
    p.add_argument('--domain-id',type=int,default=42)
    a = p.parse_args()
    if not 1 <= a.domain_id <= 232 or not math.isfinite(a.hz) or a.hz <= 0:
        raise ValueError('Invalid isolated domain or frequency')
    data = np.load(a.recording,allow_pickle=False)
    records = RecordedTransitions(data,a.count)
    a.output.mkdir(parents=True,exist_ok=False)
    protocol = dict(recording=str(a.recording),recording_sha256=hashlib.sha256(a.recording.read_bytes()).hexdigest(),
                    count=a.count,hz=a.hz,domain=a.domain_id,precision='bf16',context=2400,
                    feedback_source='recorded teacher action and macro_reward; model predictions are shadow only',
                    timing='ROS observation publication through inference response and recorded feedback commit acknowledgment',
                    excludes='model startup/warmup; physical sensors, navigation and robot execution',
                    wall_clock_pacing=True,late_prediction_policy='logged as discarded, never executed; recorded actual feedback still committed')
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rclpy.init(domain_id=a.domain_id)
    node = ReplayClient()
    rows, telemetry = [], []
    summary = dict(status='FAIL',completed=0)
    started = None
    try:
        end = time.monotonic()+15
        while node.pub.get_subscription_count() != 1 and time.monotonic()<end:
            rclpy.spin_once(node,timeout_sec=.1)
        if node.pub.get_subscription_count()!=1: raise RuntimeError('Expected one isolated bridge')
        for _ in range(10): rclpy.spin_once(node,timeout_sec=.05)
        before = node.audit_graph()
        end = time.monotonic()+30
        while True:
            try:
                node.call('status')
                break
            except (RuntimeError,TimeoutError):
                if time.monotonic()>=end: raise
                time.sleep(.1)
        init = node.call('init_task',lambda_values=records.lambda_values)
        epoch = init['epoch']
        for _ in range(3): node.call('infer',**records.observation(0,epoch))
        init = node.call('init_task',lambda_values=records.lambda_values)
        epoch = init['epoch']
        period = 1/a.hz
        started = time.monotonic()
        with (a.output/'frames.jsonl').open('w') as log:
            for i in range(a.count):
                scheduled = started+i*period
                remaining = scheduled-time.monotonic()
                if remaining>0: time.sleep(remaining)
                tick = time.monotonic()
                result = node.call('infer',**records.observation(i,epoch))
                predicted_at = time.monotonic()
                assert result['history_version']==i and result['history_size']==min(i,2399)
                assert result['state_seq']==i and result['epoch']==epoch
                action = result['goal_index']
                assert type(action) is int and 0<=action<4
                assert result['goal_grid_xy']==[[1,1],[15,1],[1,9],[15,9]][action]
                assert len(result['logits'])==4 and all(math.isfinite(x) for x in result['logits'])
                assert result['next_episode_start']==bool(data['episode_decision_id'][i]==0)
                commit = node.call('commit',**records.feedback(i,epoch,result['ticket']))
                finished = time.monotonic()
                assert commit['history_version']==i+1 and commit['history_size']==min(i+1,2399)
                row = dict(index=i,context=min(i+1,2400),prediction_ms=(predicted_at-tick)*1000,
                           transaction_ms=(finished-tick)*1000,scheduled_to_commit_ms=(finished-scheduled)*1000,
                           scheduling_lag_ms=(tick-scheduled)*1000,model_ms=result['inference_ms'],
                           goal_index=action,actual_action=int(data['action'][i]),
                           prediction_discarded=predicted_at>scheduled+period,
                           deadline_missed=finished>scheduled+period,
                           peak_cuda_allocated_bytes=result['peak_cuda_allocated_bytes'],
                           peak_cuda_reserved_bytes=result['peak_cuda_reserved_bytes'])
                rows.append(row);log.write(json.dumps(row)+'\n')
                if i%10==0: telemetry.append(dict(index=i,**resources()))
                if (i+1)%300==0:
                    log.flush()
                    print(json.dumps(dict(completed=i+1,elapsed_seconds=finished-started,
                                          deadline_misses=sum(x['deadline_missed'] for x in rows))),flush=True)
                if i%300==0: node.audit_graph()
            remaining = started+a.count*period-time.monotonic()
            if remaining>0: time.sleep(remaining)
        after = node.audit_graph()
        final = node.call('status')
        assert final['history_version']==a.count
        assert node.unexpected_responses==0
        e2e = stats([r['transaction_ms'] for r in rows])
        misses = sum(r['deadline_missed'] for r in rows)
        summary = dict(status='PASS' if misses==0 and e2e['p95_ms']<=50 and e2e['max_ms']<100 else 'TIMING_FAIL',
                       correctness='PASS',completed=len(rows),duration_seconds=time.monotonic()-started,
                       transaction=e2e,prediction=stats([r['prediction_ms'] for r in rows]),
                       model=stats([r['model_ms'] for r in rows]),deadline_misses=misses,
                       unexpected_responses=node.unexpected_responses,graph_before=before,graph_after=after,
                       episode_ends=int(np.sum(data['terminated'][:a.count]|data['truncated'][:a.count])),
                       max_scheduling_lag_ms=max(r['scheduling_lag_ms'] for r in rows),
                       peak_cuda_allocated_bytes=max(r['peak_cuda_allocated_bytes'] for r in rows),
                       peak_cuda_reserved_bytes=max(r['peak_cuda_reserved_bytes'] for r in rows),
                       minimum_available_bytes=min(t['available_bytes'] for t in telemetry),
                       maximum_soc_c=max(t['max_soc_c'] for t in telemetry),
                       peak_worker_rss_bytes=max((w['rss_bytes'] for t in telemetry for w in t['workers']),default=0),
                       by_context={str(n):stats([r['transaction_ms'] for r in rows if r['context']==n])
                                   for n in (1,30,600,2400) if any(r['context']==n for r in rows)})
    except Exception as exc:
        summary.update(error=type(exc).__name__+': '+str(exc),completed=len(rows))
        raise
    finally:
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
        (a.output/'resources.json').write_text(json.dumps(telemetry,indent=2))
        print(json.dumps(summary),flush=True)
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
    if summary['status'] != 'PASS': raise SystemExit(2)

if __name__=='__main__': main()
