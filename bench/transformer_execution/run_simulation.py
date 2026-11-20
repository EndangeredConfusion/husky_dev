"""Reproducible Gazebo/GPU execution and injected-fault integration checks."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid
import xml.etree.ElementTree as ET
from husky_transformer_policy.protocol import request


def records(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text(errors='replace').splitlines():
        try:
            result.append(json.loads(line))
        except ValueError:
            pass
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scenario', choices=['nominal', 'pose_dropout', 'worker_loss', 'heartbeat_loss'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu-python', default=str(Path.home()/'transformer_deployment/venv/bin/python'))
    p.add_argument('--model-dir', default=str(Path.home()/'transformer_deployment/model'))
    a = p.parse_args()
    if not 1 <= int(os.environ.get('ROS_DOMAIN_ID', '0')) <= 232:
        raise RuntimeError('Use an unused nonzero ROS domain')
    if os.environ.get('GZ_PARTITION') != 'husky_transformer_stage67':
        raise RuntimeError('Set GZ_PARTITION=husky_transformer_stage67')
    if a.output.exists():
        raise RuntimeError('Choose a fresh output directory; existing evidence is preserved')
    a.output.mkdir(parents=True)
    os.environ['GZ_PARTITION'] = 'husky_transformer_stage67_' + uuid.uuid4().hex
    world = a.output/'execution.sdf'
    tree = ET.parse('/opt/ros/jazzy/opt/gz_sim_vendor/share/gz/gz-sim8/worlds/diff_drive.sdf')
    root = tree.getroot().find('world')
    root.set('name', 'transformer_execution')
    for model in list(root.findall('model')):
        if model.get('name') not in ('ground_plane', 'vehicle_blue'):
            root.remove(model)
    # The bundled demo defaults to 1 Hz, unsuitable for a 1-second stale-pose gate.
    root.find("model[@name='vehicle_blue']/plugin/odom_publish_frequency").text = '50'
    tree.write(world, encoding='unicode')
    procs, handles = {}, []
    socket = str(a.output/'policy.sock')

    def start(name, command):
        handle = (a.output/(name+'.log')).open('w')
        handles.append(handle)
        procs[name] = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                       start_new_session=True)

    def stop(name, sig=signal.SIGINT):
        proc = procs[name]
        if proc.poll() is None:
            os.killpg(proc.pid, sig)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=3)

    def wait_for(predicate, seconds=20):
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.1)
        raise TimeoutError('Condition not met; inspect logs in '+str(a.output))

    def probe():
        return records(a.output/'probe.log')

    try:
        start('gazebo', ['gz', 'sim', '-s', '-r', str(world)])
        bridge = '/opt/ros/jazzy/lib/ros_gz_bridge/parameter_bridge'
        start('command_bridge', [bridge, '/model/vehicle_blue/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
              '--ros-args', '-r', '/model/vehicle_blue/cmd_vel:=/transformer_sim/cmd_vel'])
        start('pose_bridge', [bridge, '/model/vehicle_blue/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry',
              '--ros-args', '-r', '/model/vehicle_blue/odometry:=/transformer_sim/odom'])
        start('worker', [a.gpu_python, '-m', 'husky_transformer_policy.policy_worker',
                        '--model-dir', a.model_dir, '--socket', socket])
        start('probe', ['/usr/bin/python3', str(Path(__file__).with_name('sim_probe.py'))])
        start('executor', ['/usr/bin/python3', '-m', 'husky_transformer_bridge.sim_executor'])
        wait_for(lambda: 'READY' in (a.output/'worker.log').read_text())
        wait_for(lambda: any(r['topic']=='state' and r['data']['pose_seq'] > 0 for r in probe()))
        start('coordinator', ['/usr/bin/python3', '-m', 'husky_transformer_bridge.sim_coordinator',
              '--ros-args', '-p', 'socket_path:='+socket])
        if a.scenario == 'nominal':
            def finished():
                audit = [r['data'] for r in probe() if r['topic']=='audit']
                failures = [r for r in audit if r['event']=='paused']
                if failures:
                    raise RuntimeError('Nominal run paused: ' + str(failures[-1]))
                return any(r['event']=='run_complete' for r in audit)
            wait_for(finished, 360)
            committed = [r['data'] for r in probe() if r['topic']=='audit' and r['data']['event']=='history_committed']
            assert [r['worker']['history_version'] for r in committed] == list(range(1,7))
            assert any(r['feedback']['reward'] > 0 for r in committed)
            assert all(r['feedback']['primitive_steps']==5 for r in committed)
            outcome = dict(commits=6, worker=request(socket, 'status'))
        else:
            wait_for(lambda: any(r['topic']=='cmd_vel' and abs(r['data']['omega']) > .1 for r in probe()), 30)
            assert not any(r['topic']=='feedback' and r['data']['status']=='failed' for r in probe()), 'Failure preceded injection'
            injected_at = time.monotonic()
            target = dict(pose_dropout='pose_bridge', worker_loss='worker', heartbeat_loss='coordinator')[a.scenario]
            stop(target, signal.SIGKILL if a.scenario=='heartbeat_loss' else signal.SIGINT)
            wait_for(lambda: any(r['topic']=='feedback' and r['data']['status']=='failed'
                                and r['wall_time']>=injected_at for r in probe()), 10)
            failed_at = min(r['wall_time'] for r in probe() if r['topic']=='feedback'
                            and r['data']['status']=='failed' and r['wall_time']>=injected_at)
            # Ignore ordinary zero commands between primitives; associate the stop
            # with the actual failure and ensure execution does not resume.
            time.sleep(.6)
            zeros = [r for r in probe() if r['topic']=='cmd_vel' and r['wall_time']>=failed_at-.2
                     and r['data']['v']==0 and r['data']['omega']==0]
            assert zeros, 'No stop command observed'
            latency = failed_at-injected_at
            assert latency < 5, latency
            assert not any(r['topic']=='cmd_vel' and r['wall_time']>failed_at
                           and (r['data']['v']!=0 or r['data']['omega']!=0) for r in probe())
            assert not any(r['topic']=='audit' and r['data']['event']=='history_committed' for r in probe())
            outcome = dict(stop_latency_s=latency, commits=0)
            if a.scenario != 'worker_loss':
                outcome['worker'] = request(socket, 'status')
                assert outcome['worker']['history_size']==0
        result = dict(passed=True, scenario=a.scenario, partition=os.environ['GZ_PARTITION'], **outcome)
        (a.output/'result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    finally:
        for name in ('coordinator', 'executor', 'worker', 'pose_bridge', 'command_bridge', 'gazebo', 'probe'):
            if name in procs:
                stop(name)
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    main()
