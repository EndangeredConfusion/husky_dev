"""Isolated recorded replay; stops all children after replay completion."""
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration


def start(context):
    def value(name): return LaunchConfiguration(name).perform(context)
    domain = int(value('domain_id'))
    if not 1 <= domain <= 232: raise ValueError('An isolated nonzero domain is required')
    def path(name): return str(Path(value(name)).expanduser())
    worker = ExecuteProcess(cmd=[path('gpu_python'), '-m', 'husky_transformer_policy.policy_worker',
                                  '--model-dir', path('model_dir'), '--socket', path('socket_path')], output='screen')
    bridge = ExecuteProcess(cmd=['/usr/bin/python3', '-m', 'husky_transformer_bridge.ros_bridge',
                                 '--socket', path('socket_path'), '--domain-id', str(domain)], output='screen')
    replay = ExecuteProcess(cmd=['/usr/bin/python3', '-m', 'husky_transformer_bridge.replay_runner',
                                 '--recording', path('recording'), '--output', path('output'),
                                 '--count', value('count'), '--hz', value('hz'), '--domain-id', str(domain)], output='screen')
    processes = [worker, bridge, replay]
    handlers = [RegisterEventHandler(OnProcessExit(target_action=p,
        on_exit=[EmitEvent(event=Shutdown(reason='Replay or interface process exited'))])) for p in processes]
    return handlers+processes


def generate_launch_description():
    defaults = dict(gpu_python='~/transformer_deployment/venv/bin/python',model_dir='~/transformer_deployment/model',
                    socket_path='/tmp/husky_policy_stage4.sock',domain_id='42',
                    recording='~/transformer_deployment/benchmarks/history_sample.npz',count='6000',hz='10')
    args = [DeclareLaunchArgument(name, default_value=value) for name,value in defaults.items()]
    args.append(DeclareLaunchArgument('output', description='New result directory outside the repository'))
    return LaunchDescription(args+[OpaqueFunction(function=start)])
