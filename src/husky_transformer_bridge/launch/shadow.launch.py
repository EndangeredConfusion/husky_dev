"""Launch only the isolated phase-1 bridge and GPU worker."""
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration


def start(context):
    def value(name):
        return LaunchConfiguration(name).perform(context)

    domain = int(value('domain_id'))
    if not 1 <= domain <= 232:
        raise ValueError('Use an unused isolated domain in 1..232, never robot domain 0')
    gpu_python = str(Path(value('gpu_python')).expanduser())
    model_dir = str(Path(value('model_dir')).expanduser())
    socket_path = str(Path(value('socket_path')).expanduser())
    worker = ExecuteProcess(
        cmd=[gpu_python, '-m', 'husky_transformer_policy.policy_worker',
             '--model-dir', model_dir, '--socket', socket_path], output='screen')
    bridge = ExecuteProcess(
        cmd=['/usr/bin/python3', '-m', 'husky_transformer_bridge.ros_bridge',
             '--socket', socket_path, '--domain-id', str(domain)], output='screen')
    handlers = [RegisterEventHandler(OnProcessExit(
        target_action=process,
        on_exit=[EmitEvent(event=Shutdown(reason='Transformer interface process exited'))]))
        for process in (worker, bridge)]
    return handlers + [worker, bridge]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('gpu_python', default_value='~/transformer_deployment/venv/bin/python'),
        DeclareLaunchArgument('model_dir', default_value='~/transformer_deployment/model'),
        DeclareLaunchArgument('socket_path', default_value='/tmp/husky_policy_stage1.sock'),
        DeclareLaunchArgument('domain_id', default_value='42'),
        OpaqueFunction(function=start),
    ])
