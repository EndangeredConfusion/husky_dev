"""Isolated GPS -> Transformer -> A* bench. Never launches a controller."""
import os
from pathlib import Path
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
                            OpaqueFunction, RegisterEventHandler, EmitEvent)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def start(context):
    domain = int(os.environ.get('ROS_DOMAIN_ID', '0'))
    if not 1 <= domain <= 232:
        raise ValueError('Set an unused nonzero ROS_DOMAIN_ID before launching the bench')
    def value(name):
        return str(Path(LaunchConfiguration(name).perform(context)).expanduser())
    worker = ExecuteProcess(cmd=[value('gpu_python'), '-m',
        'husky_transformer_policy.policy_worker', '--model-dir', value('model_dir'),
        '--socket', value('socket_path')], output='screen')
    pipeline = IncludeLaunchDescription(PythonLaunchDescriptionSource(
        get_package_share_directory('husky_rl') + '/launch/rl.launch.py'),
        launch_arguments={'policy': 'transformer', 'enable_controller': 'false',
                          'socket_path': value('socket_path')}.items())
    return [RegisterEventHandler(OnProcessExit(target_action=worker,
                on_exit=[EmitEvent(event=Shutdown(reason='GPU worker exited'))])), worker, pipeline]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('gpu_python', default_value='~/transformer_deployment/venv/bin/python'),
        DeclareLaunchArgument('model_dir', default_value='~/transformer_deployment/model'),
        DeclareLaunchArgument('socket_path', default_value='/tmp/husky_policy_candidates.sock'),
        OpaqueFunction(function=start)])
