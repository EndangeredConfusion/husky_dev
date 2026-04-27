"""
Launch the RL policy node and lambda node together.

Fixed lambda example:
    ros2 launch husky_rl rl.launch.py adaptive:=false fixed_lambda:="[1,2,2,9]"

Adaptive lambda example:
    ros2 launch husky_rl rl.launch.py adaptive:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    adaptive_arg = DeclareLaunchArgument(
        'adaptive', default_value='false',
        description='true = Algo 2 adaptive lambda, false = fixed lambda')

    fixed_lambda_arg = DeclareLaunchArgument(
        'fixed_lambda', default_value='[1, 1, 1, 1]',
        description='Fixed lambda values, e.g. "[1,1,1,10]" (used only when adaptive:=false)')

    lambda_node = Node(
        package='husky_rl',
        executable='lambda_node',
        name='lambda_node',
        parameters=[{
            'adaptive':     LaunchConfiguration('adaptive'),
            'fixed_lambda': LaunchConfiguration('fixed_lambda'),
        }],
        output='screen',
    )

    rl_policy_node = Node(
        package='husky_rl',
        executable='rl_policy',
        name='rl_policy_node',
        output='screen',
    )

    return LaunchDescription([
        adaptive_arg,
        fixed_lambda_arg,
        lambda_node,
        rl_policy_node,
    ])
