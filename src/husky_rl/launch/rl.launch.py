"""
Launch the EKF adapter and all three RL-side nodes.

Fixed lambda example:
    ros2 launch husky_rl rl.launch.py adaptive:=false fixed_lambda:="[1,2,2,9]"

Adaptive lambda example:
    ros2 launch husky_rl rl.launch.py adaptive:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    adaptive_arg = DeclareLaunchArgument(
        'adaptive', default_value='false',
        description='true = Algo 2 adaptive lambda, false = fixed lambda')

    fixed_lambda_arg = DeclareLaunchArgument(
        'fixed_lambda', default_value='[1, 1, 1, 1]',
        description='Fixed lambda values, e.g. "[1,1,1,10]" (used only when adaptive:=false)')

    cell_size_arg = DeclareLaunchArgument(
        'cell_size_m', default_value='0.0',
        description='Measured metres per grid cell; 0 disables the Discretizer')

    origin_x_arg = DeclareLaunchArgument(
        'origin_x_m', default_value='0.0',
        description='World-frame x coordinate of grid cell (0, 0)')

    origin_y_arg = DeclareLaunchArgument(
        'origin_y_m', default_value='0.0',
        description='World-frame y coordinate of grid cell (0, 0)')

    invert_y_arg = DeclareLaunchArgument(
        'invert_y_axis', default_value='false',
        description='Invert world y when converting to grid rows')

    discretizer_node = Node(
        package='husky_rl',
        executable='discretizer',
        name='discretizer_node',
        parameters=[{
            'cell_size_m': ParameterValue(
                LaunchConfiguration('cell_size_m'), value_type=float),
            'origin_x_m': ParameterValue(
                LaunchConfiguration('origin_x_m'), value_type=float),
            'origin_y_m': ParameterValue(
                LaunchConfiguration('origin_y_m'), value_type=float),
            'invert_y_axis': ParameterValue(
                LaunchConfiguration('invert_y_axis'), value_type=bool),
        }],
        output='screen',
    )

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

    astar_node = Node(
        package='husky_rl',
        executable='astar_node',
        name='astar_node',
        # TODO: set cell_size_m once lab dimensions are measured
        parameters=[{'cell_size_m': 0.0}],
        output='screen',
    )

    controller_node = Node(
        package='husky_control',
        executable='point_controller',
        name='point_controller',
        parameters=[{'points_topic': '/rl_path'}],
        output='screen',
    )

    return LaunchDescription([
        adaptive_arg,
        fixed_lambda_arg,
        cell_size_arg,
        origin_x_arg,
        origin_y_arg,
        invert_y_arg,
        discretizer_node,
        lambda_node,
        rl_policy_node,
        astar_node,
        controller_node,
    ])
