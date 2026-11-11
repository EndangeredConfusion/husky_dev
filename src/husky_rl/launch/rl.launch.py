"""
Launch the GPS discretizer and all RL-side nodes plus the controller.

Fixed lambda example:
    ros2 launch husky_rl rl.launch.py adaptive:=false fixed_lambda:="[1,2,2,9]"

Adaptive lambda example:
    ros2 launch husky_rl rl.launch.py adaptive:=true

Outdoors on the '86 Field (controller pose from GPS):
    ros2 launch husky_rl rl.launch.py use_gps_pose:=true

Cell positions come from the cell table (config/field86_cells.yaml).  The
discretizer needs a sensor_msgs/NavSatFix on gps_fix_topic; without GPS
(e.g. in simulation) it publishes no cells.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    adaptive_arg = DeclareLaunchArgument(
        'adaptive', default_value='false',
        description='true = Algo 2 adaptive lambda, false = fixed lambda')

    fixed_lambda_arg = DeclareLaunchArgument(
        'fixed_lambda', default_value='[1, 1, 1, 1]',
        description='Fixed lambda values, e.g. "[1,1,1,10]" (used only when adaptive:=false)')

    cell_table_arg = DeclareLaunchArgument(
        'cell_table',
        default_value=PathJoinSubstitution(
            [FindPackageShare('husky_rl'), 'config', 'field86_cells.yaml']),
        description='Path to the GPS cell table YAML')

    gps_fix_topic_arg = DeclareLaunchArgument(
        'gps_fix_topic', default_value='/gps/fix',
        description='sensor_msgs/NavSatFix topic from the GPS driver')

    require_rtk_arg = DeclareLaunchArgument(
        'require_rtk', default_value='false',
        description='Reject fixes that are not RTK (NavSatStatus < GBAS_FIX)')

    reached_radius_arg = DeclareLaunchArgument(
        'reached_radius_m', default_value='1.5',
        description='A cell counts as reached within this distance of its centre')

    off_walkway_arg = DeclareLaunchArgument(
        'off_walkway_m', default_value='6.0',
        description='Farther than this from every walkway = off the walkways')

    use_gps_pose_arg = DeclareLaunchArgument(
        'use_gps_pose', default_value='false',
        description='true = run gps_pose and drive the controller from GPS; '
                    'false = controller uses uwb_local/ekf (UWB EKF or sim odom_to_ekf)')

    cmd_vel_topic_arg = DeclareLaunchArgument(
        'cmd_vel_topic', default_value='/a200_1201/cmd_vel',
        description='Velocity command topic for point_controller')

    # gps_pose corrections; measure on the robot (see
    # bench/gps_cells/HARDWARE_CHECKLIST.md, steps I6 and O3).  0 = no correction.
    heading_offset_arg = DeclareLaunchArgument(
        'heading_offset_rad', default_value='0.0',
        description='GPS heading correction: reading minus true direction (rad)')

    antenna_offset_x_arg = DeclareLaunchArgument(
        'antenna_offset_x_m', default_value='0.0',
        description='Main antenna (ANT1) position ahead of the robot centre (m)')

    antenna_offset_y_arg = DeclareLaunchArgument(
        'antenna_offset_y_m', default_value='0.0',
        description='Main antenna (ANT1) position left of the robot centre (m)')

    cell_table = LaunchConfiguration('cell_table')
    use_gps_pose = LaunchConfiguration('use_gps_pose')

    discretizer_node = Node(
        package='husky_rl',
        executable='discretizer',
        name='discretizer_node',
        parameters=[{
            'cell_table': cell_table,
            'gps_fix_topic': LaunchConfiguration('gps_fix_topic'),
            'require_rtk': ParameterValue(
                LaunchConfiguration('require_rtk'), value_type=bool),
            'reached_radius_m': ParameterValue(
                LaunchConfiguration('reached_radius_m'), value_type=float),
            'off_walkway_m': ParameterValue(
                LaunchConfiguration('off_walkway_m'), value_type=float),
        }],
        output='screen',
    )

    lambda_node = Node(
        package='husky_rl',
        executable='lambda_node',
        name='lambda_node',
        parameters=[{
            'adaptive':     LaunchConfiguration('adaptive'),
            # lambda_node declares fixed_lambda as a string (JSON); without
            # value_type=str launch turns "[1, 1, 1, 1]" into an int array.
            'fixed_lambda': ParameterValue(
                LaunchConfiguration('fixed_lambda'), value_type=str),
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
        parameters=[{'cell_table': cell_table}],
        output='screen',
    )

    gps_pose_node = Node(
        package='husky_localization',
        executable='gps_pose',
        name='gps_pose',
        parameters=[{
            'cell_table': cell_table,
            'gps_fix_topic': LaunchConfiguration('gps_fix_topic'),
            'pose_topic': '/gps/local_pose',
            'heading_offset_rad': ParameterValue(
                LaunchConfiguration('heading_offset_rad'), value_type=float),
            'antenna_offset_x_m': ParameterValue(
                LaunchConfiguration('antenna_offset_x_m'), value_type=float),
            'antenna_offset_y_m': ParameterValue(
                LaunchConfiguration('antenna_offset_y_m'), value_type=float),
        }],
        condition=IfCondition(use_gps_pose),
        output='screen',
    )

    controller_node = Node(
        package='husky_control',
        executable='point_controller',
        name='point_controller',
        parameters=[{
            'points_topic': '/rl_path',
            'cmd_vel_topic': LaunchConfiguration('cmd_vel_topic'),
            'localization_topic': PythonExpression([
                "'/gps/local_pose' if '", use_gps_pose,
                "'.lower() == 'true' else 'uwb_local/ekf'"]),
        }],
        output='screen',
    )

    return LaunchDescription([
        adaptive_arg,
        fixed_lambda_arg,
        cell_table_arg,
        gps_fix_topic_arg,
        require_rtk_arg,
        reached_radius_arg,
        off_walkway_arg,
        use_gps_pose_arg,
        cmd_vel_topic_arg,
        heading_offset_arg,
        antenna_offset_x_arg,
        antenna_offset_y_arg,
        discretizer_node,
        lambda_node,
        rl_policy_node,
        astar_node,
        gps_pose_node,
        controller_node,
    ])
