"""
Field walk-through check: GPS fix -> cell table, printed and logged to CSV.

Start the GPS driver separately (it must publish sensor_msgs/NavSatFix).

    ros2 launch husky_rl field_check.launch.py
    ros2 launch husky_rl field_check.launch.py csv_path:=/tmp/walk1.csv
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    args = [
        DeclareLaunchArgument(
            'cell_table',
            default_value=PathJoinSubstitution(
                [FindPackageShare('husky_rl'), 'config', 'field86_cells.yaml']),
            description='Path to the cell table YAML'),
        DeclareLaunchArgument('gps_fix_topic', default_value='/gps/fix'),
        DeclareLaunchArgument('heading_topic', default_value='/heading'),
        DeclareLaunchArgument(
            'csv_path', default_value='',
            description='CSV log file; empty = ~/field_check_<time>.csv'),
        DeclareLaunchArgument('reached_radius_m', default_value='1.5'),
        DeclareLaunchArgument('off_walkway_m', default_value='6.0'),
    ]

    field_check = Node(
        package='husky_rl',
        executable='field_check',
        name='field_check',
        parameters=[{
            'cell_table': LaunchConfiguration('cell_table'),
            'gps_fix_topic': LaunchConfiguration('gps_fix_topic'),
            'heading_topic': LaunchConfiguration('heading_topic'),
            'csv_path': ParameterValue(
                LaunchConfiguration('csv_path'), value_type=str),
            'reached_radius_m': ParameterValue(
                LaunchConfiguration('reached_radius_m'), value_type=float),
            'off_walkway_m': ParameterValue(
                LaunchConfiguration('off_walkway_m'), value_type=float),
        }],
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription(args + [field_check])
