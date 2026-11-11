"""
Simulation: the full RL pipeline on the '86 Field cells, with fake GPS.

Start Gazebo with the '86 Field world first, then this:

    ros2 launch clearpath_gz simulation.launch.py world:=rpi_86_field_1
    ros2 launch husky_rl rl_sim.launch.py robot_namespace:=a200_0000

World metres = cell-table metres (origin Star_1, +x east, +y north).  The
default spawn x=y=yaw=0 is Star_1 = cell (1,1), facing east; to spawn on
another cell, give clearpath_gz its east_m / north_m from
config/field86_cells.yaml as x / y.

sim_gps publishes /gps/fix + /heading from the robot's TRUE pose in Gazebo
(pose_source:=gazebo, default), so the spawn pose does not need repeating
here.  pose_source:=odom uses wheel odometry instead, placed at x / y / yaw
(which must then match the spawn); it drifts on turns.  Everything else is
rl.launch.py with use_gps_pose:=true, the same nodes as on the robot.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    ns = LaunchConfiguration('robot_namespace')
    args = [
        DeclareLaunchArgument('robot_namespace', default_value='a200_1201',
                              description='Clearpath robot namespace in Gazebo'),
        DeclareLaunchArgument('pose_source', default_value='gazebo',
                              description="'gazebo' = true pose from Gazebo (no drift); "
                                          "'odom' = wheel odometry placed at x/y/yaw"),
        DeclareLaunchArgument('gz_world', default_value='rpi_86_field_1',
                              description='Gazebo world name (for its pose topic)'),
        DeclareLaunchArgument('x', default_value='0.0',
                              description='Gazebo spawn x (east, m); same as clearpath_gz x'),
        DeclareLaunchArgument('y', default_value='0.0',
                              description='Gazebo spawn y (north, m); same as clearpath_gz y'),
        DeclareLaunchArgument('yaw', default_value='0.0',
                              description='Gazebo spawn yaw (rad, 0 = east); same as clearpath_gz yaw'),
        DeclareLaunchArgument('noise_std_m', default_value='0.0',
                              description='Fake GPS position noise (std, metres)'),
        DeclareLaunchArgument('adaptive', default_value='false'),
        DeclareLaunchArgument('fixed_lambda', default_value='[1, 1, 1, 1]'),
    ]

    sim_gps = Node(
        package='husky_localization',
        executable='sim_gps',
        name='sim_gps',
        parameters=[{
            'pose_source': LaunchConfiguration('pose_source'),
            'gz_world': LaunchConfiguration('gz_world'),
            'gz_model': [ns, '/robot'],
            'odom_topic': ['/', ns, '/platform/odom/filtered'],
            'spawn_x': ParameterValue(LaunchConfiguration('x'), value_type=float),
            'spawn_y': ParameterValue(LaunchConfiguration('y'), value_type=float),
            'spawn_yaw': ParameterValue(LaunchConfiguration('yaw'), value_type=float),
            'noise_std_m': ParameterValue(
                LaunchConfiguration('noise_std_m'), value_type=float),
        }],
        output='screen',
    )

    rl = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare('husky_rl'), 'launch', 'rl.launch.py'])),
        launch_arguments={
            'use_gps_pose': 'true',
            'cmd_vel_topic': ['/', ns, '/cmd_vel'],
            'adaptive': LaunchConfiguration('adaptive'),
            'fixed_lambda': LaunchConfiguration('fixed_lambda'),
        }.items(),
    )

    return LaunchDescription(args + [sim_gps, rl])
