"""Sim in one command: Gazebo + Husky, sensor bridge, Nav2 (odom frame), RViz.

Worlds are found in husky_bringup/worlds/ first, then Clearpath's own worlds.

    ros2 launch husky_bringup sim.launch.py
    ros2 launch husky_bringup sim.launch.py world:=warehouse use_rviz:=false
"""
import os

from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable, DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    world = LaunchConfiguration('world')
    namespace = LaunchConfiguration('namespace')
    setup_path = LaunchConfiguration('setup_path')
    rviz = LaunchConfiguration('use_rviz')  # not 'rviz': Clearpath's spawn sets that one to false
    nav2_delay = LaunchConfiguration('nav2_delay')

    def clearpath(pkg, launch_file):
        return PythonLaunchDescriptionSource(PathJoinSubstitution([FindPackageShare(pkg), 'launch', launch_file]))

    # Our worlds (husky_bringup/worlds) on Gazebo's search path, so world:=rpi_86_field resolves to our file
    worlds_path = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH', PathJoinSubstitution([FindPackageShare('husky_bringup'), 'worlds']))

    # Gazebo with the world, then the Husky built from ~/clearpath/robot.yaml.
    # Calls Clearpath's two inner launch files directly (simulation.launch.py only accepts its own world names).
    gz = IncludeLaunchDescription(
        clearpath('clearpath_gz', 'gz_sim.launch.py'),
        launch_arguments={'world': world, 'setup_path': setup_path}.items())
    spawn = IncludeLaunchDescription(
        clearpath('clearpath_gz', 'robot_spawn.launch.py'),
        launch_arguments={'use_sim_time': 'true', 'setup_path': setup_path, 'world': world, 'rviz': 'false',
                          'x': '0.0', 'y': '0.0', 'z': '0.3', 'yaw': '0.0'}.items())

    # Gazebo -> ROS for our own sensors (RPLidar S2E, GPS antennas)
    bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge', name='husky_extras_bridge',
        parameters=[{'config_file': PathJoinSubstitution([FindPackageShare('husky_bringup'), 'config', 'gz_bridge.yaml']),
                     'use_sim_time': True}],
        output='screen')

    # Nav2: slam:=true gives a rolling global costmap (no SLAM map needed for the open field).
    # Delayed so the sim is up first; raise nav2_delay if your sim starts slowly.
    nav2 = TimerAction(period=nav2_delay, actions=[IncludeLaunchDescription(
        clearpath('clearpath_nav2_demos', 'nav2.launch.py'),
        launch_arguments={'setup_path': setup_path, 'use_sim_time': 'true', 'slam': 'true'}.items())])

    # RViz with our saved view (config/sim.rviz). Namespaced like Clearpath's view_navigation,
    # so the Nav2 panel, Nav2 Goal tool and relative topics work.
    rviz_node = Node(
        package='rviz2', executable='rviz2', name='rviz2', namespace=namespace,
        arguments=['-d', PathJoinSubstitution([FindPackageShare('husky_bringup'), 'config', 'sim.rviz'])],
        parameters=[{'use_sim_time': True}],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static'),
                    ('/goal_pose', 'goal_pose'), ('/clicked_point', 'clicked_point'),
                    ('/initialpose', 'initialpose')],
        condition=IfCondition(rviz),
        output='screen')

    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='rpi_86_field'),
        DeclareLaunchArgument('namespace', default_value='a200_0000', description='Must match robot.yaml'),
        DeclareLaunchArgument('setup_path', default_value=os.path.expanduser('~/clearpath/')),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('nav2_delay', default_value='20.0', description='Seconds to wait before starting Nav2'),
        worlds_path, gz, spawn, bridge, nav2, rviz_node,
    ])