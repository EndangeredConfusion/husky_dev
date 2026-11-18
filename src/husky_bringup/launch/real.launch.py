"""Real robot: drivers for the sensors Clearpath does not run (Clearpath's services already run the base).

    ros2 launch husky_bringup real.launch.py
Frame ids must match the link names in husky_description/urdf/husky_extras.urdf.xacro.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    namespace = LaunchConfiguration('namespace')
    lidar_ip = LaunchConfiguration('lidar_ip')

    # RPLidar S2E over Ethernet (Slamtec sllidar_ros2). Check the lidar's IP and port in its manual/config.
    lidar = Node(
        package='sllidar_ros2', executable='sllidar_node', name='rplidar', namespace=namespace,
        parameters=[{
            'channel_type': 'udp',
            'udp_ip': lidar_ip,
            'udp_port': 8089,
            'frame_id': 'rplidar_laser',
            'inverted': False,
            'angle_compensate': True,
            'scan_mode': 'DenseBoost',
        }],
        remappings=[('scan', 'sensors/lidar2d_0/scan')],
        output='screen')

    # TODO: u-blox ZED-X20D driver. Publish sensors/gps_0/fix (frame gps_front_link)
    # and sensors/gps_1/fix (frame gps_rear_link), plus heading if the driver provides it.

    return LaunchDescription([
        DeclareLaunchArgument('namespace', default_value='a200_0000', description='Must match robot.yaml'),
        DeclareLaunchArgument('lidar_ip', default_value='192.168.11.2'),
        lidar,
    ])
