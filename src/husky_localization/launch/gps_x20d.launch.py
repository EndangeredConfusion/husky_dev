"""
u-blox ZED-X20D on the robot: /gps/fix (position) + /heading (dual antenna).

    ros2 launch husky_localization gps_x20d.launch.py
    ros2 launch husky_localization gps_x20d.launch.py port:=/dev/ttyACM0

x20d_nmea_reader owns the serial port, skips binary UBX data, publishes every
NMEA sentence on nmea_sentence and the THS heading on /heading.
nmea_navsat_driver's nmea_topic_driver turns the sentences (GGA) into
/gps/fix, the same parser the bench used (bench/gps_navfn/start_gps.sh).
Do not run start_gps.sh at the same time: only one program can own the port.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    args = [
        DeclareLaunchArgument(
            'port',
            default_value='/dev/serial/by-id/usb-u-blox_AG_-_www.u-blox.com_u-blox_GNSS_receiver-if00'),
        DeclareLaunchArgument('baud', default_value='115200'),
        DeclareLaunchArgument('frame_id', default_value='gps_link'),
        DeclareLaunchArgument('accepted_modes', default_value='A',
                              description='THS modes accepted as valid heading (A = autonomous)'),
    ]
    reader = Node(
        package='husky_localization',
        executable='x20d_nmea_reader',
        name='x20d_nmea_reader',
        parameters=[{
            'port': LaunchConfiguration('port'),
            'baud': ParameterValue(LaunchConfiguration('baud'), value_type=int),
            'frame_id': LaunchConfiguration('frame_id'),
            'accepted_modes': ParameterValue(
                LaunchConfiguration('accepted_modes'), value_type=str),
        }],
        output='screen',
    )
    gga = Node(
        package='nmea_navsat_driver',
        executable='nmea_topic_driver',
        name='nmea_topic_driver',
        remappings=[('fix', '/gps/fix')],
        output='screen',
    )
    return LaunchDescription(args + [reader, gga])
