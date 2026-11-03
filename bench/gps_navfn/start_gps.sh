#!/usr/bin/env bash
set -e
source /opt/ros/jazzy/setup.bash
port=/dev/serial/by-id/usb-u-blox_AG_-_www.u-blox.com_u-blox_GNSS_receiver-if00
test -r "$port" && test -w "$port" || { echo "Missing access to $port" >&2; exit 1; }
if fuser "$port" >/dev/null 2>&1; then echo "Serial port already in use" >&2; exit 1; fi
export ROS_DOMAIN_ID=0 PYTHONNOUSERSITE=1
exec /opt/ros/jazzy/lib/nmea_navsat_driver/nmea_serial_driver --ros-args -p port:="$port" -p baud:=115200 -p frame_id:=gps_link -r fix:=/gps/fix --log-level error
