"""Run on the Husky host after sourcing ROS Jazzy."""
import math
import time

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import NavSatFix, NavSatStatus


def main():
    rclpy.init(domain_id=0)
    node = rclpy.create_node("husky_gps_bench_receiver")
    stamps = set()

    def receive(msg):
        if msg.header.frame_id != "gps_link":
            raise ValueError(f"Unexpected antenna frame {msg.header.frame_id}")
        if msg.status.status == NavSatStatus.STATUS_NO_FIX:
            if not (math.isnan(msg.latitude) and math.isnan(msg.longitude)):
                raise ValueError("Expected unknown coordinates for this no-fix fixture")
        elif not (-90 <= msg.latitude <= 90 and -180 <= msg.longitude <= 180):
            raise ValueError("Fix has invalid coordinates")
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        if stamp != (0, 0):
            stamps.add(stamp)
        print(f"frame={msg.header.frame_id} stamp={stamp} status={msg.status.status}")

    node.create_subscription(NavSatFix, "/gps/fix", receive, qos_profile_sensor_data)
    deadline = time.monotonic() + 15
    try:
        while len(stamps) < 2 and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.25)
        if len(stamps) < 2:
            raise SystemExit("Did not receive two distinct GPS samples")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
