"""Record isolated simulation interfaces for execution/fault assertions."""
import json
import os
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist


def main():
    if int(os.environ.get('ROS_DOMAIN_ID', '0')) == 0:
        raise RuntimeError('Probe is for isolated simulation domains only')
    rclpy.init()
    node = Node('transformer_sim_probe')
    def emit(topic, data):
        print(json.dumps(dict(wall_time=time.monotonic(), topic=topic, data=data)), flush=True)
    for topic in ('state', 'feedback', 'audit'):
        node.create_subscription(String, '/transformer_sim/' + topic,
            lambda msg, topic=topic: emit(topic, json.loads(msg.data)), 100)
    node.create_subscription(Twist, '/transformer_sim/cmd_vel',
        lambda msg: emit('cmd_vel', dict(v=msg.linear.x, omega=msg.angular.z)), 100)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
