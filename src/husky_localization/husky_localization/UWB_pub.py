import rclpy
from rclpy.node import Node

from std_msgs.msg import String


import serial

from husky_interfaces.msg import UwbReading, UwbReadingArray, UwbPos, UwbPosMap

import yaml


class UWB_Pub(Node):

    def __init__(self):
        super().__init__('uwb_publisher')
        # ranges param
        self.declare_parameter('uwb_ranges_topic', '/uwb/ranges')
        self.uwb_ranges_topic = self.get_parameter('uwb_ranges_topic').value
        # map param topic
        self.declare_parameter('uwb_map_topic', '/uwb/map')
        self.uwb_map_topic = self.get_parameter('uwb_map_topic').value
        # uwb tag usb port
        self.declare_parameter('uwb_port', '/dev/ttyUSB0')
        self.uwb_port = self.get_parameter('uwb_port').value
        # uwb locations config file
        self.declare_parameter('uwb_locations_config_path', '~/config/uwb_locations.yaml')
        self.uwb_locations_config_path = self.get_parameter('uwb_locations_config_path').value

        self.range_publisher = self.create_publisher(UwbReadingArray, self.uwb_ranges_topic, 10)

        # latched qos (quality of service) for one time map topic
        map_qos = rclpy.qos.QoSProfile(
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL
        )
        self.map_publisher = self.create_publisher(UwbPosMap, self.uwb_map_topic, map_qos)
        
        with open(self.uwb_locations_config_path, 'r') as f:
            uwb_locations = yaml.safe_load(f)
            pos_messages = []
            for anchor in uwb_locations['anchors']:
                pos_msg = UwbPos()
                pos_msg.anchor_id = anchor['id']
                pos_msg.position_m.x = anchor['position'][0]
                pos_msg.position_m.y = anchor['position'][1]
                pos_msg.position_m.z = anchor['position'][2]
                pos_messages.append(pos_msg)
        pos_map_msg = UwbPosMap()
        pos_map_msg.uwb_positions_array = pos_messages
        self.map_publisher.publish(pos_map_msg)
        

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.range_publisher.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()