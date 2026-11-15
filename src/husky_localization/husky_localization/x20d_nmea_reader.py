#!/usr/bin/env python3
'''
u-blox ZED-X20D serial reader: NMEA sentences + dual-antenna heading.

  in : serial port (USB), NMEA mixed with binary UBX
  out: nmea_sentence  nmea_msgs/Sentence               every valid NMEA sentence
       heading_topic  geometry_msgs/QuaternionStamped  (default /heading) from THS

Run with nmea_navsat_driver's nmea_topic_driver, which turns the sentences
(GGA) into /gps/fix exactly as the bench driver does:
    ros2 launch husky_localization gps_x20d.launch.py

Heading is published like nmea_navsat_driver does for HDT: the compass heading
(0 = north, clockwise) as the quaternion's yaw, in radians.  That matches
gps_pose's default heading_convention=cw_from_north.  Only THS sentences with
a mode in accepted_modes (default 'A', autonomous = valid) are published.
'''

import math
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from geometry_msgs.msg import QuaternionStamped

from husky_localization.nmea_stream import NmeaExtractor, parse_ths, sentence_type


class X20dNmeaReader(Node):
    def __init__(self):
        super().__init__('x20d_nmea_reader')
        self.declare_parameter(
            'port', '/dev/serial/by-id/usb-u-blox_AG_-_www.u-blox.com_u-blox_GNSS_receiver-if00')
        self.declare_parameter('baud', 115200)
        self.declare_parameter('frame_id', 'gps_link')
        self.declare_parameter('sentence_topic', 'nmea_sentence')
        self.declare_parameter('heading_topic', '/heading')
        self.declare_parameter('accepted_modes', 'A')
        # false = heading only (no nmea_msgs needed, no /gps/fix downstream);
        # useful for testing the heading on its own.
        self.declare_parameter('publish_sentences', True)

        def p(name):
            return self.get_parameter(name).value

        # Imported here so the parsing tests and the rest of the package do not
        # need these installed.
        import serial
        self.publish_sentences = bool(p('publish_sentences'))
        if self.publish_sentences:
            from nmea_msgs.msg import Sentence
            self._Sentence = Sentence

        self.port_name = str(p('port'))
        self.baud = int(p('baud'))
        self.frame_id = str(p('frame_id'))
        self.accepted = set(str(p('accepted_modes')))
        self.serial = serial.Serial(self.port_name, self.baud, timeout=0)
        self.extractor = NmeaExtractor()

        self.sentence_pub = (
            self.create_publisher(self._Sentence, str(p('sentence_topic')), 10)
            if self.publish_sentences else None)
        self.heading_pub = self.create_publisher(QuaternionStamped, str(p('heading_topic')), 10)

        self.counts = {'sentences': 0, 'ths': 0, 'heading': 0, 'ths_rejected': 0}
        self.last_mode = None
        self.last_report = time.monotonic()
        self.create_timer(0.01, self.read_serial)
        self.create_timer(10.0, self.report)
        self.get_logger().info(
            f'reading {self.port_name} at {self.baud} baud; heading from THS '
            f'(modes {sorted(self.accepted)}) -> {p("heading_topic")}')

    def read_serial(self):
        try:
            data = self.serial.read(4096)
        except Exception as exc:  # unplugged USB, etc.
            self.get_logger().error(f'serial read failed: {exc}', throttle_duration_sec=5.0)
            return
        if not data:
            return
        stamp = self.get_clock().now().to_msg()
        for text in self.extractor.feed(data):
            self.counts['sentences'] += 1
            if self.sentence_pub is not None:
                msg = self._Sentence()
                msg.header.stamp = stamp
                msg.header.frame_id = self.frame_id
                msg.sentence = text
                self.sentence_pub.publish(msg)
            if sentence_type(text) == 'THS':
                self.handle_ths(text, stamp)

    def handle_ths(self, text, stamp):
        self.counts['ths'] += 1
        parsed = parse_ths(text)
        if parsed is None:
            self.counts['ths_rejected'] += 1
            return
        heading_deg, mode = parsed
        if mode != self.last_mode:
            self.get_logger().info(f'THS heading mode: {mode!r}')
            self.last_mode = mode
        if heading_deg is None or mode not in self.accepted:
            self.counts['ths_rejected'] += 1
            return
        yaw = math.radians(heading_deg)
        out = QuaternionStamped()
        out.header.stamp = stamp
        out.header.frame_id = self.frame_id
        out.quaternion.z = math.sin(yaw / 2.0)
        out.quaternion.w = math.cos(yaw / 2.0)
        self.heading_pub.publish(out)
        self.counts['heading'] += 1

    def report(self):
        c = self.counts
        self.get_logger().info(
            f"last 10 s: {c['sentences']} sentences, {c['ths']} THS, "
            f"{c['heading']} headings published, {c['ths_rejected']} THS rejected, "
            f'{self.extractor.bad_checksums} bad checksums in total')
        if not c['sentences']:
            self.get_logger().warn(
                f'no NMEA data from {self.port_name} for 10 s: check the cable, '
                'the port/baud, and that NMEA output is enabled on this port')
        elif not c['ths']:
            self.get_logger().warn(
                'no THS sentences: enable NMEA THS output on this port (u-center 2)')
        for key in c:
            c[key] = 0

    def destroy_node(self):
        try:
            self.serial.close()
        finally:
            super().destroy_node()


def main():
    rclpy.init()
    node = X20dNmeaReader()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
