import rclpy
from rclpy.node import Node
import serial
from husky_interfaces.msg import UwbReading, UwbReadingArray, UwbPos, UwbPosMap
from geometry_msgs.msg import Point
import yaml
from pathlib import Path

    
dwm_cfg_get = bytes([0x08, 0x00])
desired_cfg_get_prefix = bytes([0x40, 0x01, 0x00, 0x46, 0x02])

def decode_uwb_status(status_bytes) -> str:
    outstr = "UWB Status:\n"
    mappings = [lambda bytes: f"uwb mode: {bytes[0] & 0x03}",
                lambda bytes: f"fw update en: {(bytes[0] >> 2) & 0x01}",
                lambda bytes: f"ble en: {(bytes[0] >> 3) & 0x01}",
                lambda bytes: f"led en: {(bytes[0] >> 4) & 0x01}",
                # byte 0 bit 5 reserved
                lambda bytes: f"lco engine en: {(bytes[0] >> 6) & 0x01}",
                lambda bytes: f"low power en: {(bytes[0] >> 7) & 0x01}",
                lambda bytes: f"meas_mode (0-TWR, 1-3 n/a): {bytes[1] & 0x03}",
                lambda bytes: f"accel en: {(bytes[1] >> 2) & 0x01}",
                lambda bytes: f"bridge: {(bytes[1] >> 3) & 0x01}",
                lambda bytes: f"initiator: {(bytes[1] >> 4) & 0x01}",
                lambda bytes: f"mode (0-tag, 1-anchor): {(bytes[1] >> 5) & 0x01}"
                # byte 1 bit 6 reserved
                # byte 1 bit 7 reserved
                ]
    for mapping in mappings:
        outstr += mapping(status_bytes) + "\n"
    return outstr

def validate_serial_read(ser, expected_length):
    resp = ser.read(expected_length)
    if len(resp) != expected_length:
        return False, f"Expected {expected_length} bytes but got {len(resp)}"
    return True, resp


class UwbAnchorBuffer:
    def __init__(self, dist, repeat_count):
        self.old = dist
        self.repeat_count = repeat_count

class UwbRangeValidator:
    '''
    Class to buffer the last few readings from anchors and then tell us if the anchor is in range or not.
    '''
    def __init__(self, num_repeats_allowed=3):
        self.num_repeats_allowed = num_repeats_allowed
        self.uwbs = dict() # key is anchor id, values are uwb anchor buffers

    def add_range(self, anchor_id, distance) -> bool:
        if anchor_id not in self.uwbs:
            self.uwbs[anchor_id] = UwbAnchorBuffer(distance, 0)
            return True
        else:
            # check if new value is same as head, replace, repeat
            buffer = self.uwbs[anchor_id]
            if buffer.old == distance:
                buffer.repeat_count += 1
            else:
                buffer.repeat_count = 0
            buffer.old = distance
            return buffer.repeat_count < self.num_repeats_allowed

    def is_in_range(self, anchor_id):
        return anchor_id in self.uwbs and self.uwbs[anchor_id].repeat_count < self.num_repeats_allowed
    
            
class UWB_Pub(Node):
    def __init__(self):
        super().__init__('uwb_publisher')
        self.logger = self.get_logger()
        self.logger.info('UWB Publisher node has been started.')

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
        
        config_path = Path(self.uwb_locations_config_path).expanduser()
        with open(config_path, 'r') as f:
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
        self.logger.info(f"Published UWB anchor positions from config file: {self.uwb_locations_config_path}")
        
        self.init_uwb_reader()        

    def init_uwb_reader(self):
        # set up reader
        self.logger.info(f"Opening serial port: {self.uwb_port}")
        try:
            self.ser = serial.Serial(self.uwb_port, 115200, timeout=1)
            self.logger.info(f"Successfully opened serial port: {self.uwb_port}")
        except serial.SerialException as e:
            self.logger.error(f"Failed to open serial port {self.uwb_port}: {e}")
            return
            self.ser.reset_input_buffer()
        
        # check that reader is working (and that uwb is a tag)
        self.ser.write(dwm_cfg_get) # send config get command to check connection
        self.ser.flush()
        success, resp = validate_serial_read(self.ser, 7)
        if success and resp[:5] == desired_cfg_get_prefix:
            self.logger.info(f"Successfully communicated with UWB tag. Current config:\n{decode_uwb_status(resp[5:7])}")
            self.uwb_validator = UwbRangeValidator() # class to filter out uwbs out of range based on repeated readings
            self.timer = self.create_timer(.1, self.read_uwb_ranges)  # read UWB data 10x every second

        else:
            self.logger.error(f"Unexpected response from UWB tag: {resp}. Check connection and config.")
            return
            self.ser.reset_input_buffer()
            

    def read_uwb_ranges(self):
        read_command = bytes([0x0C, 0x00])
        transaction_good_prefix = bytes([0x40, 0x01, 0x00])
        self.ser.write(read_command)
        self.ser.flush()
        success, resp = validate_serial_read(self.ser, 3)
        if success and resp == transaction_good_prefix:
            self.logger.debug("Transaction beginning successfully")
        else:
            self.logger.error("Transaction beginning failed")
            return
            self.ser.reset_input_buffer()

        success, position_resp = validate_serial_read(self.ser, 15)
        position_good_prefix = bytes([0x41, 0x0D])
        if success and position_resp[:2] == position_good_prefix:
            self.logger.debug("Position data received successfully")
        else:
            self.logger.error("Position data received failed")
            self.ser.reset_input_buffer()
            return
            
        position = position_resp[2:]
        tag_posx = position[:4]
        tag_posy = position[4:8]
        tag_posz = position[8:12]
        tag_pos_quality = position[12]

        success, ranging_count_resp = validate_serial_read(self.ser, 3)
        ranging_count_good_prefix = bytes([0x49])
        if success and ranging_count_resp[:1] == ranging_count_good_prefix:
            self.logger.debug("Ranging count data received successfully")
        else:
            self.logger.error("Ranging count data received failed")
            self.ser.reset_input_buffer()
            return
        
        num_incoming_bytes = ranging_count_resp[1] - 1
        ranging_count = ranging_count_resp[2]
        if num_incoming_bytes != ranging_count*20:
            self.logger.error("number of bytes and number of ranging count do not match, possible desync")
            self.ser.reset_input_buffer()
            return
        
        self.ranges_array = []
        for i in range(ranging_count):
            success, ranging_resp = validate_serial_read(self.ser, 20)
            if not success:
                self.logger.error(f"Failed to read ranging response {i+1}/{ranging_count}")
                self.ser.reset_input_buffer()
                return
                
            address = ranging_resp[:2] # first 2 bytes are address
            distance = ranging_resp[2:6] # next 4 bytes are distance
            dist_quality = ranging_resp[6] # then 1 byte quality
            # 13 bytes position data then the next range
            posx = ranging_resp[7:11]
            posy = ranging_resp[11:15]
            posz = ranging_resp[15:19]
            anchor_pos_quality = ranging_resp[19]
            
            # log buffers
            anchor_id = int.from_bytes(address, byteorder='little')
            distance_m = int.from_bytes(distance, byteorder='little') / 1000.
            
            if self.uwb_validator.add_range(anchor_id, distance_m):
                uwb_reading = UwbReading()
                uwb_reading.anchor_id = anchor_id
                uwb_reading.distance_m = distance_m
                uwb_reading.position_m.x = int.from_bytes(posx, byteorder='little', signed=True) / 1000.0
                uwb_reading.position_m.y = int.from_bytes(posy, byteorder='little', signed=True) / 1000.0
                uwb_reading.position_m.z = int.from_bytes(posz, byteorder='little', signed=True) / 1000.0
                uwb_reading.quality = dist_quality
                self.ranges_array.append(uwb_reading)
        self.range_publisher.publish(UwbReadingArray(uwb_readings_array=self.ranges_array))
        self.logger.debug(f"Published {len(self.ranges_array)} UWB ranges")
        
    def destroy_node(self):
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = UWB_Pub()

    rclpy.spin(node)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()