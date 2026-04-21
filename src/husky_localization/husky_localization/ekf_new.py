#!/usr/bin/env python3

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped
from std_msgs.msg import Float64MultiArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from husky_interfaces.msg import UwbReading, UwbReadingArray, UwbPos, UwbPosMap, EkfState

class Point3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


def angle_wrap(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi
        
        
# EKF holds state [x, y, theta]
# action = [v, omega]
class EkfLocalizationNode(Node):
    def __init__(self):
        super().__init__("ekf_localization")
        
        # initial state vector
        self.declare_parameter('initial_state', [0.0, 0.0, 0.0])
        self.initial_state = np.array(
            self.get_parameter('initial_state').value, 
            dtype=np.float64
        )
        self.state = self.initial_state
        self.covariance = np.diag([.01, .01, 0.035])
                
        # process noise covariance
        self.declare_parameter('proc_noise_diag', [0.1, 0.1, 0.1])
        q = np.array(
            self.get_parameter('proc_noise_diag').value,
            dtype=np.float64
        )
        self.proc_noise_cov = np.diag(q)


        # measurement variances
        self.declare_parameter('odom_variance', .01)
        self.uwb_variance = float(self.get_parameter('odom_variance').value)
        self.declare_parameter('uwb_variance', .05)
        self.odom_variance = float(self.get_parameter('uwb_variance').value)
        
        # height of the uwb tag on the robot .45 meters
        self.declare_parameter('uwb_height_on_robot', .45)
        self.uwb_height_on_robot = float(self.get_parameter('uwb_height_on_robot').value)
        
        # subscribe to the UWB ranges topic
        self.declare_parameter('uwb_ranges_topic', '/uwb/ranges')
        self.uwb_ranges_topic = str(self.get_parameter('uwb_ranges_topic').value)
        
        self.uwb_subscriber = self.create_subscription(
            UwbReadingArray,
            self.uwb_ranges_topic,
            self.uwb_ranges_sub_callback,
            10)
        
        # subscribe to the Husky's filtered Odometry
        self.last_odom_time = None
        self.declare_parameter('odom_topic', '/a200_1201/platform/odom/filtered')
        self.odom_topic = str(self.get_parameter('odom_topic').value)
        self.odom_subscriber = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_sub_callback,
            10)

        
        # read the provided UWB Map
        self.declare_parameter('uwb_map_topic', '/uwb/map')
        self.uwb_map_topic = str(self.get_parameter('uwb_map_topic').value)
        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
            )
        self.uwb_locations = None
        self.odom_subscriber = self.create_subscription(
            UwbPosMap,
            self.uwb_map_topic,
            self.uwb_map_sub_callback,
            map_qos
            )
        
        # publish the belief and covariance
        self.declare_parameter('ekf_topic', 'uwb_local/ekf')
        self.ekf_topic = str(self.get_parameter('ekf_topic').value)
        self.state_publisher = self.create_publisher(EkfState, self.ekf_topic, 10)
        timer_period = 0.05 # 20 Hz
        self.timer = self.create_timer(timer_period, self.ekf_publish_callback)
        
        
    def h_s_n(self, uwb_location: Point3D):
        # State | sx (self)x
        sx = self.state[0]
        sy = self.state[1]
        sz = self.uwb_height_on_robot
        # UWBS | ux (uwb)x
        ux = uwb_location.x
        uy = uwb_location.y
        uz = uwb_location.z
        return np.sqrt((sx - ux)**2 + (sy - uy)**2 + (uz - sz)**2)
        
    def J_h_s_n(self, uwb_location: Point3D):
        # State | sx (self)x
        sx = self.state[0]
        sy = self.state[1]
        sz = self.uwb_height_on_robot
        # UWBS | ux (uwb)x
        ux = uwb_location.x
        uy = uwb_location.y
        uz = uwb_location.z
        denom = np.sqrt((sx - ux)**2 + (sy - uy)**2 + (uz - sz)**2)
        if denom < 1e-6:
            return [0.0, 0.0, 0.0]
        else:
            return [(sx - ux)/(denom), 
                    (sy - uy)/(denom), 
                    0]
    
    def uwb_ranges_sub_callback(self, msg: UwbReadingArray):
        if len(msg.uwb_readings_array) == 0:
            return
        
        uwb_jacobians = []
        uwb_expected_measurements = []
        uwb_measurements = []
   
        for uwb_reading in msg.uwb_readings_array:
            anchor_id = uwb_reading.anchor_id
            distance = float(uwb_reading.distance_m)
            anchor_pos = uwb_reading.position_m
            quality = uwb_reading.quality
            stamp = uwb_reading.timestamp
            
            # update non mapped anchors with where the anchor thinks itself is
            if self.uwb_locations is None:
                self.uwb_locations = dict()
            if anchor_id not in self.uwb_locations:
                anchor_x = anchor_pos.x
                anchor_y = anchor_pos.y
                anchor_z = anchor_pos.z
                self.uwb_locations[anchor_id] = Point3D(anchor_x, anchor_y, anchor_z)
                
            uwb_jacobians.append(self.J_h_s_n(self.uwb_locations[anchor_id]))
            uwb_expected_measurements.append(self.h_s_n(self.uwb_locations[anchor_id]))
            uwb_measurements.append(distance)
        
        np_uwb_jacobians = np.array(uwb_jacobians, dtype=np.float64)
        np_uwb_expected_measurements = np.array(uwb_expected_measurements, dtype=np.float64)
        np_uwb_measurements = np.array(uwb_measurements, dtype=np.float64)
        sensor_noise_mat = np.diag([self.uwb_variance]*len(np_uwb_jacobians))
        cross_covariance = self.covariance @ np_uwb_jacobians.T
        kalman_gain = cross_covariance @ np.linalg.inv(np_uwb_jacobians @ self.covariance @ np_uwb_jacobians.T + sensor_noise_mat)
        new_belief = self.state + kalman_gain @ (np_uwb_measurements - np_uwb_expected_measurements)
        # new_covariance = (np.eye(len(self.state)) - kalman_gain @ np_uwb_jacobians) @ self.covariance
        # joseph form instead
        I = np.eye(len(self.state))
        new_covariance = (I - kalman_gain @ np_uwb_jacobians) @ self.covariance @ (I - kalman_gain @ np_uwb_jacobians).T + kalman_gain @ sensor_noise_mat @ kalman_gain.T
        
        self.state = new_belief
        self.state[2] = angle_wrap(self.state[2])
        self.covariance = new_covariance
        
        
    def J_F(self, v, dt, theta) -> list[float]:
        return np.array([[1, 0, -v*dt*np.sin(theta)],
                         [0, 1, v*dt*np.cos(theta)],
                         [0, 0, 1]])
        
    
    def odom_sub_callback(self, msg: Odometry):
        ''' when we get an odometry reading, we should perform a kalman predict step and maintain our covariance 
        TODO potentially skip predictions when the velocity is very low to reduce large covariance increases while sitting still'''
        t = rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds
        
        if self.last_odom_time is None:
            self.last_odom_time = t
            return
            
        dt = (t - self.last_odom_time)/(1.0e9)
        if dt <= 0:
            return
        
        self.last_odom_time = t
        v = float(msg.twist.twist.linear.x)
        if -1e-4 < v < 1e-4:
            v = 0.0
        omega = msg.twist.twist.angular.z
        if -1e-4 < omega < 1e-4:
            omega = 0
        # kalman predict
        # handle state update
        x = self.state[0]
        y = self.state[1]
        theta = self.state[2]
        self.state[0] = x + v*np.cos(theta)*dt
        self.state[1] = y + v*np.sin(theta)*dt
        self.state[2] = angle_wrap(theta + omega*dt)
        # covariance update
        J_F = self.J_F(v, dt, theta)
        new_cov = J_F @ self.covariance @ J_F.T + self.proc_noise_cov
        self.covariance = new_cov
            
    
    def uwb_map_sub_callback(self, msg: UwbPosMap):
        if self.uwb_locations is None:
            self.uwb_locations = dict()
        for uwb_pos in msg.uwb_positions_array:
            anchor_id = uwb_pos.anchor_id
            posx = float(uwb_pos.position_m.x)
            posy = float(uwb_pos.position_m.y)
            posz = float(uwb_pos.position_m.z)
            loc = Point3D(posx, posy, posz)
            self.uwb_locations[anchor_id] = loc
            
    def ekf_publish_callback(self):
        pose_msg = EkfState()
        '''
        builtin_interfaces/Time timestamp
        geometry_msgs/Point position_m
        float64 theta_rad
        '''
        pose_msg.timestamp = self.get_clock().now().to_msg()
        pose_msg.position_m.x = float(self.state[0])
        pose_msg.position_m.y = float(self.state[1])
        pose_msg.position_m.z = 0.0
        pose_msg.theta_rad = float(self.state[2])
        
        pose_msg.covariance = list(self.covariance.flatten())

        self.state_publisher.publish(pose_msg)


def main():
    rclpy.init()
    node = EkfLocalizationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    