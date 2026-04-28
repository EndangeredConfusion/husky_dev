#!/usr/bin/env python3
'''
expected: some point array to parameters 'points_topic'
output: Twist message to /cmd_vel which will drive the robot along a continuous curve fit to the points topic
notes:
- every time the points array is published the robot will move. publishing only once per path is required.
'''

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped, Twist, Vector3
from std_msgs.msg import Float64MultiArray, Int32MultiArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from husky_interfaces.msg import UwbReading, UwbReadingArray, UwbPos, UwbPosMap, EkfState

import numpy as np
from scipy.interpolate import make_interp_spline


def angle_wrap(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi
        
        
class PointController(Node):
    def __init__(self):
        super().__init__("point_controller")

        # TODO points topic needs to exist
        self.declare_parameter('points_topic', '')
        self.points_topic = str(self.get_parameter('points_topic').value)
        
        # cmd vel publishment topic
        self.declare_parameter('cmd_vel_topic', '/a200_1201/cmd_vel')
        self.cmd_topic = str(self.get_parameter('cmd_vel_topic').value)
        
        
        # TODO points topic needs to exist
        self.declare_parameter('nominal_velocity', '.3')
        self.nominal_velocity = float(self.get_parameter('nominal_velocity').value)

        
        # ekf (localization) topic
        self.declare_parameter('localization_topic', 'uwb_local/ekf')
        self.localization_topic = str(self.get_parameter('uwb_local/ekf').value)
        
        self.state = None
        self.localization_subscriber = self.create_subscription(
            EkfState,
            self.localization_topic,
            self.localization_callback,
            10)
        
        self.point_array = None
        self.smoothed_points = None
        self.path_obj = None
        self.points_subscriber = self.create_subscription(
            Int32MultiArray,
            self.points_topic,
            self.point_callback,
            10)
            
        self.control_publisher = self.create_publisher(Twist, self.cmd_topic, 10)
        timer_period = .02 # 50 Hz
        self.timer = self.create_timer(timer_period, self.control_publish_callback)
    
    
    def chaikin_smooth(self, points, iterations=3, keep_ends=True):
        ''' chaikin smoothing algorihm to help spline cut corners '''
        pts = np.asarray(points, dtype=float)
        for _ in range(iterations):
            new_pts = []
            if keep_ends:
                new_pts.append(pts[0])
            for i in range(len(pts) - 1):
                p0 = pts[i]
                p1 = pts[i + 1]
                q = 0.75 * p0 + 0.25 * p1
                r = 0.25 * p0 + 0.75 * p1
                new_pts.extend([q, r])
            if keep_ends:
                new_pts.append(pts[-1])
            pts = np.array(new_pts)
        return pts
    
    
    def fit_continuous_path(path, npoints=75):
        ''' applies third degree spline interpolation on the smoothed points '''
        path_array = np.array(path)
        # split by columns to get X and Y
        # x, y = path_array[:, 0], path_array[:, 1]
        # create parameter
        t = np.linspace(0, 1, len(path_array))
        # interpolate stacked x and y as functions of t
        spl = make_interp_spline(t, path_array, k=3)

        return spl
    
    
    def point_callback(self, msg: Int32MultiArray):
        ''' updates internal continuous path with the published desired points '''
        stride = 2
        xs = msg.data[::stride]
        ys = msg.data[1::stride]
        self.point_array = zip(xs, ys)
        self.smoothed_points = self.chaikin_smooth(self.point_array)
        self.path_obj = self.fit_continuous_path(self.point_array)
        
        
    def localization_callback(self, msg: EkfState):
        ''' updates internal state based off the localization topic'''
        mdata = msg.data
        x = mdata.position_m.x
        y = mdata.position_m.y
        theta = mdata.theta_rad
        self.state = [x, y, theta]
    
    
    def control_publish_callback(self):
        ''' Implementing Stanley control along the continuous interpolated path
        https://ai.stanford.edu/~gabeh/papers/hoffmann_stanley_control07.pdf '''
        x, y, theta = self.state

        t_vals = np.linspace(0, 1, 150)
        path_pts = self.path_obj(t_vals)
        
        # we want to minimize (pathx - x)^2 + (pathy - y)^2
        deltas = path_pts - np.array((x, y))
        sq_dists = np.sum(deltas * deltas, axis=1)
        min_idx = np.argmin(sq_dists)
        t_star = t_vals[min_idx]
        path_point = path_pts(t_star)
        px, py = path_point
        path_deriv = self.path_obj.derivative(1)(t_star)

        # find cross track error with already computed distances
        error_vec = np.array([(x - px), (y - py)])
        tangent_norm_vec = path_deriv / (np.linalg.norm(path_deriv) + 1e-9)
        # invert terms to find the perpendicular normal vector
        perpendicular_path_vec = np.array([-tangent_norm_vec[1], tangent_norm_vec[0]])
        cross_track_e = np.dot(error_vec, perpendicular_path_vec)

        px_prime, py_prime = path_deriv
        path_theta = np.arctan2(py_prime, px_prime)
        angle_error = angle_wrap(path_theta - theta)
        
        # stanley correction constants
        v = self.nominal_velocity
        k = 1
        cross_track_e_term = np.arctan2(k * cross_track_e, abs(v) + 1e-6)
        
        k_omega = 1
        k_cross = 1
        desired_omega = k_omega * angle_error + k_cross * cross_track_e_term
        # reverse heading tendency as robot might turn opposite way TODO check this
        desired_omega *= -1
        # no point in as much if we are facing in the wrong direction
        # still will move a bit to help localization
        if -np.pi < angle_error < np.pi:
            desired_v = self.nominal_velocity
        else:
            desired_v = self.nominal_velocity / 2
        
        twist_msg = Twist()
        '''
        geometry_msgs/Vector3 linear (x, y, z) (x is velocity)
        geometry_msgs/Vector3 angular (x, y, z) (z is angular)
        '''
        twist_msg.linear.x = desired_v
        twist_msg.angular.z = desired_omega
        # all other fields are zero

        self.control_publisher.publish(twist_msg)


def main():
    rclpy.init()
    node = PointController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    