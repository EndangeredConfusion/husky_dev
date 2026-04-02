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

from your_msgs_package.msg import UwbRanges, UwbMap  # TODO: replace


def angle_wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass
class AnchorMeasurement:
    anchor_id: str
    distance_m: float
    position_xy: np.ndarray


class EkfLocalizationNode(Node):
    def __init__(self) -> None:
        super().__init__("ekf_localization")

        self.declare_parameter("update_rate_hz", 20.0)
        self.declare_parameter("initial_state", [0.0, 0.0, 0.0, 0.0, 0.0])
        self.declare_parameter("initial_cov_diag", [2.0, 2.0, 2.0, 2.0, 2.0])
        self.declare_parameter("process_noise_diag", [0.05**2, 0.05**2, 0.1**2, 0.1**2, 0.05**2])
        self.declare_parameter("odom_noise_diag", [0.1, 0.1])
        self.declare_parameter("uwb_base_std", 1.0)
        self.declare_parameter("uwb_range_std_scale", 0.01)

        self.declare_parameter("odom_topic", "/a200_1201/platform/odom/filtered")
        self.declare_parameter("uwb_ranges_topic", "/uwb/ranges")
        self.declare_parameter("uwb_map_topic", "/uwb/map")

        self.declare_parameter("min_dt", 1e-3)
        self.declare_parameter("max_dt", 0.25)

        update_rate_hz = float(self.get_parameter("update_rate_hz").value)

        initial_state = np.array(self.get_parameter("initial_state").value, dtype=float)
        initial_cov_diag = np.array(self.get_parameter("initial_cov_diag").value, dtype=float)
        process_noise_diag = np.array(self.get_parameter("process_noise_diag").value, dtype=float)
        self.odom_noise_diag = np.array(self.get_parameter("odom_noise_diag").value, dtype=float)

        self.uwb_base_std = float(self.get_parameter("uwb_base_std").value)
        self.uwb_range_std_scale = float(self.get_parameter("uwb_range_std_scale").value)

        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.uwb_ranges_topic = str(self.get_parameter("uwb_ranges_topic").value)
        self.uwb_map_topic = str(self.get_parameter("uwb_map_topic").value)

        self.min_dt = float(self.get_parameter("min_dt").value)
        self.max_dt = float(self.get_parameter("max_dt").value)

        self.x = initial_state.reshape(5, 1)
        self.P = np.diag(initial_cov_diag)
        self.Q_base = np.diag(process_noise_diag)

        self.last_action = np.zeros((2, 1), dtype=float)

        self.latest_odom: Optional[Tuple[float, float]] = None
        self.latest_uwb_measurements: List[AnchorMeasurement] = []
        self.anchor_map_xy: Dict[str, np.ndarray] = {}

        self.last_filter_time: Optional[Time] = None

        self.create_subscription(
            Float64MultiArray,
            "cmd_vel_ekf",
            self.cmd_callback,
            10,
        )

        self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            20,
        )
        self.create_subscription(
            UwbRanges,
            self.uwb_ranges_topic,
            self.uwb_ranges_callback,
            20,
        )
        self.create_subscription(
            UwbMap,
            self.uwb_map_topic,
            self.uwb_map_callback,
            10,
        )

        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, "ekf_pose", 10)
        self.twist_pub = self.create_publisher(TwistWithCovarianceStamped, "ekf_twist", 10)

        self.timer = self.create_timer(1.0 / update_rate_hz, self.timer_callback)

    def cmd_callback(self, msg: Float64MultiArray) -> None:
        if len(msg.data) < 2:
            self.get_logger().warning("cmd_vel_ekf must contain [v_cmd, omega_cmd].")
            return
        self.last_action = np.array([[float(msg.data[0])], [float(msg.data[1])]], dtype=float)

    def odom_callback(self, msg: Odometry) -> None:
        v = float(msg.twist.twist.linear.x)
        omega = float(msg.twist.twist.angular.z)
        self.latest_odom = (v, omega)

    def uwb_map_callback(self, msg: UwbMap) -> None:
        new_map: Dict[str, np.ndarray] = {}
        for anchor in msg.uwb_positions_array:
            anchor_id = str(anchor.anchor_id)
            new_map[anchor_id] = np.array(
                [float(anchor.position_m.x), float(anchor.position_m.y)],
                dtype=float,
            )
        self.anchor_map_xy = new_map

    def uwb_ranges_callback(self, msg: UwbRanges) -> None:
        measurements: List[AnchorMeasurement] = []

        for reading in msg.uwb_readings_array:
            anchor_id = str(reading.anchor_id)
            distance_m = float(reading.distance_m)
            if distance_m <= 0.0:
                continue

            if anchor_id in self.anchor_map_xy:
                pos_xy = self.anchor_map_xy[anchor_id]
            else:
                pos_xy = np.array(
                    [float(reading.position_m.x), float(reading.position_m.y)],
                    dtype=float,
                )

            measurements.append(
                AnchorMeasurement(
                    anchor_id=anchor_id,
                    distance_m=distance_m,
                    position_xy=pos_xy,
                )
            )

        self.latest_uwb_measurements = measurements

    def timer_callback(self) -> None:
        if self.latest_odom is None:
            return

        now = self.get_clock().now()

        if self.last_filter_time is None:
            self.last_filter_time = now
            return

        dt = (now - self.last_filter_time).nanoseconds * 1e-9
        self.last_filter_time = now

        if dt <= 0.0:
            self.get_logger().warning(f"Non-positive dt={dt}, skipping update.")
            return

        dt = max(self.min_dt, min(dt, self.max_dt))

        self.x, self.P = self.ekf_step(
            state=self.x,
            action=self.last_action,
            cov=self.P,
            uwb_measurements=self.latest_uwb_measurements,
            odom_measurement=self.latest_odom,
            dt=dt,
        )

        self.publish_estimate()

    def ekf_step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        cov: np.ndarray,
        uwb_measurements: Sequence[AnchorMeasurement],
        odom_measurement: Tuple[float, float],
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        x_pred = self.predict_state(state, action, dt)
        F = self.jacob_F(state, action, dt)

        # Simple continuous-to-discrete approximation:
        # smaller dt -> less accumulated process noise
        Q = self.Q_base * dt
        P_pred = F @ cov @ F.T + Q

        usable_positions: List[np.ndarray] = []
        usable_ranges: List[float] = []

        for meas in uwb_measurements:
            if meas.distance_m <= 0.0:
                continue
            usable_positions.append(meas.position_xy)
            usable_ranges.append(meas.distance_m)

        z_odom = np.array([[float(odom_measurement[0])], [float(odom_measurement[1])]], dtype=float)

        if len(usable_positions) == 0:
            markers = np.empty((0, 2), dtype=float)
            z = z_odom
            H = self.jacob_H(x_pred, markers, include_uwb=False)
            z_hat = self.predict_measurements(x_pred, markers, include_uwb=False)
            R = np.diag(self.odom_noise_diag)
        else:
            markers = np.vstack(usable_positions).reshape(-1, 2)
            z_uwb = np.array(usable_ranges, dtype=float).reshape(-1, 1)
            z = np.vstack((z_uwb, z_odom))

            H = self.jacob_H(x_pred, markers, include_uwb=True)
            z_hat = self.predict_measurements(x_pred, markers, include_uwb=True)

            uwb_stds = self.uwb_base_std * (
                1.0 + self.uwb_range_std_scale * np.array(usable_ranges, dtype=float)
            )
            R = np.diag([*uwb_stds.tolist(), *self.odom_noise_diag.tolist()])

        return self.kalman_update(x_pred, P_pred, z, z_hat, H, R)

    def predict_state(self, state: np.ndarray, action: np.ndarray, dt: float) -> np.ndarray:
        x, y, theta, v, omega = state.flatten()
        v_cmd, omega_cmd = action.flatten()

        x_next = x + v * math.cos(theta) * dt
        y_next = y + v * math.sin(theta) * dt
        theta_next = angle_wrap(theta + omega * dt)

        return np.array([[x_next], [y_next], [theta_next], [v_cmd], [omega_cmd]], dtype=float)

    def jacob_F(self, state: np.ndarray, action: np.ndarray, dt: float) -> np.ndarray:
        _, _, theta, v, _ = state.flatten()
        _ = action

        return np.array(
            [
                [1.0, 0.0, -dt * v * math.sin(theta), dt * math.cos(theta), 0.0],
                [0.0, 1.0,  dt * v * math.cos(theta), dt * math.sin(theta), 0.0],
                [0.0, 0.0, 1.0,                         0.0,                dt],
                [0.0, 0.0, 0.0,                         0.0,                0.0],
                [0.0, 0.0, 0.0,                         0.0,                0.0],
            ],
            dtype=float,
        )

    def predict_measurements(self, state: np.ndarray, field_markers: np.ndarray, include_uwb: bool) -> np.ndarray:
        px, py, _, v, omega = state.flatten()
        parts: List[np.ndarray] = []

        if include_uwb and len(field_markers) > 0:
            dx = px - field_markers[:, 0]
            dy = py - field_markers[:, 1]
            ranges = np.sqrt(dx**2 + dy**2).reshape(-1, 1)
            parts.append(ranges)

        parts.append(np.array([[v], [omega]], dtype=float))
        return np.vstack(parts)

    def jacob_H(self, state: np.ndarray, field_markers: np.ndarray, include_uwb: bool) -> np.ndarray:
        px, py, _, _, _ = state.flatten()
        rows: List[np.ndarray] = []

        if include_uwb and len(field_markers) > 0:
            for marker_x, marker_y in field_markers:
                dx = px - marker_x
                dy = py - marker_y
                r = math.hypot(dx, dy)
                if r < 1e-9:
                    dr_dx = 0.0
                    dr_dy = 0.0
                else:
                    dr_dx = dx / r
                    dr_dy = dy / r
                rows.append(np.array([dr_dx, dr_dy, 0.0, 0.0, 0.0], dtype=float))

        rows.append(np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=float))
        rows.append(np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=float))
        return np.vstack(rows)

    def kalman_update(
        self,
        x_pred: np.ndarray,
        P_pred: np.ndarray,
        z: np.ndarray,
        z_hat: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        y = z - z_hat
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        x_upd = x_pred + K @ y
        x_upd[2, 0] = angle_wrap(x_upd[2, 0])

        I = np.eye(P_pred.shape[0])
        P_upd = (I - K @ H) @ P_pred
        P_upd = 0.5 * (P_upd + P_upd.T)

        return x_upd, P_upd

    def publish_estimate(self) -> None:
        px, py, theta, v, omega = self.x.flatten()

        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.pose.position.x = float(px)
        pose_msg.pose.pose.position.y = float(py)
        pose_msg.pose.pose.position.z = 0.0
        pose_msg.pose.pose.orientation.z = math.sin(theta / 2.0)
        pose_msg.pose.pose.orientation.w = math.cos(theta / 2.0)

        pose_cov = np.zeros((6, 6), dtype=float)
        pose_cov[0, 0] = self.P[0, 0]
        pose_cov[1, 1] = self.P[1, 1]
        pose_cov[5, 5] = self.P[2, 2]
        pose_msg.pose.covariance = pose_cov.reshape(-1).tolist()

        twist_msg = TwistWithCovarianceStamped()
        twist_msg.header = pose_msg.header
        twist_msg.twist.twist.linear.x = float(v)
        twist_msg.twist.twist.angular.z = float(omega)

        twist_cov = np.zeros((6, 6), dtype=float)
        twist_cov[0, 0] = self.P[3, 3]
        twist_cov[5, 5] = self.P[4, 4]
        twist_msg.twist.covariance = twist_cov.reshape(-1).tolist()

        self.pose_pub.publish(pose_msg)
        self.twist_pub.publish(twist_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EkfLocalizationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()