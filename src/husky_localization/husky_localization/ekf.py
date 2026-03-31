#!/usr/bin/env python3

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped


def angle_wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class EkfLocalizationNode(Node):
    """
    State:
        x = [px, py, theta, v, omega]^T

    Control/action:
        u = [v_cmd, omega_cmd]^T

    Measurement:
        z = [range_0, ..., range_n-1, v_odom, omega_odom]^T
        where n may change at every update step.
    """

    def __init__(self) -> None:
        super().__init__("ekf_localization")
        # Parameters
        self.declare_parameter("update_rate_hz", 20.0)
        self.declare_parameter("initial_state", [0.0, 0.0, 0.0, 0.0, 0.0])
        self.declare_parameter("initial_cov_diag", [2.0, 2.0, 2.0, 2.0, 2.0])
        self.declare_parameter("process_noise_diag", [0.05**2, 0.05**2, 0.1**2, 0.1**2, 0.05**2])
        self.declare_parameter("odom_noise_diag", [0.1, 0.1])
        self.declare_parameter("uwb_base_std", 1.0)
        self.declare_parameter("uwb_range_std_scale", 0.01)

        self.dt = float(self.get_parameter("dt").value)
        update_rate_hz = float(self.get_parameter("update_rate_hz").value)

        initial_state = np.array(self.get_parameter("initial_state").value, dtype=float)
        initial_cov_diag = np.array(self.get_parameter("initial_cov_diag").value, dtype=float)
        process_noise_diag = np.array(self.get_parameter("process_noise_diag").value, dtype=float)
        self.odom_noise_diag = np.array(self.get_parameter("odom_noise_diag").value, dtype=float)

        self.uwb_base_std = float(self.get_parameter("uwb_base_std").value)
        self.uwb_range_std_scale = float(self.get_parameter("uwb_range_std_scale").value)

        raw_markers = self.get_parameter("field_markers").value
        self.field_markers = np.array(raw_markers, dtype=float).reshape(-1, 2)

        self.x = initial_state.reshape(5, 1)
        self.P = np.diag(initial_cov_diag)
        self.Q = np.diag(process_noise_diag)

        self.last_action = np.zeros((2, 1), dtype=float)

        # Optional commanded-velocity input
        self.create_subscription(
            Float64MultiArray,
            "cmd_vel_ekf",
            self.cmd_callback,
            10,
        )

        # Outputs
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, "ekf_pose", 10)
        self.twist_pub = self.create_publisher(TwistWithCovarianceStamped, "ekf_twist", 10)

        self.timer = self.create_timer(1.0 / update_rate_hz, self.timer_callback)

        self.get_logger().info("EKF localization node started.")

    # ----------------------------
    # ROS callbacks
    # ----------------------------

    def cmd_callback(self, msg: Float64MultiArray) -> None:
        if len(msg.data) < 2:
            self.get_logger().warning("cmd_vel_ekf must contain [v_cmd, omega_cmd].")
            return
        self.last_action = np.array([[float(msg.data[0])], [float(msg.data[1])]], dtype=float)

    def timer_callback(self) -> None:
        uwb_measurements = self.get_uwbs()
        odom_measurement = self.get_odom()

        if odom_measurement is None:
            self.get_logger().warning("No odometry measurement available; skipping EKF update.")
            return

        self.x, self.P = self.ekf_step(
            state=self.x,
            action=self.last_action,
            cov=self.P,
            uwb_measurements=uwb_measurements,
            odom_measurement=odom_measurement,
            field_markers=self.field_markers,
            dt=self.dt,
        )

        self.publish_estimate()

    # ----------------------------
    # User-requested stubs
    # ----------------------------

    def get_uwbs(self) -> List[Tuple[int, float]]:
        """
        Stub.

        Return a list of (anchor_index, measured_range_m) pairs.
        Example:
            [(0, 12.4), (2, 8.9)]

        Requirements:
        - anchor_index should index into self.field_markers
        - omit invalid/dropout readings entirely
        - the number of returned UWB measurements may change every update
        """
        return []

    def get_odom(self) -> Optional[Tuple[float, float]]:
        """
        Stub.

        Return:
            (measured_linear_velocity_mps, measured_angular_velocity_radps)

        Return None if unavailable for the current cycle.
        """
        return 0.0, 0.0

    # ----------------------------
    # EKF core
    # ----------------------------

    def ekf_step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        cov: np.ndarray,
        uwb_measurements: Sequence[Tuple[int, float]],
        odom_measurement: Tuple[float, float],
        field_markers: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mirrors the original file's EKF structure:
        - predict using state/action
        - if no valid UWB readings: update with odom only
        - otherwise: update with [UWB..., odom]
        """
        x_pred = self.predict_state(state, action, dt)
        F = self.jacob_F(state, action, dt)
        P_pred = F @ cov @ F.T + self.Q

        usable_indices: List[int] = []
        usable_ranges: List[float] = []

        for anchor_idx, measured_range in uwb_measurements:
            if anchor_idx < 0 or anchor_idx >= len(field_markers):
                self.get_logger().warning(f"Ignoring out-of-range anchor index: {anchor_idx}")
                continue
            if measured_range <= 0.0:
                continue
            usable_indices.append(anchor_idx)
            usable_ranges.append(measured_range)

        z_odom = np.array([[float(odom_measurement[0])], [float(odom_measurement[1])]], dtype=float)

        if len(usable_indices) == 0:
            z = z_odom
            H = self.jacob_H(x_pred, field_markers=np.empty((0, 2)), include_uwb=False)
            z_hat = self.predict_measurements(x_pred, field_markers=np.empty((0, 2)), include_uwb=False)
            R = np.diag(self.odom_noise_diag)
        else:
            used_markers = field_markers[usable_indices]
            z_uwb = np.array(usable_ranges, dtype=float).reshape(-1, 1)
            z = np.vstack((z_uwb, z_odom))

            H = self.jacob_H(x_pred, field_markers=used_markers, include_uwb=True)
            z_hat = self.predict_measurements(x_pred, field_markers=used_markers, include_uwb=True)

            # Preserve the original idea: UWB noise scales with range, then odom appended. :contentReference[oaicite:1]{index=1}
            uwb_stds = self.uwb_base_std * (1.0 + self.uwb_range_std_scale * np.array(usable_ranges, dtype=float))
            R = np.diag([*uwb_stds.tolist(), *self.odom_noise_diag.tolist()])

        x_upd, P_upd = self.kalman_update(x_pred, P_pred, z, z_hat, H, R)
        x_upd[2, 0] = angle_wrap(x_upd[2, 0])

        return x_upd, P_upd

    def predict_state(self, state: np.ndarray, action: np.ndarray, dt: float) -> np.ndarray:
        """
        Same state convention as the source:
            [x, y, theta, v, omega]
        and same control convention:
            [v_cmd, omega_cmd]
        The source prediction/control logic stores commanded v, omega into the state. :contentReference[oaicite:2]{index=2}
        """
        x, y, theta, v, omega = state.flatten()
        v_cmd, omega_cmd = action.flatten()

        x_next = x + v * math.cos(theta) * dt
        y_next = y + v * math.sin(theta) * dt
        theta_next = angle_wrap(theta + omega * dt)

        return np.array(
            [[x_next], [y_next], [theta_next], [v_cmd], [omega_cmd]],
            dtype=float,
        )

    def jacob_F(self, state: np.ndarray, action: np.ndarray, dt: float) -> np.ndarray:
        x, y, theta, v, omega = state.flatten()
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

    def predict_measurements(
        self,
        state: np.ndarray,
        field_markers: np.ndarray,
        include_uwb: bool,
    ) -> np.ndarray:
        """
        Returns:
            if include_uwb:
                [range_0, ..., range_n-1, v, omega]^T
            else:
                [v, omega]^T
        """
        px, py, theta, v, omega = state.flatten()
        _ = theta

        parts: List[np.ndarray] = []

        if include_uwb and len(field_markers) > 0:
            dx = px - field_markers[:, 0]
            dy = py - field_markers[:, 1]
            ranges = np.sqrt(dx**2 + dy**2).reshape(-1, 1)
            parts.append(ranges)

        parts.append(np.array([[v], [omega]], dtype=float))
        return np.vstack(parts)

    def jacob_H(
        self,
        state: np.ndarray,
        field_markers: np.ndarray,
        include_uwb: bool,
    ) -> np.ndarray:
        """
        Dynamic measurement Jacobian.
        """
        px, py, theta, v, omega = state.flatten()
        _ = theta
        _ = omega

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

        rows.append(np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=float))  # v
        rows.append(np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=float))  # omega

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
        I = np.eye(P_pred.shape[0])
        P_upd = (I - K @ H) @ P_pred

        # Symmetrize numerically
        P_upd = 0.5 * (P_upd + P_upd.T)
        return x_upd, P_upd

    # ----------------------------
    # Publishing
    # ----------------------------

    def publish_estimate(self) -> None:
        px, py, theta, v, omega = self.x.flatten()

        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"

        pose_msg.pose.pose.position.x = float(px)
        pose_msg.pose.pose.position.y = float(py)
        pose_msg.pose.pose.position.z = 0.0

        half_theta = theta / 2.0
        pose_msg.pose.pose.orientation.z = math.sin(half_theta)
        pose_msg.pose.pose.orientation.w = math.cos(half_theta)

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