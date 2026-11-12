"""Pure helpers for gps_pose (no ROS), so they can be unit-tested."""

import math

HEADING_CONVENTIONS = ('cw_from_north', 'enu_yaw')


def angle_wrap(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Return the yaw (rotation about z) of a quaternion."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def heading_to_enu_yaw(angle_rad: float, convention: str,
                       offset_rad: float = 0.0) -> float:
    """Convert a GPS heading to the controller's yaw.

    The controller wants yaw in the local frame: 0 = facing east, counter-
    clockwise positive (pi/2 = facing north).

    convention:
      'cw_from_north' -- angle is a compass heading (0 = north, clockwise
                         positive), as u-blox reports dual-antenna heading.
      'enu_yaw'       -- angle is already ENU yaw.
    offset_rad is added afterwards (antenna baseline vs robot forward axis).
    VERIFY ON HARDWARE: which convention the chosen driver uses is not known yet.
    """
    if convention == 'cw_from_north':
        yaw = math.pi / 2.0 - angle_rad
    elif convention == 'enu_yaw':
        yaw = angle_rad
    else:
        raise ValueError(
            f'heading_convention must be one of {HEADING_CONVENTIONS}, got {convention!r}')
    return angle_wrap(yaw + offset_rad)


def antenna_to_base(east_m: float, north_m: float, yaw_rad: float,
                    offset_x_m: float, offset_y_m: float):
    """Shift an antenna position to the robot centre.

    (offset_x_m, offset_y_m) is the antenna position in the robot frame
    (x forward, y left).  Returns the robot centre (east_m, north_m).
    """
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return (east_m - (c * offset_x_m - s * offset_y_m),
            north_m - (s * offset_x_m + c * offset_y_m))


def enu_yaw_to_compass(yaw_rad: float) -> float:
    """ENU yaw (0 = east, CCW) -> compass heading (0 = north, CW), in [0, 2*pi)."""
    return (math.pi / 2.0 - yaw_rad) % (2.0 * math.pi)


def odom_to_local(x: float, y: float, yaw: float,
                  start_east_m: float, start_north_m: float, start_yaw_rad: float):
    """Place a simulator odometry pose into the field's local frame.

    The odom origin (0, 0, yaw 0) is put at (start_east_m, start_north_m)
    facing start_yaw_rad (ENU).  Returns (east_m, north_m, yaw_rad).
    """
    c, s = math.cos(start_yaw_rad), math.sin(start_yaw_rad)
    return (start_east_m + c * x - s * y,
            start_north_m + s * x + c * y,
            angle_wrap(start_yaw_rad + yaw))
