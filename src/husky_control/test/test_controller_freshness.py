"""Exercise real controller callbacks without constructing a node or publisher."""
from types import SimpleNamespace as NS
from unittest.mock import Mock
import time
from husky_control.point_controller import PointController


def controller():
    c = NS(state=(0., 0., 0.), pose_received_at=time.monotonic(), pose_stamp=10.,
           pose_timeout_s=1., follower=Mock(active=True), publish_cmd=Mock(),
           get_clock=lambda: NS(now=lambda: NS(nanoseconds=10_000_000_000)),
           get_logger=lambda: Mock())
    c.pose_fresh = lambda: PointController.pose_fresh(c)
    c.stop_path = lambda reason: PointController.stop_path(c, reason)
    return c


def test_receipt_timeout_stops_and_clears_path():
    c = controller()
    c.pose_received_at -= 2
    PointController.control_publish_callback(c)
    c.publish_cmd.assert_called_once_with(0., 0.)
    c.follower.clear.assert_called_once()
    c.follower.step.assert_not_called()


def test_old_source_timestamp_stops_despite_recent_receipt():
    c = controller()
    c.pose_stamp = 8.
    PointController.control_publish_callback(c)
    c.publish_cmd.assert_called_once_with(0., 0.)
    c.follower.step.assert_not_called()


def test_invalid_or_future_pose_invalidates_cached_state():
    for x, sec in [(float('nan'), 10), (0., 12)]:
        c = controller()
        msg = NS(position_m=NS(x=x, y=0.), theta_rad=0., timestamp=NS(sec=sec, nanosec=0))
        PointController.localization_callback(c, msg)
        assert c.state is None and c.pose_received_at is None
        c.publish_cmd.assert_called_once_with(0., 0.)


def test_fresh_pose_runs_existing_follower():
    c = controller()
    c.follower.step.return_value = (.3, .1, False)
    PointController.control_publish_callback(c)
    c.publish_cmd.assert_called_once_with(.3, .1)
    c.follower.clear.assert_not_called()


def test_stale_pose_rejects_new_path():
    c = controller()
    c.pose_stamp = 0.
    PointController.point_callback(c, NS(data=[0., 0., 1., 1.]))
    c.follower.set_path.assert_not_called()
    c.publish_cmd.assert_called_once_with(0., 0.)
