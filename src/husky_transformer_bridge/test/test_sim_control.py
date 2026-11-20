"""Control callback checks without constructing a ROS node or publisher."""
import json
from types import SimpleNamespace as NS
from unittest.mock import Mock
from husky_transformer_bridge.sim_coordinator import SimCoordinator


def coordinator():
    session = Mock(commit_request=None, task={'epoch': 'new', 'history_version': 0})
    return NS(stopped=True, state={'busy': False}, initialized=True, session=session,
              lam=[7,2,2,2], command=Mock(), emit=Mock(), fault='limit', committed=6)


def test_reset_waits_for_executor_initialization_ack():
    c = coordinator()
    def command(op, **kwargs):
        assert op == 'initialize'
        assert c.initialized is False
    c.command = command
    SimCoordinator.on_control(c, NS(data=json.dumps({'op': 'reset', 'lambda_values': [1,7,2,2]})))
    assert c.fault is None and c.committed == 0 and c.initialized is False
    assert c.lam == [1,7,2,2]


def test_reset_rejects_running_execution():
    c = coordinator()
    c.state['busy'] = True
    SimCoordinator.on_control(c, NS(data='{"op":"reset"}'))
    c.session.initialize.assert_not_called()
    c.command.assert_not_called()
    assert c.emit.call_args.args[0] == 'control_rejected'


def test_reset_preserves_uncertain_commit():
    c = coordinator()
    c.session.commit_request = {'ticket': 'unacknowledged'}
    SimCoordinator.on_control(c, NS(data='{"op":"reset"}'))
    c.session.reset_after_worker_loss.assert_not_called()
    c.command.assert_not_called()
    assert c.emit.call_args.args[0] == 'control_rejected'
