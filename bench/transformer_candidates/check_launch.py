"""Check policy exclusivity and disabled control without starting any process."""
import importlib.util
from pathlib import Path
from launch import LaunchContext
from launch_ros.actions import Node
from launch.utilities import perform_substitutions, normalize_to_list_of_substitutions

root = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('rl_launch', root / 'src/husky_rl/launch/rl.launch.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for policy in ('dqn', 'transformer'):
    context = LaunchContext()
    context.launch_configurations.update(policy=policy, enable_controller='false', use_gps_pose='false')
    active = []
    for action in module.generate_launch_description().entities:
        if isinstance(action, Node) and (action.condition is None or action.condition.evaluate(context)):
            active.append(perform_substitutions(context, normalize_to_list_of_substitutions(action.node_executable)))
    assert 'point_controller' not in active
    assert ('rl_policy' in active) == (policy == 'dqn'), active
    assert ('transformer_policy' in active) == (policy == 'transformer'), active
    print(policy, active)
