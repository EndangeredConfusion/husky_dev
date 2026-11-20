"""Run against candidate_bench.launch.py in an unused nonzero ROS domain."""
import json
import os
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import String, Float64MultiArray
from husky_rl.cell_table import CellTable, default_table_path


def main():
    assert 1 <= int(os.environ.get('ROS_DOMAIN_ID', '0')) <= 232
    rclpy.init()
    node = Node('candidate_bench_check')
    results, states, decisions = [], [], []
    paths = []
    node.create_subscription(Float64MultiArray, '/rl_path', lambda m: paths.append(list(m.data)), 10)
    node.create_subscription(String, '/planner/result', lambda m: results.append(json.loads(m.data)), 10)
    node.create_subscription(String, '/agent_grid_cell/state', lambda m: states.append(json.loads(m.data)), 10)
    node.create_subscription(String, '/transformer/decision', lambda m: decisions.append(json.loads(m.data)), 10)
    gps = node.create_publisher(NavSatFix, '/gps/fix', 10)
    table = CellTable.load(default_table_path())
    start = (4, 3)
    latitude, longitude = table.frame.to_latlon(*table.center(start))
    decision_pub = node.create_publisher(String, '/transformer/decision', 10)

    def run(seconds, valid=None, stale=False, off_walkway=False):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if valid is not None:
                fix = NavSatFix()
                fix.header.stamp = node.get_clock().now().to_msg()
                if stale:
                    fix.header.stamp.sec -= 10
                fix.latitude, fix.longitude = latitude, longitude
                if off_walkway:
                    fix.latitude += 0.01
                fix.status.status = 0 if valid else -1
                gps.publish(fix)
            rclpy.spin_once(node, timeout_sec=0.1)

    try:
        run(15, True)
        assert decisions, 'No Transformer decision'
        matching = [r for r in results if r['decision_id'] == decisions[-1]['decision_id']]
        assert matching and matching[-1]['status'] == 'planned', results
        path = matching[-1]
        assert path['cells'][0] == list(start) and path['cells'][-1] == decisions[-1]['goal']
        assert len(path['cells']) > 1
        assert len(path['path_metres']) == 2 * len(path['cells'])
        table = CellTable.load(default_table_path())
        expected = table.path_to_local([tuple(c) for c in path['cells']])
        assert path['path_metres'] == expected and expected in paths
        assert all(abs(a[0]-b[0]) + abs(a[1]-b[1]) == 1
                   for a, b in zip(path['cells'], path['cells'][1:]))
        assert decisions[-1]['history_version'] == 0
        count = len(decisions)
        run(3, True)
        assert len(decisions) == count, 'Repeated stationary fixes generated new decisions'
        # Explicit new decision with same goal must replan; duplicate ID must not.
        replay = dict(decisions[-1], decision_id='bench-same-goal',
                      source_stamp=node.get_clock().now().nanoseconds * 1e-9)
        decision_pub.publish(String(data=json.dumps(replay)))
        run(1, True)
        assert sum(r['decision_id'] == 'bench-same-goal' for r in results) == 1
        decision_pub.publish(String(data=json.dumps(replay)))
        run(0.5, True)
        assert sum(r['decision_id'] == 'bench-same-goal' for r in results) == 1
        run(3, False)
        assert states and states[-1]['valid'] is False
        run(3, True)
        run(3)
        assert states[-1]['valid'] is False, 'Dropout retained valid position'
        run(3, True)
        run(3, True, stale=True)
        assert states[-1]['valid'] is False, 'Old GPS timestamp accepted'
        run(3, True)
        run(3, True, off_walkway=True)
        assert states[-1]['valid'] is False, 'Off-walkway position accepted'
        names = node.get_node_names()
        assert not any('controller' in n or n == 'rl_policy_node' for n in names), names
        topics = node.get_topic_names_and_types()
        assert not any('cmd_vel' in n for n, _ in topics), topics
        print(json.dumps(dict(passed=True, nodes=names, first_path=path,
                              transformer_decisions=len(decisions),
                              checks=['gpu_to_metric_path', 'stationary_dedup',
                                      'same_goal_new_decision', 'duplicate_decision',
                                      'invalid_fix', 'gps_dropout', 'stale_timestamp',
                                      'off_walkway', 'no_controller_or_cmd_vel'])))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
