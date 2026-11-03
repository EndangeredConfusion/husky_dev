#!/usr/bin/env bash
set -eo pipefail
bench=$(cd "$(dirname "$0")" && pwd)
export HUSKY_WS=${HUSKY_WS:-$(cd "$bench/../.." && pwd)}
deployment=${TRANSFORMER_DEPLOYMENT:-$HOME/transformer_deployment}
source /opt/ros/jazzy/setup.bash
source "$HUSKY_WS/install/setup.bash"
export TRAINING_WHEEL=${TRAINING_WHEEL:-$deployment/checks/goal_navfn/pathfinding-1.0.20-py3-none-any.whl}
[[ -r "$TRAINING_WHEEL" ]] || { echo "Missing pathfinding wheel: $TRAINING_WHEEL" >&2; exit 1; }
export ROS_DOMAIN_ID=87 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
unset ROS_STATIC_PEERS ROS_DISCOVERY_SERVER
nodes=$(ros2 node list --no-daemon --spin-time 2)
[[ -z "$nodes" ]] || { echo "Domain 87 is occupied: $nodes" >&2; exit 1; }
mkdir -p "$deployment/checks/goal_navfn"
run=$(mktemp -d "$deployment/checks/goal_navfn/run.XXXXXXXX")
cd "$run"
echo "Results and logs: $run"
export BENCH_SOCKET="$run/worker.sock"
children=()
cleanup() {
  for pid in "${children[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${children[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT TERM
"$deployment/venv/bin/python" -m husky_transformer_policy.policy_worker --model-dir "$deployment/model" --socket "$BENCH_SOCKET" >worker.log 2>&1 &
worker=$!
children+=("$worker")
/opt/ros/jazzy/lib/tf2_ros/static_transform_publisher --frame-id map --child-frame-id bench_base >tf.log 2>&1 &
children+=("$!")
/opt/ros/jazzy/lib/nav2_planner/planner_server --ros-args --params-file "$bench/planner.yaml" >planner.log 2>&1 &
children+=("$!")
for ((i=0; i<120; i++)); do
  kill -0 "$worker"
  [[ -S "$BENCH_SOCKET" ]] && break
  sleep 0.5
done
[[ -S "$BENCH_SOCKET" ]] || { echo "Worker startup timed out" >&2; exit 1; }
for transition in configure activate; do
  timeout 20 ros2 lifecycle set /planner_server "$transition"
done
/usr/bin/python3 "$bench/bench.py"
