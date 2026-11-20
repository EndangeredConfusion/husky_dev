# Transformer candidate pipeline bench (stages 1–5)

This bench connects timestamped GPS observations to grid cells, the Jetson BF16
Transformer, and metric A* paths. It does **not** execute paths, calculate real
robot rewards, or commit transitions to model history. MPPI and execution feedback
remain separate follow-up work.

## Jetson setup

Source baseline: `2c3ade3` plus the candidate integration changes in this directory
and the ROS packages. Keep the GPU model and venv in `~/transformer_deployment`.
The GPU worker and ROS adapter must run on the same machine (Unix socket).

```bash
cd ~/husky_ws
source /opt/ros/jazzy/setup.bash
~/transformer_deployment/venv/bin/python -m pip install \
  --target ~/transformer_deployment/ros_deps pathfinding==1.0.18
colcon build --build-base build_candidates --install-base install_candidates \
  --packages-select husky_interfaces husky_transformer_policy husky_transformer_bridge husky_rl
source install_candidates/setup.bash
export PYTHONPATH="$HOME/transformer_deployment/ros_deps:$PYTHONPATH"
export ROS_DOMAIN_ID=73  # verify this domain is unused first
export ROS_LOCALHOST_ONLY=1
ros2 node list --no-daemon
ros2 launch husky_transformer_bridge candidate_bench.launch.py
```

In another terminal, source the same environment and run:

```bash
cd ~/husky_ws
python3 bench/transformer_candidates/check_pipeline.py
```

The launch rejects domain 0 and hardcodes `enable_controller=false`. It starts
the worker, Transformer adapter, GPS discretizer, lambda publisher and A* only.
The test publishes **synthetic** GPS coordinates; never run it in the robot's
operational ROS domain. Stop the launch with Ctrl-C after testing.

## Interfaces and behavior

| Interface | Contract |
|---|---|
| `/gps/fix` | NavSatFix; valid status, finite coordinates, current ROS timestamp |
| `/agent_grid_cell` | Legacy GridCell for existing consumers; invalid/stale GPS stops repeats |
| `/agent_grid_cell/state` | JSON String: valid, cell, source_stamp, state_seq; periodic state is not a new transition |
| `/lambda_values` | Four raw values in [0,10], reliable/transient-local |
| `/rl_goal`, `/rl_goal_index` | Legacy candidate goal cell and index; no velocity command |
| `/transformer/decision` | Atomic decision ID, observation sequence, start, goal, source timestamp, logits |
| `/planner/result` | Decision-correlated planned/rejected/no_path result; path_metres in field86 local east/north |
| `/rl_path` | Float64MultiArray of metric cell centers, compatible with the existing follower |

The adapter polls at 10 Hz but proposes once per new cell/task or invalid-to-valid
recovery. It does not switch goals at 10 Hz or count repeated observations as
primitive steps. Inputs stay in raw training units; existing model packing owns
normalization. A pending response is discarded if the task/state changes or the
observation becomes invalid. No candidate is committed as an executed action.

`rl.launch.py` and `rl_sim.launch.py` accept `policy:=dqn|transformer`, with DQN as
the compatibility default. **Controller default is now false**; existing simulation
users must explicitly enable it when they intend motion. The standalone pipeline
expects a worker already running at `socket_path`; the candidate bench starts it.
Use the same ROS time source for GPS, discretizer, adapter and planner. GPS timestamps
older than 2 seconds, zero timestamps and future timestamps are rejected.

The A* route is a four-neighbor grid route converted through `field86_cells.yaml`,
not a continuous obstacle-aware navigation guarantee. Identical goal coordinates
with a new decision ID replan; duplicate decision IDs are suppressed. Existing
DQN GridCell goal handling remains available when decision_mode is false.

## Acceptance scope

The automated bench checks GPU-derived candidates reach metric paths, stationary
deduplication, same-goal replanning, duplicate decision suppression, invalid GPS,
GPS dropout, and absence of controller/DQN nodes and cmd_vel topics. This does not
establish field localization accuracy, collision avoidance, primitive-step fidelity,
or a history-updating execution loop.
