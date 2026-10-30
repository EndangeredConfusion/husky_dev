# Transformer interface: stages 0 and 1

This adds a versioned ROS-to-GPU communication skeleton, without changing the
existing DQN, localization or motion packages. It is not a complete robot policy
deployment: live state adaptation, calibration, physical feedback and full ROS
10 Hz acceptance remain later stages.

## Layout and interpreters

| Component | Source | Runtime |
| --- | --- | --- |
| ROS JSON request/response bridge | `src/husky_transformer_bridge` | System `/usr/bin/python3`, ROS Jazzy |
| Local socket protocol and GPU policy worker | `src/husky_transformer_policy` | Existing GPU virtual environment |
| Contract and model identity | `docs/transformer` | Documentation/configuration |

The policy package's `__init__` and protocol do not import torch. The ROS process
can therefore use the protocol without installing torch into system Python.
The GPU process never imports rclpy. Both packages are installed into the workspace
overlay; the GPU interpreter inherits that overlay's `PYTHONPATH`.

Keep these external assets under `~/transformer_deployment`:

- `model/model.pt`, `config.json`, `ad_transformer.py`, `ad_online.py`;
- `venv/` with Python 3.12, torch 2.10.0+cu130 and NumPy 1.26.4;
- logs, recordings, benchmarks and check results.

See `model_manifest.json` for expected hashes and runtime versions. The worker
checks the checkpoint hash before loading. The original model code owns input
normalization; ROS/IPC sends raw grid positions, lambda and macro rewards.
Do not install the archived training requirements into system Python or blindly
replace the tested Jetson CUDA environment. The previously observed cuSPARSELt
wheel-platform-tag `pip check` exception is documented despite successful GPU
operation; it is not a claim that dependency checks are universally clean.

## Build on Jetson

Use system Python for colcon; do not activate the GPU venv for this build.

```bash
source /opt/ros/jazzy/setup.bash
cd ~/husky_ws
colcon build --packages-select husky_transformer_policy husky_transformer_bridge
source install/setup.bash
```

No torch installation or model download is performed by the build. Re-source the
overlay in each test terminal. Keep the source tree and install directory at their
build locations; rebuild if moving the workspace.

## Isolated launch

First check domain 42 is not being used by another project. The launch and bridge
reject domain 0. All application ROS traffic is under `/transformer/shadow`.
Standard ROS logging and parameter services also exist in the isolated domain.

```bash
ROS_DOMAIN_ID=42 ros2 node list --no-daemon --spin-time 3
ros2 launch husky_transformer_bridge shadow.launch.py
```

The launch explicitly starts the worker with the GPU interpreter and the bridge
with system Python. It loads weights and warms CUDA before printing `READY`.
Wait for `READY` before sending requests. A request made earlier returns an error;
the bridge does not queue work indefinitely or reuse an old prediction.
If either child exits, launch shuts down the other child too.

Optional launch arguments are `gpu_python`, `model_dir`, `socket_path` and
`domain_id`. Their defaults use the existing deployment directory, socket
`/tmp/husky_policy_stage1.sock`, and domain 42. Only use another domain after
checking that it is isolated. The acceptance client currently targets domain 42
and the default topic names.

The ROS test envelope uses `std_msgs/String` with JSON on
`/transformer/shadow/request` and `/transformer/shadow/response`. This is deliberately
a stage-1 RPC envelope, not the future typed StateSnapshot/MacroTransition package.
There are no navigation clients, goal publishers, or velocity publishers.
Do not use the older `husky_rl/rl.launch.py` for these checks: it starts a controller.

## Acceptance checks

Run from a result directory outside the repository, with the overlay sourced:

```bash
mkdir -p ~/transformer_deployment/checks/workspace_interface
cd ~/transformer_deployment/checks/workspace_interface
python3 ~/husky_ws/src/husky_transformer_policy/test/check_transport.py
~/transformer_deployment/venv/bin/python \
  ~/husky_ws/src/husky_transformer_policy/test/check_model_contract.py \
  --model-dir ~/transformer_deployment/model
python3 ~/husky_ws/src/husky_transformer_bridge/test/check_ros.py
```

The model check covers exact raw-input normalization, BF16 equality against a
direct model call, non-mutating repeated inference, actual-action commits,
idempotent retry of the last commit, stale/invalid request rejection and task
reset. The ROS check performs task initialization, ten predictions and a stale
epoch rejection. Outputs are `*_result.json`, not source-controlled recordings.

To check unavailable-worker propagation, stop the launch with Ctrl-C, then run
only the bridge in one terminal:

```bash
/usr/bin/python3 -m husky_transformer_bridge.ros_bridge \
  --socket /tmp/husky_policy_stage1.sock --domain-id 42
```

In the other terminal, run `check_ros.py --expect-unavailable`. Stop the bridge
afterward. The transport check separately verifies that an unresponsive endpoint
times out. Never delete a socket belonging to a live worker; worker startup
refuses to overwrite existing sockets, and graceful shutdown removes its socket.

Feedback used by these checks is explicitly synthetic. Passing them does not
demonstrate navigation performance, physical reward correctness or motion.

## Boundaries

- The response polling timer is 10 Hz. This is not measured 100 ms end-to-end
  acceptance; the transport failure timeout is separately set to one second.
- History is in memory and capped at 2399 completed records plus the current
  observation. Restart requires an explicit new task.
- The last identical commit retry is acknowledged once; old/changed requests are
  rejected. At most 32 prediction tickets are retained.
- Task lambda is fixed per epoch. Ordinary episode boundaries retain history.
- Walkability, localization freshness and calibration checks belong to the later
  state adapter. This skeleton only checks integer grid bounds.
- Full 2400-step ROS replay, typed ROS messages and real execution feedback remain
  future work. A new message-only package is not created until those definitions
  are implemented.
- No service/autostart configuration is installed. The separate policy workspace
  and pre-existing `husky-sentry-bot` project are not modified by this launch.
