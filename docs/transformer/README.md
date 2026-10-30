# Transformer interface: history transactions and isolated replay

This adds a versioned ROS-to-GPU communication skeleton, without changing the
existing DQN, localization or motion packages. It is not a complete robot policy
deployment: live state adaptation, calibration and physical feedback remain later stages.
Recorded ROS/GPU replay and deterministic history checks are available below.

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

- The bridge wakes its ROS executor when GPU/IPC work completes; no 100 ms polling
  delay is added to each response. The replay source controls the 10 Hz pace.
  The transport failure timeout is separately set to one second.
- History is in memory and capped at 2399 completed records plus the current
  observation. Restart requires an explicit new task.
- Identical commit retries within a 256-ticket ledger return the original acknowledgment
  without mutation; conflicting feedback is rejected. At most 32 prediction tickets
  are retained. Older acknowledgments are not current status snapshots.
- Task lambda is fixed per epoch. Ordinary episode boundaries retain history.
- Walkability, localization freshness and calibration checks belong to the later
  state adapter. This skeleton only checks integer grid bounds.
- Typed ROS messages and real execution feedback remain future work.
  A new message-only package is not created until those definitions
  are implemented.
- No service/autostart configuration is installed. The separate policy workspace
  and pre-existing `husky-sentry-bot` project are not modified by this launch.

## Stage 3: deterministic history verification

`history_manager.py` separates task epochs, prediction snapshots, feedback validation
and atomic history commits from GPU inference. `cancel` discards a candidate that
was never executed. `abort` marks uncertain execution and blocks further inference
until a deliberate task reset. Neither invents a transition.

`transition_assembler.py` validates a recorded trajectory and creates observation
and feedback requests from its actual action and macro reward. It does not define
the missing real-robot macro-step or reward semantics.

With the workspace overlay sourced, run outside the repository:

```bash
~/transformer_deployment/venv/bin/python \
  ~/husky_ws/src/husky_transformer_policy/test/check_history.py \
  --model-dir ~/transformer_deployment/model \
  --recording ~/transformer_deployment/benchmarks/history_sample.npz
```

This compares all committed observation/action/feedback arrays over 6000 records
against the original `ad_online.py`, including ordinary episode boundaries and the
2400-to-2401 rolling window transition. Selected batch tensors and BF16 logits are
also checked exactly. Duplicates, conflicting feedback, sequence mismatch,
cancellation, abort, reset and restart are checked without moving a robot.

## Stage 4: 10 Hz isolated recorded replay

Check domain 42 is unused, then launch into a **new** result directory:

```bash
ros2 launch husky_transformer_bridge replay.launch.py \
  output:=/home/jetson/transformer_deployment/replay/run_6000 \
  count:=6000 hz:=10
```

The launch starts the GPU worker, ROS bridge and system-Python replay client.
Startup/warmup are excluded from measured pacing. The default 6000 records run
for at least 600 seconds at 10 Hz, reaching full context and then rolling it.
The original recording supplies actual actions, macro rewards and end flags;
predicted goals never drive a controller and are not substituted for recorded actions.
Unexpected nodes or execution topics in the isolated graph abort the run.
No live sensor subscription, navigation action client or velocity publisher exists.

Results include `protocol.json`, per-decision `frames.jsonl`, `resources.json`, and
`summary.json`. Timing spans ROS observation publication through the prediction
and feedback commit acknowledgment. Worker inference timing (history snapshot,
batch preparation and GPU output retrieval included) is recorded separately.
Scheduled deadline misses include launch jitter; transaction time excludes pacing
sleep. Late predictions are logged as discarded, while the corresponding actual
recorded transition is still committed to preserve the dataset's history.

Timing acceptance targets: transaction P95 <= 50 ms, maximum < 100 ms, zero
scheduled 100 ms deadline misses, no duplicate/unexpected responses or history
corruption. `TIMING_FAIL` is distinct from correctness failure and produces a
nonzero replay process exit. Inspect `summary.json`; the parent ROS launch's
exit status alone is not the acceptance result. Resource guards stop the replay
below 512 MiB available memory or above 85 C maximum sampled SoC temperature.
If any child exits, launch stops the remaining children. This is a stationary
software replay, not proof of physical navigation quality or complete sensor load.

## Validation on 2026-10-30

History verification passed for all 6000 recorded transitions and 20 episode ends.
All history arrays and ten selected BF16 output checks matched the original code
exactly. Model-contract regression and transport timeout/framing checks passed.

The Jetson BF16/domain-42 replay completed 6000 transactions in 600.04 seconds at
10 Hz. Correctness passed, but strict timing acceptance **failed**:

| Full ROS inference + feedback transaction | Measured |
| --- | ---: |
| P50 | 40.29 ms |
| P95 | 43.31 ms |
| P99 | 44.06 ms |
| Maximum | 168.40 ms |
| Scheduled 100 ms deadline misses | 2 / 6000 |

At zero-based index 3646, worker inference took 156.42 ms; that prediction was
marked discarded. The next transaction took 35.19 ms but inherited scheduling
delay and finished 104.16 ms after its scheduled start. There was no continuing
backlog and no unexpected response. The root cause of the isolated worker-time
spike is unresolved; this result does not establish a hard 10 Hz deadline guarantee.

The full 2400-observation context had 3601 samples and transaction P95 43.61 ms.
Contexts 1, 30 and 600 are single observations in this increasing-history replay,
not independent latency distributions. Peak worker RSS was 1.21 GiB; peak PyTorch
CUDA allocation/reservation was 51.93/70.00 MiB (not total device memory).
Minimum system available memory was 5.42 GiB and maximum sampled temperature
57.812 C. The read-only post-run power-mode query reported 25 W.

Results are retained outside Git at
`~/transformer_deployment/replay/stage4_full_20261030/`; checks and launch logs are
under `~/transformer_deployment/checks/stage34/`. The result manifest records tested
source hashes. Both child processes exited cleanly and removed the worker socket.
Only the two replay nodes and shadow request/response topics were present, plus
standard ROS logging/parameter topics. No robot commands were issued.

Next timing work should trace CPU scheduling and GPU clocks/contention around
spikes, then repeat the same acceptance run; do not weaken the recorded threshold
or discard the failed run. Live calibration, input adaptation and physical feedback
remain separate unfinished stages.
