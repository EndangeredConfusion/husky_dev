# Transformer execution/history simulation (stages 6–7)

This suite runs the real Jetson BF16 policy worker with a Gazebo differential-drive
model, observed odometry, primitive execution feedback and transactional history
commits. It never launches the Husky point controller or publishes to robot topics.
The generated world publishes odometry at 50 Hz; the bundled example's 1 Hz default
would race the unchanged one-second stale-pose watchdog.
The existing RL and simulation launch files still default `enable_controller=false`.

## Scope and geometry

The Gazebo vehicle is the installed Gazebo 8 differential-drive example, not the full
Clearpath Husky model. This tests interfaces and fault behavior, not Husky dynamics,
collision avoidance or field readiness. The world is flat and unobstructed.
Grid walls are enforced as virtual blocked-forward primitives. The executor maps
observed odometry to the field86 cell centers, assigning its initial pose to `(4,3)`.
This initial alignment is a test fixture, not GPS/map calibration.

Logical directions follow training (right/down/left/up). A right-turn primitive
physically turns clockwise by 90 degrees. A forward primitive aligns to the measured
metric edge and drives to its endpoint; such alignment is part of the forward motion,
not an additional logical training step. This distinction matters on the irregular
field map. Successful primitive completion requires observed pose, not elapsed time.

Training semantics come from the supplied archive's
`code_evaluation/thesis/envs/goal_selection_wrapper.py` and `astar_planner.py`:

- Hold one selected goal for five primitive steps, including right turns and blocked
  forward actions. Replan at macro start; preserve A* event count across macros.
- At every fifth episode primitive step, if positioned at a goal, primitive reward
  is that goal's raw lambda multiplied by `0.99 ** max(0, astar_events - 1)`.
- Macro reward is the sum of primitive rewards discounted by `0.99 ** offset`.
- The reference environment overrides termination to false; the configured episode
  limit is 1500 primitive steps. End flags are passed through to history.

The simulation executable uses the same macro/event logic but is not a replacement
for validating a continuous-motion deployment policy against the training simulator.

## Run on Jetson

Stop any earlier stage-6/7 test processes before starting a new run. Choose an unused
nonzero ROS domain. The Gazebo partition below must contain only this test world.

```bash
cd ~/husky_ws
source /opt/ros/jazzy/setup.bash
source install_candidates/setup.bash
export PYTHONPATH="$HOME/husky_ws/src/husky_transformer_bridge:$HOME/husky_ws/src/husky_transformer_policy:$HOME/husky_ws/src/husky_control:$HOME/transformer_deployment/ros_deps:$PYTHONPATH"
export ROS_DOMAIN_ID=74
export ROS_LOCALHOST_ONLY=1
export PYTHONNOUSERSITE=1
export GZ_PARTITION=husky_transformer_stage67

python3 bench/transformer_execution/run_simulation.py \
  --scenario nominal --output /tmp/transformer_nominal_new
```

The runner creates a fresh output directory, builds the test world from the locally
installed Gazebo example, assigns a unique Gazebo partition for that run, then starts
its own process groups, records ROS interfaces, checks results, then stops only the
processes it owns. Existing output directories
are rejected to preserve evidence. Use a new directory for each run.

Available fault scenarios: `pose_dropout`, `worker_loss`, `heartbeat_loss`.
They interrupt execution after a nonzero turn command, then verify failure feedback,
a fault-associated zero command, no resumed nonzero commands, and no history commit
for the interrupted macro. Pose and command bridges are separate so sensor loss
does not also remove the channel that sends the stop command.

## Interfaces

All new execution interfaces use `/transformer_sim/`:

| Topic | Purpose |
|---|---|
| `odom` | Gazebo wheel odometry, with monotonically advancing source timestamps |
| `cmd_vel` | Simulation-only Twist, bridged to `/model/vehicle_blue/cmd_vel` |
| `state` | Cell, logical direction, pose sequence, validity and busy state |
| `command` | initialize / execute / stop; execute includes epoch, decision ID, ticket and history version |
| `feedback` | initialized / accepted / primitive_completed / completed / failed / stopped |
| `heartbeat` | Coordinator lease; executor stops after two seconds without renewal |
| `audit` | Decisions, committed histories, pauses, reset and recovery records |
| `control` | Explicit operator stop / retry_commit / reset requests |

Worker status is checked every 0.5 seconds; RPC timeout is one second. Stale
simulation odometry stops execution after one second. Source timestamps must advance,
so repeated old odometry cannot renew validity. Primitive execution has a 120-second
timeout. Runtime checks reject ROS domain 0 and the wrong Gazebo partition.

## Commit and recovery contract

Only an acknowledged, completed macro with matching task epoch, decision ID,
observation sequence and action can update history. Failed, cancelled or incomplete
macros never synthesize rewards. Duplicate identical feedback returns the original
acknowledgment; conflicting duplicates and old task feedback are rejected.

If a commit reply is lost, the exact payload and ticket are retained and inference
pauses. `retry_commit` retries that transaction; the worker's commit ledger prevents
double counting. The recorded-feedback GPU check exercises this with a real commit
followed by an intentionally lost acknowledgment:

```bash
# Coordinator must be stopped before this separate recorded-feedback test.
python3 bench/transformer_execution/check_gpu_transactions.py \
  --socket /path/to/live/test/worker.sock --simulation-log /path/to/coordinator.log
```

Task changes require stop acknowledgment before reset. An interrupted primitive can
leave the vehicle between grid centers or midway through a turn. In that case the
executor refuses reinitialization: stop the test processes and start a fresh Gazebo
episode and task, rather than silently accepting the cached cell. The automated
runner performs this fresh initialization for each invocation. No automatic resume
is attempted after uncertain execution or worker loss. If the worker was lost during
an uncertain commit, preserve the audit log and restart the test episode/processes;
the old in-memory history cannot be reconstructed from a missing acknowledgment.

The separate point-controller change checks both EkfState source time and local
receipt age, rejects nonfinite pose/path input, sends zero and clears the active path
on pose loss. It requires a new path to resume. Its callback tests create no ROS node
or publisher. This does not enable that controller on the real robot.

MPPI, hardware motion, field calibration and full Husky/Gazebo model validation remain
outside this simulation interface task.
