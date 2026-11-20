# Execution/history simulation validation — 2026-11-20

Baseline: `245f314` plus the stage-6/7 working-tree changes. Tested on Jetson through
Husky SSH. Real hardware control stayed disabled. ROS test domains were nonzero,
localhost-only; the final runner assigns a fresh Gazebo partition per run.

## Completed checks

| Check | Observed result |
|---|---|
| Real GPU + Gazebo execution | Six consecutive completed macros, 30 confirmed primitive steps |
| Worker history | Size/version progressed from 0 through 6; subsequent decisions used the updated version |
| Macro rewards | `0`, `0`, `6.590361045807`, `6.524457435349`, `6.459212860995`, `6.394620732385` |
| First macro | Three right turns and two forward moves: `(4,3)` to `(4,1)` |
| Lost commit acknowledgment | Real worker committed once; exact retry returned history version 1 without increment |
| Duplicate/conflicting feedback | Identical duplicate acknowledged once; conflicting feedback rejected |
| Task change | New epoch/history reset; previous task feedback rejected |
| Pose dropout during execution | Fault-associated stop in approximately 1.01 s; no later nonzero command; zero commits, worker paused |
| Worker exit during execution | Fault-associated stop in approximately 0.48 s; zero commits |
| Coordinator killed during execution | Executor heartbeat lease expired in approximately 2.01 s; no later nonzero command; history remained zero |
| Controller freshness and existing follower tests | 16 passed; callback tests instantiate no ROS node or publisher |
| Execution-session/macro/task-control tests | 11 passed |
| Fresh-episode recovery after fault suite | New Gazebo partition and worker epoch; six macros committed, history size/version 6 |

The heartbeat test leaves an idle worker task with zero committed history; the
executor is paused. The coordinator cannot abort the worker after being killed.
Recovery therefore creates a fresh task/episode rather than resuming its old ticket.

## Issues found and resolved during validation

- The shutdown path now keeps ROS alive long enough to send stop, and inhibits timer
  callbacks from starting another action during shutdown.
- Early heartbeat/recovery runs paused because the bundled Gazebo example published
  odometry at only 1 Hz, racing the one-second watchdog. The world now publishes at
  50 Hz; the watchdog is unchanged. Those failed runs were not counted as passes.
  Final runs also use distinct Gazebo partitions and reject failures preceding injection.
- Stop-latency assertions associate zero commands with the fault, rather than
  accidentally counting an ordinary pause between primitives.
- System SciPy 1.11.4 could not import with user-site NumPy 2.5.4. Tests pass with
  `PYTHONNOUSERSITE=1`, using system NumPy 1.26.4. No global package replacement or
  GPU environment modification was required.

## Evidence locations on Jetson

- Initial six-macro run: `/tmp/transformer_sim_success.log`
- Final 50 Hz recovery/nominal run: `/tmp/transformer_nominal_50hz/`
- Real GPU transaction checks: `/tmp/transformer_sim_transactions.log`
- Pose fault: `/tmp/transformer_pose_dropout_50hz/`
- Worker fault: `/tmp/transformer_worker_loss_50hz/`
- Heartbeat fault: `/tmp/transformer_heartbeat_loss_50hz/`
- Controller tests: `/tmp/transformer_controller_tests.log`
- Session/macro/control tests: `/tmp/transformer_session_tests.log`

Each runner output directory contains the generated world, process logs, timestamped
ROS probe records and a result JSON. These temporary paths can be regenerated with
the instructions in README.md. Do not treat the earlier failed heartbeat `_02` run
or the preliminary stop-latency measurement as successful final evidence.

## Limits

This is an execution/interface simulation using Gazebo's differential-drive example,
field-table metric centers and virtual training-grid walls. It is not a collision
test or a full Clearpath Husky dynamics validation. Forward motions include heading
alignment along irregular map edges, while logical training actions retain their
discrete turn/forward accounting. Hardware primitive semantics and field calibration
still require separate validation before enabling physical motion. MPPI remains TODO.

No real-robot controller was launched. The existing launch defaults remain
`enable_controller=false`. Validation was performed before committing these changes to main.
