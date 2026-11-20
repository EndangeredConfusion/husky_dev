# Husky + Jetson policy integration

ROS 2 Jazzy workspace for GPS localization, DQN/Transformer goal selection and path
planning. The Transformer runs on Jetson GPU using BF16.

**Robot controllers remain disabled by default (`enable_controller=false`).**
The validated execution loop runs only in an isolated Gazebo environment.

## Current status

- GPS → valid grid observation → Transformer → candidate goal → A* metric path:
  bench-tested. Select `policy:=dqn` or `policy:=transformer` in the RL launch files.
- Gazebo execution → macro feedback → Transformer history: validated for six
  consecutive decisions (30 primitive steps), including nonzero rewards.
- Duplicate/stale feedback, lost commit acknowledgments, localization dropout,
  worker loss and coordinator loss have dedicated checks.
- Full Husky simulation, physical map calibration and hardware execution remain
  follow-up work. MPPI is TODO. Three existing map/tracker test failures are recorded
  in the candidate bench results.

## Run and validate

| Task | Instructions and results |
|---|---|
| GPS reception and NavFn/A* bench | [GPS/planning guide](bench/gps_navfn/README.md) |
| Transformer candidates, controller off | [Candidate guide](bench/transformer_candidates/README.md) · [Results](bench/transformer_candidates/RESULTS.md) |
| Gazebo execution and history updates | [Simulation guide](bench/transformer_execution/README.md) · [Results](bench/transformer_execution/RESULTS.md) |
| Model setup and recorded replay | [Deployment](docs/transformer/README.md) · [Input contract](docs/transformer/CONTRACT.txt) |

Weights and the GPU Python environment live in `~/transformer_deployment`, outside
Git. Follow the guides for build order, environment setup and isolated ROS domains.
The simulation runner uses Gazebo's differential-drive example, not the full Husky
model; its results do not establish collision avoidance or field readiness.

## Gazebo handoff

Execution topics are under `/transformer_sim/`: `odom`, `cmd_vel`, `state`, `command`,
`feedback`, `heartbeat` and `audit`. Adapt the full Husky model to these interfaces,
verify the field86 coordinate mapping, and retain the execution/history checks.
One policy decision holds a goal for up to **five primitive steps**, including turns;
reaching the end of a path is not automatically a completed macro action.

The previous UWB/DQN setup notes are preserved in [historical setup](docs/legacy_setup.md).
