# Jetson candidate integration validation — 2026-11-20

Host: `jetson@192.168.131.5`, reached through Husky. Source baseline `2c3ade3`.
Validation was performed on the integration changes before committing them to main.
Runtime: ROS Jazzy, PyTorch `2.10.0+cu130`, CUDA available, BF16 supported.
Checkpoint SHA-256: `012765a00264a5b64cc14ef2cd3227a2d2f4d8d03463de4f79a1969c87a53f50`.

## Verified

- Independent `install_candidates` build: all four selected packages succeeded.
- Launch selection: DQN selects discretizer/lambda/DQN/A*; Transformer selects
  discretizer/lambda/Transformer/A*. Neither selection starts a controller when disabled.
- Isolated ROS domain 73, localhost only: synthetic GPS reached the real GPU worker
  and generated a candidate and metric path. No DQN node, controller node or cmd_vel
  topic appeared in the test graph.
- Nontrivial route: `(4,3) → (4,2) → (4,1) → (3,1) → (2,1) → (1,1)`.
  Six cell centers; the path topic matched the planner result and cell-table conversion.
  First metric point approximately `(19.0115, -11.8444)` m; last `(0, 0)` m.
- Repeated stationary fixes do not create new candidate decisions. A new decision ID
  with the same goal replans; a duplicate ID does not.
- Invalid GPS, GPS dropout, stale timestamps and off-walkway coordinates invalidate the state.
- Worker status after candidate testing: history size/version both zero, BF16,
  context capacity 2400. Unexecuted candidates do not create synthetic rewards.
- Existing GPU model contract test passed: direct BF16 logits match, input packing
  matches, repeated inference is pure, actual-action commits are idempotent, and
  stale/invalid inputs are rejected. This separate test exercises history in memory;
  the candidate pipeline itself never commits.

## Existing baseline failures

The cell-table/tracker suite returns **38 passed, 3 failed**, both in the modified
tree and a separate extraction of unchanged `2c3ade3`:

- `test_latlon_to_local_matches_table`
- `test_edited_latlon_without_east_north_is_reported`
- `test_closest_non_adjacent_pair_no_flicker`

These are not introduced by this change. GPS table consistency and tracking geometry
still need review with the map/simulation owners. Logs report stored metric versus
lat/lon discrepancies up to approximately 0.26 m; the current mapper uses lat/lon.
The bench does not establish physical map accuracy or certify field operation.

## Follow-up scope

Stages 6–7 remain: execution acknowledgment, macro-step/reward semantics, commit
feedback, cancellation and controller stale-pose handling. MPPI remains TODO.
There was no field test and no robot motion command. Candidate generation uses no
execution history; this is not a completed recurrent decision/execution loop.

Jetson logs: `/tmp/candidate_check.log`, `/tmp/candidate_launch.log`,
`/tmp/candidate_model.log`, `/tmp/candidate_unit.log`, `/tmp/candidate_baseline_unit.log`.
These are temporary; the findings above are retained in this repository.
