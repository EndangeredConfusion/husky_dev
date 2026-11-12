# GPS cell-table pipeline: hardware checklist

Checks for bringing the '86 Field GPS pipeline (branch `gps-cell-table`) onto
the real Husky. **Part 1** can be done indoors (no satellite fix needed).
**Part 2** needs open sky on the field.

Setup assumed (from [bench/gps_navfn](../gps_navfn/README.md)):

- u-blox receiver on USB to the **Jetson** (`192.168.131.5`), driver
  `nmea_navsat_driver`, started by `bench/gps_navfn/start_gps.sh`.
- Output `/gps/fix` (`sensor_msgs/NavSatFix`, frame `gps_link`) and, if the
  receiver sends NMEA HDT, `/heading` (`geometry_msgs/QuaternionStamped`).
- Robot network on **ROS domain 0**; Husky host `cpr-a200-0679`.

Every terminal below: `source /opt/ros/jazzy/setup.bash`, the workspace's
`install/setup.bash`, and `export ROS_DOMAIN_ID=0`.

Write every measured value in [Results](#results) at the end.

---

## Part 1: Indoor checks (no fix needed)

### I1. Software on the robot computers
- [ ] Branch `gps-cell-table` is checked out and built on the computer that
      runs the RL pipeline (and on the Jetson if nodes run there):
      `colcon build --symlink-install --packages-select husky_interfaces husky_control husky_localization husky_rl`
      (if colcon fails with `option --uninstall not recognized`, prefix with
      `PYTHONNOUSERSITE=1`).
- [ ] Unit tests pass:
      `cd src/husky_rl && PYTHONPATH=$PWD python3 -m pytest -q test` (52 passed)
      `cd src/husky_localization && PYTHONPATH=$PWD python3 -m pytest -q test/test_gps_pose_math.py` (14 passed)
- [ ] `stable_baselines3` and `pathfinding` import on that computer
      (`python3 -c "import stable_baselines3, pathfinding"`).

### I2. Identify the receiver
- [ ] The USB device is the **ZED-X20D** (not another u-blox board). Check in
      u-center 2 (or the board's label); note the firmware version.
      Heading needs the **HDG** firmware (HDG 2.00 or later).
- [ ] Both antennas (ANN-MB2) are connected: ANT1 = position antenna,
      ANT2 = heading (secondary) antenna. Note which physical antenna is which.

### I3. Receiver output configuration (u-center 2)
`nmea_navsat_driver` reads **only NMEA**: GGA (position, fix quality) and
**HDT** (heading). It ignores UBX-NAV-DAHEADING.
- [ ] NMEA **GGA** enabled on the USB port.
- [ ] NMEA **HDT** enabled on the USB port (otherwise `/heading` never appears).
- [ ] Output rate 5–10 Hz.
- [ ] Configuration saved to the receiver's flash (survives power-off).

### I4. Driver and topics
- [ ] On the Jetson: `bash ~/husky_ws/bench/gps_navfn/start_gps.sh` runs
      without "Missing access" / "Serial port already in use".
- [ ] On the pipeline computer: `ros2 topic hz /gps/fix` shows the configured
      rate (5–10 Hz).
- [ ] `ros2 topic echo /gps/fix --once`: frame `gps_link`; indoors expect
      status `-1` and NaN coordinates. That is fine.
- [ ] `ros2 topic list | grep heading` shows `/heading`. (It may only publish
      once there is a heading solution, i.e. outdoors.)

### I5. Robot namespace and velocity topic
The launch default is `/a200_1201/cmd_vel`; this robot is probably `a200_0679`.
- [ ] `ros2 topic list | grep cmd_vel` → note the exact topic:
      `cmd_vel_topic = ____________`
- [ ] `ros2 topic info <that topic> -v` shows a `geometry_msgs/msg/TwistStamped`
      subscriber (otherwise the controller needs `use_stamped:=false`).

### I6. Measure the antenna mounting
Robot frame: origin = robot centre (`base_link`), **x forward, y left**.
- [ ] ANT1 position: `antenna_offset_x_m = ______`, `antenna_offset_y_m = ______`
- [ ] Baseline ANT1 → ANT2: length `______ m`, direction relative to the
      robot's forward axis `______ deg` (0 = ANT2 straight ahead of ANT1,
      positive = to the left). This is the starting guess for
      `heading_offset_rad`; Part 2 confirms it.
- [ ] Antennas are rigidly mounted, with a clear view of the sky, and the
      baseline/ground plane follow the u-blox ZED-X20D integration manual.

### I7. Pipeline dry run (motors disabled, e-stop pressed)
- [ ] `ros2 launch husky_rl field_check.launch.py csv_path:=$HOME/fc_indoor.csv`
      prints `REJECTED: no fix` once a second and the CSV file is created.
- [ ] `ros2 launch husky_rl rl.launch.py use_gps_pose:=true cmd_vel_topic:=<I5 topic>`
      starts every node with no traceback; expect `GPS fix rejected: no fix`
      (discretizer) and `fix rejected: no fix` (gps_pose); no `/rl_path`, no motion.
- [ ] Stop with **one** Ctrl-C; nothing keeps running afterwards
      (`ros2 node list` is empty for these nodes).

### I8. RTK corrections (prepare)
- [ ] Decide the correction source (NTRIP caster / PointPerfect / own base).
- [ ] Credentials and the way corrections reach the receiver are ready.
      Without RTK expect ~1–2 m error, too much for the 1.5 m "reached" radius.

### I9. Field kit
- [ ] Charged robot battery, laptop, e-stop/remote, tape measure, phone
      compass, this checklist and the anchor table below.

---

## Part 2: Outdoor checks ('86 Field, open sky)

### Anchor points (from `config/field86_cells.yaml`)

| Anchor | Cell | lat | lon | east_m | north_m |
|---|---|---|---|---|---|
| Star_1 | (1,1) | 42.730705 | -73.680592 | 0.00 | 0.00 |
| A | (4,1) | 42.730682 | -73.680334 | 21.10 | -2.56 |
| B | (12,1) | 42.730575 | -73.679384 | 98.78 | -14.47 |
| Star_2 | (15,1) | 42.730527 | -73.679088 | 122.98 | -19.81 |
| J2 | (4,3) | 42.7305366 | -73.6804124 | 14.68 | -18.74 |
| J1 | (1,5) | 42.7302812 | -73.6806933 | -8.28 | -47.17 |
| J3 | (12,6) | 42.7300832 | -73.6794345 | 94.65 | -69.21 |
| Star_3 | (1,9) | 42.730140 | -73.680727 | -11.04 | -62.90 |
| E | (7,9) | 42.729967 | -73.680030 | 45.95 | -82.15 |
| F | (10,9) | 42.729919 | -73.679610 | 80.30 | -87.50 |
| Star_4 | (15,9) | 42.729932 | -73.679259 | 109.00 | -86.05 |

Start `field_check` once at the beginning and leave it running for O1–O5
(one CSV per session; use a new `csv_path` for each session):

```bash
ros2 launch husky_rl field_check.launch.py csv_path:=$HOME/fc_field_1.csv
```

### O1. Fix quality (robot still, open sky)
- [ ] `fix=` changes from `NO_FIX` to `FIX`; time to first fix `______ s`.
- [ ] With RTK: `fix=GBAS/RTK` and `std` ≈ 0.02–0.05 m (RTK **fixed**).
      `std` of several metres while showing GBAS/RTK = RTK **float**
      (the driver reports fixed and float with the same status; only the
      covariance differs).
- [ ] Without RTK: note the `std`: `______ m`.

### O2. Static test at Star_1 (2 minutes)
- [ ] Park the robot centre on the Star_1 marker.
- [ ] `field_check` shows `nearest=1,1` and the distance to its centre:
      `______ m` (offset of the table/GPS at Star_1).
- [ ] Keep still 2 minutes. From the CSV (or ask Claude to analyse it):
      spread of `east_m`/`north_m` = `______ m`.
      Pass: spread well below `reached_radius_m` (1.5 m); with RTK fixed
      expect < 0.1 m.

### O3. Heading
`field_check` prints `heading_raw` = the receiver's compass heading
(0 = north, clockwise; values past 180° show as negative, e.g. 190° → −170°).
- [ ] Point the robot along the **top walk** from Star_1 toward A.
      Expected walk direction: compass **96.9°**. Reading: `______`
- [ ] Point the robot along the **left walk** from Star_1 toward J1.
      Expected: compass **190.0°** (shows ≈ **−170°**). Reading: `______`
- [ ] Turn slowly through 360°: heading changes smoothly, no jumps or freezes.
- [ ] `heading_offset_rad` = (reading − walk direction), in radians.
      Example: reading 101.9° on the top walk (5° clockwise of 96.9°) →
      `heading_offset_rad ≈ +0.087`. Average the two walks. Value: `______`
      (Should agree with the baseline angle measured in I6.)

### O4. Pose for the controller (`gps_pose`)
Run `gps_pose` with the values from I6 and O3:
```bash
ros2 run husky_localization gps_pose --ros-args \
  -p heading_offset_rad:=<O3> -p antenna_offset_x_m:=<I6> -p antenna_offset_y_m:=<I6>
ros2 topic echo /gps/local_pose
```
- [ ] At Star_1: `position_m` ≈ (0, 0) within the O2 offset.
- [ ] Facing along the top walk: `theta_rad` ≈ **−0.12** (−6.9°);
      along the left walk: ≈ **−1.75** (−100°).
- [ ] Spin in place slowly: `position_m` stays within ~0.2 m. If it draws a
      circle, the antenna offset (I6) is wrong.

### O5. Anchor survey (motors off or joystick only)
- [ ] Drive/push to each anchor in the table; stop ~30 s with the robot centre
      on the real fork/marker. Note the time and anchor name for each stop.
- [ ] Afterwards, give Claude the CSV and the stop list: it compares the
      measured anchor positions with the table, and you update the `lat`/`lon`
      in `field86_cells.yaml` (only `lat`/`lon` are needed; cell centres are
      computed from them), then re-run the unit tests.

### O6. Walk every walk once (motors off or joystick only)
- [ ] Drive along all walks (top, left, bottom, middle branch, right side,
      lower stair). In `field_check`:
  - [ ] never `OFF WALKWAY` while on a walk;
  - [ ] `reached` steps through the expected cells;
  - [ ] note any place where a cell is never reached: `______`

### O7. Controller only (first powered test; e-stop in hand)
Two terminals, `gps_pose` (O4) running:
```bash
ros2 run husky_control point_controller --ros-args \
  -p localization_topic:=/gps/local_pose -p cmd_vel_topic:=<I5 topic> \
  -p points_topic:=/test_path -p nominal_velocity:=0.2
# robot at Star_1, facing roughly east; drive to anchor A (21.1, -2.56):
ros2 topic pub --once /test_path std_msgs/msg/Float64MultiArray "{data: [21.1, -2.56]}"
```
- [ ] The robot turns toward A and drives along the top walk.
- [ ] It stops near A; stop error `______ m` (goal tolerance 0.15 m).
- [ ] If it heads off at an angle: stop, re-check O3/O4 (heading offset).

### O8. Full pipeline, short run (e-stop in hand)
```bash
cd ~ && ros2 bag record -o field_robot_1 --topics /gps/fix /heading /gps/local_pose /agent_grid_cell /rl_goal /rl_path <I5 topic>
ros2 launch husky_rl rl.launch.py use_gps_pose:=true cmd_vel_topic:=<I5 topic> fixed_lambda:="[1,1,10,1]" \
  heading_offset_rad:=<O3> antenna_offset_x_m:=<I6> antenna_offset_y_m:=<I6>
```
(Leave out any offset that is 0; the default is 0.)
- [ ] Start at Star_1: log shows `Reached cell (1, 1)`, then a goal and an
      A* path.
- [ ] Robot follows the walk; cells are reached in order; no `OFF the walkways`.
- [ ] Note skipped-cell warnings: `______`
- [ ] Stop with one Ctrl-C; give Claude the bag for analysis.

---

## Results

| Item | Value |
|---|---|
| Receiver model / firmware (I2) | |
| NMEA GGA + HDT enabled, rate (I3) | |
| `/gps/fix` rate (I4) | |
| `cmd_vel_topic`, stamped? (I5) | |
| `antenna_offset_x_m`, `antenna_offset_y_m` (I6) | |
| Baseline length / angle (I6) | |
| RTK source (I8) | |
| Time to first fix (O1) | |
| Fix type and `std` (O1) | |
| Star_1 offset (O2) | |
| Static spread over 2 min (O2) | |
| Heading readings, top / left walk (O3) | |
| `heading_offset_rad` (O3) | |
| Spin-in-place drift (O4) | |
| Anchor survey CSV file (O5) | |
| Cells never reached (O6) | |
| Controller stop error at A (O7) | |
| Full-run bag (O8) | |
