# Jetson GNSS and candidate-to-path bench

The GPS check uses the robot ROS domain (0). The planning check runs on the Jetson in localhost-only domain 87 with a synthetic map and starts just the GPU worker, a bench transform and the planner server. No controller or motion publisher is involved.

## GPS on Jetson, reception on Husky

Use the existing robot network and ROS discovery environment on both hosts. The USB u-blox receiver is connected to the Jetson. Install the Jazzy serial driver if missing:

```bash
sudo apt-get install ros-jazzy-nmea-navsat-driver
port=/dev/serial/by-id/usb-u-blox_AG_-_www.u-blox.com_u-blox_GNSS_receiver-if00
# If the current user lacks access, grant access to this device for this session.
sudo setfacl -m "u:$(id -un):rw" "$(readlink -e "$port")"
bash ~/husky_ws/bench/gps_navfn/start_gps.sh
```

If the bench driver is already publishing, go straight to the receiver check. The script runs in the foreground at 115200 baud; Ctrl-C stops it. The device ACL may need restoring after USB reconnect or reboot. The script uses the system Python packages because the Jetson user-installed NumPy is incompatible with one driver dependency. The receiver also emits binary data; the tested driver logs errors only to avoid repeated mixed-stream warnings.

On **Husky**, source the ROS environment and run:

```bash
source /opt/ros/jazzy/setup.bash
ROS_DOMAIN_ID=0 timeout 15 ros2 topic echo /gps/fix sensor_msgs/msg/NavSatFix --once
python3 ~/husky_ws/bench/gps_navfn/check_gps.py
```

If Husky has no checkout, copy just the check from the Jetson, then run it on Husky:

```bash
scp jetson@192.168.131.5:husky_ws/bench/gps_navfn/check_gps.py /tmp/check_husky_gps.py
python3 /tmp/check_husky_gps.py
```

The check waits up to 15 seconds for two distinct stamps in frame `gps_link`. A no-fix sample is expected on the bench and carries status -1 and NaN coordinates. This checks cross-host reception; field accuracy and heading are separate tests.

## Candidate goal to NavFn and training A*

Use the built Transformer policy overlay, model and GPU venv from the [workspace deployment guide](../../docs/transformer/README.md). Jetson needs Jazzy nav2_planner, nav2_navfn_planner, nav2_msgs and tf2_ros. Keep the archived training dependency outside the repository:

```bash
cd ~/husky_ws
mkdir -p ~/transformer_deployment/checks/goal_navfn
python3 -m pip download --no-deps --only-binary=:all: --dest ~/transformer_deployment/checks/goal_navfn pathfinding==1.0.20
bash bench/gps_navfn/run_planner.sh
```

The runner checks that domain 87 is free, configures and activates GridBased/NavFn, then calls the ROS ComputePathToPose action with an explicit start and GPU-selected goal. Each case waits for the requested occupancy map to reach the planner costmap. A* separately uses the training version of the four-neighbor planner and the workspace walkable cells. It publishes the centers as a nav_msgs/Path on `/transformer/shadow/training_path` during the run. The synthetic grid uses 0.5 m per model cell, origin (0,0) and positive y; the costmap resolution is 0.1 m. These values are test fixtures independent of the source map scale.

Each run prints a fresh directory under `~/transformer_deployment/checks/goal_navfn/` for result.json and process logs. The runner stops its child processes on exit. Optional overrides are HUSKY_WS, TRANSFORMER_DEPLOYMENT and TRAINING_WHEEL (use absolute paths).

## Observed results, 2026-11-03

Husky cpr-a200-0679 received Jetson's /gps/fix before and after restarting the driver: frame gps_link, status -1, service 1, latitude/longitude/altitude NaN. No physical fix was required.

GPU BF16 inference from cell (4,3) with lambda [7,2,2,2] selected (1,1). The original [machine-readable result](result.json) includes the checkpoint digest and A* grid and metric coordinates. The A* route contained six cells and the model history remained at version zero.

| Synthetic NavFn case | Action outcome | Path samples |
| --- | --- | ---: |
| Open | Succeeded | 35 |
| Barrier with gap | Succeeded, detoured via gap | 71 |
| Full barrier | Aborted, NO_VALID_PATH (208) | 0 |

These results cover path generation. Site map registration, path tracking, physical execution and macro-action feedback remain separate integration steps.
