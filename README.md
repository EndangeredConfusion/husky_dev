robot: a200-balto


(mdns is set up such that)


`ssh <username>@a200-balto.local`
works while on the OARBOT networks.


the UWB configured
`lrwxrwxrwx 1 root root 13 Apr  2 18:32 usb-SEGGER_J-Link_000760168714-if00 -> ../../ttyACM0`

map of lab:
<img width="509" height="509" alt="image" src="https://github.com/user-attachments/assets/5dfee510-45ea-4d4c-993d-df247aa27079" />

Anchor positions can be configured via:
`~/config/uwb_locations.yaml`
The anchor ID is whatever the anchor thinks it is. If an anchor is publishing that is not in this map wherever the anchor thinks it is (what it's firmware was configured to) will be used instead. This may be very wrong as many of the anchors are set to some pretty random locations.


publish twist:
`ros2 topic pub /a200_1201/cmd_vel geometry_msgs/msg/Twist "{linear: {x: .1, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" -r 20`

start ekf:
`ros2 run husky_localization ekf`

start uwb reader:
`ros2 run husky_localization uwb_pub --ros-args -p uwb_port:=/dev/serial/by-id/usb-SEGGER_J-Link_000760168714-if00`


## RL Pipeline

Pipeline:
```
EKF -> Discretizer -> RL Policy -> A* -> Controller -> Motors
                          ^
                    Lambda Node
```

All RL-side nodes live in `src/husky_rl/`. Build order:
```
colcon build --packages-select husky_interfaces
colcon build --packages-select husky_rl husky_control
```

**Launch (fixed lambda):**
```
ros2 launch husky_rl rl.launch.py adaptive:=false fixed_lambda:="[1,2,2,9]"
```

**Launch (adaptive lambda / Algo 2):**
```
ros2 launch husky_rl rl.launch.py adaptive:=true
```

This starts all four nodes: `lambda_node`, `rl_policy`, `astar_node`, `point_controller`.

### Topics

| Topic                | Type                             | Publisher        | Subscriber                         |
| -------------------- | -------------------------------- | ---------------- | ---------------------------------- |
| `/agent_grid_cell`   | `husky_interfaces/GridCell`      | Discretizer      | rl_policy, lambda_node, astar_node |
| `/lambda_values`     | `std_msgs/Float32MultiArray`     | lambda_node      | rl_policy                          |
| `/rl_goal`           | `husky_interfaces/GridCell`      | rl_policy        | astar_node                         |
| `/rl_goal_index`     | `std_msgs/Int32`                 | rl_policy        | debug                              |
| `/rl_path`           | `std_msgs/Int32MultiArray`       | astar_node       | point_controller                   |
| `/a200_1201/cmd_vel` | `geometry_msgs/Twist`            | point_controller | motors                             |

### TODO
- **cell_size_m**: measure physical size of one grid cell in the lab (metres), then set in `src/husky_rl/launch/rl.launch.py` under `astar_node` parameters. Until then A* publishes raw grid integers and Stanley control will not be scaled correctly.
- **Discretizer node**: must publish `husky_interfaces/GridCell` to `/agent_grid_cell`.
- Tune EKF
- Review and finalize `point_controller.py` integration
