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
