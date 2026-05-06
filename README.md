# slam
```bash
python3 autodrive.py
```
# Foxglove
```bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
```
# Autodrive
```bash
ros2 launch autodrive_roboracer bringup_headless.launch.py
```
# Disparity extender:
```bash
cd disparity_extender && colcon build && source install/setup.bash
ros2 launch disparity_extender disparity_extender.launch.py
```
