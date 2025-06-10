
#!/usr/bin/env python3

# Import the subprocess and time modules
import subprocess
import time

# List of commands to run
commands = [
    # Run the Micro XRCE-DDS Agent
    "MicroXRCEAgent udp4 -p 8888",

    # Run the PX4 SITL simulation
    "cd ~/PX4-Autopilot && make px4_sitl gz_osprey_walls",

    # Run the bridge for sensors
    "source ~/osprey_ws/install/setup.bash && ros2 run ros_gz_bridge parameter_bridge --ros-args -p config_file:=/home/susan/PX4-Autopilot/bridge.yaml",

    #Run the transform with RVIZ2
    "source ~/ws_sensor_combined/install/setup.bash && ros2 launch drone_description transform_launch.py",

    # Run QGroundControl
    "cd ~/gazebo_px4_sitl_backup && ./QGroundControl-x86_64.AppImage "
]

# Loop through each command in the list
for command in commands:
    # Each command is run in a new tab of the gnome-terminal
    subprocess.run(["gnome-terminal", "--tab", "--", "bash", "-c", command + "; exec bash"])
    
    # Pause between each command
    time.sleep(1)