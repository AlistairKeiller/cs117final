FROM autodriveecosystem/autodrive_roboracer_api:2026-icra-practice

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    ros-$ROS_DISTRO-foxglove-bridge \
    ros-$ROS_DISTRO-ackermann-msgs

RUN pip3 install gtsam warp-lang scipy "numpy<2"

RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> ~/.bashrc
