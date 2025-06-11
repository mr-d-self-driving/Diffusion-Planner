# diffusion planner

```bash
cd src
pip install lanelet2
git clone https://github.com/autowarefoundation/autoware_cmake
git clone https://github.com/autowarefoundation/autoware_msgs
git clone https://github.com/autowarefoundation/autoware_lanelet2_extension
rosdep update
rosdep install -y --from-paths src --ignore-src --rosdistro $ROS_DISTRO
```
