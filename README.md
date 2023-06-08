# moveit_neural_collision
## Build 
```
mkdir -p ros_ws/src
cd ros_ws/src
git clone https://github.com/pac48/moveit-neural-collision.git  
vcs import < moveit-neural-collision/external.repos.yaml
cd ..
rosdep install --from-paths src --ignore-src -y
colcon build --allow-overriding moveit_configs_utils common_interfaces diagnostic_msgs geometry_msgs moveit_common moveit_msgs moveit_resources_panda_moveit_config nav_msgs sensor_msgs sensor_msgs_py shape_msgs std_msgs std_srvs trajectory_msgs visualization_msgs --symlink-install --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release

```

## Run
Start by launch MoveIt with a robot configuration setup. 

`ros2 launch sawyer_moveit_config moveit.launch.py`

Create training data for mesh object.

`ros2 run  collision dist`

Finally, train the neural network and publish to ROS as a point cloud.

`ros2 run collision main`


<p align="center">
  <img src="demo.gif" width="500" title="demo">
</p>