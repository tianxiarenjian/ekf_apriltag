# 这个项目是为了实现“通过Apriltag和EKF实现空地机器人协同定位
# This project is to enable "the collaborative positioning of air-ground robots through Apriltag and EKF.

# 1.运行calib//main.py获取相机内参矩阵和相机畸变
# 2.更改show_apriltag.py中的相机内参矩阵和相机畸变
# 3.运行show_apriltag.py，实现识别图片中的apriltag并计算相机与apriltag间的转移矩阵以及欧拉角
# 待开发：加入EKF，提高精确度

# 1.Run calib//main.py to get the camera parameter matrix and camera distortion
# 2.Changes the camera internal parameter matrix and camera distortion in show_apriltag.py
# 3.Run show_apriltag.py to recognize apriltag in the picture and calculate the transition matrix and Euler Angle between camera and apriltag
# To be developed: Add EKF to improve accuracy
