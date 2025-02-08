# apriltag_pose_estimation.py
import cv2
import numpy as np

def estimate_pose(corners, tag_size, camera_matrix, dist_coeffs):
    """根据Apriltag的角点计算姿态"""
    obj_points = np.array([[-tag_size / 2, tag_size / 2, 0],
                           [tag_size / 2, tag_size / 2, 0],
                           [tag_size / 2, -tag_size / 2, 0],
                           [-tag_size / 2, -tag_size / 2, 0]], dtype=np.float32)

    image_points = np.array(corners, dtype=np.float32)

    # 使用solvePnP计算姿态
    success, rotation_vector, translation_vector = cv2.solvePnP(obj_points, image_points, camera_matrix, dist_coeffs)

    # 计算旋转矩阵
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # 计算Apriltag在相机坐标系中的中心位置
    tag_center_camera = -np.linalg.inv(rotation_matrix) @ translation_vector

    # 计算距离相机
    distance = np.linalg.norm(translation_vector)

    # 计算相机在世界坐标系中的位置
    camera_position_world = translation_vector

    return rotation_matrix, translation_vector, tag_center_camera, distance, camera_position_world, rotation_vector

def estimate_pose_from_state(translation_vector, tag_size, camera_matrix, dist_coeffs):
    """根据滤波后的平移向量计算姿态"""
    obj_points = np.array([[-tag_size / 2, tag_size / 2, 0],
                           [tag_size / 2, tag_size / 2, 0],
                           [tag_size / 2, -tag_size / 2, 0],
                           [-tag_size / 2, -tag_size / 2, 0]], dtype=np.float32)

    # 假设Apriltag的角点在世界坐标系中的位置
    # 这里假设Apriltag的角点在世界坐标系中的位置是已知的
    # 实际应用中可能需要根据Apriltag的ID和已知的角点位置来计算
    image_points = np.array([[-tag_size / 2, tag_size / 2],
                             [tag_size / 2, tag_size / 2],
                             [tag_size / 2, -tag_size / 2],
                             [-tag_size / 2, -tag_size / 2]], dtype=np.float32)

    # 使用solvePnP计算姿态
    success, rotation_vector, translation_vector = cv2.solvePnP(obj_points, image_points, camera_matrix, dist_coeffs)

    # 计算旋转矩阵
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # 计算Apriltag在相机坐标系中的中心位置
    tag_center_camera = -np.linalg.inv(rotation_matrix) @ translation_vector

    # 计算距离相机
    distance = np.linalg.norm(translation_vector)

    # 计算相机在世界坐标系中的位置
    camera_position_world = translation_vector

    return rotation_matrix, translation_vector, tag_center_camera, distance, camera_position_world, rotation_vector
