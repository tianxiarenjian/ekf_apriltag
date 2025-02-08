# draw_axis.py
import cv2
import numpy as np


def draw_axis(image, center, rotation_matrix, translation_vector, camera_matrix, dist_coeffs,
                origin=(0, 0), length=0.1):
    """绘制坐标轴"""
    # 定义坐标轴的端点
    points = np.float32([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]]).reshape(-1, 3)

    # 将3D点投影到2D图像平面上
    image_points, _ = cv2.projectPoints(points, rotation_matrix, translation_vector, camera_matrix, dist_coeffs)

    # 将点转换成整数类型
    image_points = image_points.astype(int)

    # 提取中心点和轴的端点
    center = tuple(image_points[0].ravel())
    x = tuple(image_points[1].ravel())
    y = tuple(image_points[2].ravel())
    z = tuple(image_points[3].ravel())

    # 绘制坐标轴
    cv2.arrowedLine(image, center, x, (0, 0, 255), 3)  # X轴 - 红色
    cv2.arrowedLine(image, center, y, (0, 255, 0), 3)  # Y轴 - 绿色
    cv2.arrowedLine(image, center, z, (255, 0, 0), 3)  # Z轴 - 蓝色

    # # 投影相机位置
    # camera_position_image, _ = cv2.projectPoints(camera_position_world.reshape(-1, 3), rotation_vector,
    #                                              translation_vector, camera_matrix, dist_coeffs)
    # camera_position_image = tuple(camera_position_image[0].ravel())
    #
    # # 绘制由原点指向相机的箭头
    # cv2.arrowedLine(image, center, camera_position_image, (255, 255, 0), 4)  # 箭头 - 黄色

    return image
