# show_apriltag.py
import cv2
import numpy as np
from apriltag_detection import detect_april_tags
from draw.draw_apriltag import draw_apriltag
from draw.draw_axis import draw_axis
from apriltag_pose_estimation import estimate_pose
from rotationMatrixToEulerAngles import MatrixToEulerAngles


# 定义常量
TAG_SIZE = 0.0334  # AprilTag的物理尺寸，单位：米
CAMERA_MATRIX = np.array([[3.04968627e+03, 0, 1.51285758e+03],
                          [0, 3.05841216e+03, 2.12043328e+03],
                          [0, 0, 1]])  # 相机内参矩阵
DIST_COEFFS = np.array([1.60253106e-01, -1.17653405e+00, 7.02970072e-03, -1.64375577e-03, 2.98520009e+00])  # 相机畸变

def show_apriltag_combined(image, results, tag_size, camera_matrix, dist_coeffs):
    try:
        if not results:
            print("未检测到任何AprilTag")
            return

        for idx, r in enumerate(results, 1):
            # 绘制边框和中心点
            draw_apriltag(image, r.corners, r.center, idx)

            # 估计姿态
            rotation_matrix, translation_vector, tag_center_camera, distance, camera_position_world, rotation_vector = estimate_pose(
                r.corners, tag_size, camera_matrix, dist_coeffs
            )

            # 计算并打印欧拉角
            euler_angles = MatrixToEulerAngles(rotation_matrix)
            euler_degrees = np.degrees(euler_angles)
            print(f"Apriltag36h11-{r.tag_id} 欧拉角(ZYX): {euler_degrees}")

            # 输出相对位姿
            print(f"Apriltag36h11-{r.tag_id} 的相对位姿:")
            print(f"旋转矩阵:\n{rotation_matrix}")
            print(f"旋转向量:\n{rotation_vector}")
            print(f"平移向量:\n{translation_vector}")
            print(f"相对位置:\n{tag_center_camera}")
            print(f"距离相机:\n{distance}")

            # 绘制坐标轴
            draw_axis(image, r.center, rotation_matrix, translation_vector,
                      camera_matrix, dist_coeffs)

        display_image(image)
    finally:
        cv2.destroyAllWindows()

def display_image(image):
    """显示图像的辅助函数"""
    # 将图片大小缩放1/2
    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    # 创建可调节的窗口并显示图像
    cv2.namedWindow('Image with Apriltags', cv2.WINDOW_NORMAL)
    cv2.imshow('Image with Apriltags', image)
    cv2.waitKey(0)


if __name__ == "__main__":
    image_path = '.\\pic\\photo_apriltag\\tag4.jpg'
    image, results = detect_april_tags(image_path)

    if image is None:
        print("无法读取图像文件")
        exit()

    show_apriltag_combined(image,results,TAG_SIZE,CAMERA_MATRIX,DIST_COEFFS)
