# draw_april_tag.py
import cv2
import numpy as np

def draw_apriltag(image, corners, center, count):
    """绘制AprilTag的框和标记文本"""
    corners = corners.astype(int)
    a = tuple(corners[0])
    b = tuple(corners[1])
    c = tuple(corners[2])
    d = tuple(corners[3])

    # 绘制边框
    for i in range(4):
        cv2.line(image, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 0, 255), thickness=5, lineType=cv2.LINE_AA)

    # 绘制标记文本
    cv2.putText(image, f"{count}", (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
