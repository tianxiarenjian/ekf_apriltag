# apriltag_detection.py
import cv2
import numpy as np
import pupil_apriltags as apriltag

def detect_april_tags(image_or_path):
    """检测图像中的AprilTag并返回检测结果"""
    # 检查输入是文件路径还是图像数组
    if isinstance(image_or_path, str):
        image = cv2.imread(image_or_path)
        if image is None:
            print(f"无法读取图像文件: {image_or_path}")
            return None, None
    else:
        image = image_or_path

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 创建一个apriltag检测器
    detector = apriltag.Detector(families='tag36h11')
    results = detector.detect(gray)

    return image, results
