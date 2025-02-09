import cv2
import numpy as np
from apriltag_detection import detect_april_tags
from draw.draw_apriltag import draw_apriltag
from draw.draw_axis import draw_axis
from apriltag_pose_estimation import estimate_pose
from rotationMatrixToEulerAngles import MatrixToEulerAngles
import csv

# 定义常量
TAG_SIZE = 0.0334  # AprilTag的物理尺寸，单位：米
CAMERA_MATRIX = np.array([[3.04968627e+03, 0, 1.51285758e+03],
                          [0, 3.05841216e+03, 2.12043328e+03],
                          [0, 0, 1]])  # 相机内参矩阵
DIST_COEFFS = np.array([1.60253106e-01, -1.17653405e+00, 7.02970072e-03, -1.64375577e-03, 2.98520009e+00])  # 相机畸变


# --------------------- EKF类定义 ---------------------
class TagEKF:
    """针对单个AprilTag的扩展卡尔曼滤波器"""

    def __init__(self, tag_id, init_position, dt=1 / 30):
        """
        状态向量: [x, y, z, vx, vy, vz] (单位：米和米/秒)
        观测向量: [x, y, z] (来自单帧检测的3D位置)
        """
        self.tag_id = tag_id
        self.dt = dt

        # 状态初始化（位置由首次检测确定，速度初始为0）
        self.state = np.array([init_position[0], init_position[1], init_position[2], 0, 0, 0])

        # 状态协方差矩阵（较大的初始不确定性）
        self.P = np.eye(6) * 10

        # 过程噪声（假设加速度噪声标准差为0.5 m/s²）
        q = 0.5
        G = np.array([[0.5 * dt ** 2], [0.5 * dt ** 2], [0.5 * dt ** 2], [dt], [dt], [dt]])
        self.Q = G @ G.T * q ** 2

        # 观测噪声（假设检测误差标准差为0.02米）
        self.R = np.eye(3) * (0.02) ** 2

    def predict(self):
        """预测步骤：使用匀速模型"""
        F = np.array([[1, 0, 0, self.dt, 0, 0],
                      [0, 1, 0, 0, self.dt, 0],
                      [0, 0, 1, 0, 0, self.dt],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1]])
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        """更新步骤：融合新的观测"""
        H = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0]])

        # 计算卡尔曼增益
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # 更新状态和协方差
        self.state += K @ (z - H @ self.state)
        self.P = (np.eye(6) - K @ H) @ self.P


# --------------------- 全局EKF管理器 ---------------------
class EKFManager:
    """管理所有检测到的AprilTag的EKF实例"""

    def __init__(self):
        self.ekf_instances = {}  # 格式：{tag_id: TagEKF实例}

    def process_frame(self, results, dt):
        """处理当前帧的检测结果"""
        current_tags = set()

        # 遍历所有检测到的tag
        for r in results:
            # 估计tag在相机坐标系中的3D位置
            rotation_matrix, translation_vector, *_ = estimate_pose(
                r.corners, TAG_SIZE, CAMERA_MATRIX, DIST_COEFFS
            )
            z = translation_vector.flatten()  # 观测值

            if r.tag_id not in self.ekf_instances:
                # 新检测到的tag，初始化EKF
                self.ekf_instances[r.tag_id] = TagEKF(r.tag_id, z[:3], dt)
            else:
                # 已有tag，执行预测+更新
                ekf = self.ekf_instances[r.tag_id]
                ekf.dt = dt  # 更新时间间隔
                ekf.predict()
                ekf.update(z[:3])

            current_tags.add(r.tag_id)

        # 处理未检测到的tag（仅预测）
        for tag_id in list(self.ekf_instances.keys()):
            if tag_id not in current_tags:
                ekf = self.ekf_instances[tag_id]
                ekf.predict()
                # 可选：长时间未检测到则移除
                # if ...: del self.ekf_instances[tag_id]

    def get_tag_position(self, tag_id):
        """获取指定tag的滤波后位置"""
        if tag_id in self.ekf_instances:
            return self.ekf_instances[tag_id].state[:3]
        return None


# --------------------- 主处理函数 ---------------------
def main():
    ekf_manager = EKFManager()
    video_path = './pic/video/1.mp4'
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = int(1000 / fps) if fps > 0 else 30
    prev_time = cv2.getTickCount()

    while True:
        # 读取帧并计算时间间隔
        ret, frame = cap.read()

        if not ret:
            print('视频结束')
            break

        current_time = cv2.getTickCount()
        dt = (current_time - prev_time) / cv2.getTickFrequency()  # 单位：秒
        prev_time = current_time

        # 检测AprilTag
        image, results = detect_april_tags(frame)
        if image is None:
            continue

        # EKF处理
        ekf_manager.process_frame(results, dt)

        # 可视化结果
        for r in results:
            # 获取原始检测结果
            rotation_matrix, translation_vector, *_ = estimate_pose(
                r.corners, TAG_SIZE, CAMERA_MATRIX, DIST_COEFFS
            )
            euler_angles = MatrixToEulerAngles(rotation_matrix)
            euler_degrees = np.degrees(euler_angles)

            # 获取滤波后的位置
            filtered_pos = ekf_manager.get_tag_position(r.tag_id)
            if filtered_pos is not None:
                # 用滤波后的位置替换原始检测值
                translation_vector = filtered_pos.reshape(3, 1)

            # 绘制tag边框和坐标轴
            draw_apriltag(image, r.corners, r.center, r.tag_id)
            draw_axis(image, r.center, rotation_matrix, translation_vector,
                      CAMERA_MATRIX, DIST_COEFFS)

            # 显示滤波后的位置
            cv2.putText(image, f"Filtered X: {filtered_pos[0]:.2f}m, Y: {filtered_pos[1]:.2f}m,Z: {filtered_pos[2]:.2f}m",
                        (int(r.center[0]) + 20, int(r.center[1]) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            # 显示欧拉角
            cv2.putText(image, f"Euler Angles: Z: {euler_degrees[0]:.5f}, ",
                        (int(r.center[0])+20,int(r.center[1])+40),cv2.FONT_HERSHEY_TRIPLEX,0.5,(0,255,255),2)
            cv2.putText(image, f"Euler Angles: Y: {euler_degrees[1]:.5f}",
                        (int(r.center[0]) + 20, int(r.center[1]) + 60), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(image, f"Euler Angles: X: {euler_degrees[2]:.5f}",
                        (int(r.center[0]) + 20, int(r.center[1]) + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 255), 2)

        # 显示帧率
        fps = 1.0 / dt
        cv2.putText(image, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 显示图像
        cv2.imshow('AprilTag Tracking with EKF', cv2.resize(image, (0, 0), fx=0.7, fy=0.7))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()