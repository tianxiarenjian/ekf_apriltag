from rotationMatrixToEulerAngles import MatrixToEulerAngles
import numpy as np
def euler_angles(rotation_matrix):
    # 输出欧拉角
    euler_angles = MatrixToEulerAngles(rotation_matrix)
    euler_angles_degrees = np.degrees(euler_angles)
    print(f"欧拉角(ZYX):\n{euler_angles_degrees}")