import cv2
import glob
import os

def dedistortion(inter_corner_shape, img_dir, img_type, save_dir, mat_inter, coff_dis):
    # 获取棋盘格角点的内部角点数量
    w, h = inter_corner_shape

    # 获取指定目录下所有指定类型的图像文件路径
    images = glob.glob(os.path.join(img_dir, f'*.{img_type}'))

    # 遍历每一张图像
    for fname in images:
        # 获取图像文件名
        img_name = os.path.basename(fname)

        # 读取图像
        img = cv2.imread(fname)

        # 计算新的相机矩阵和感兴趣区域（ROI）
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mat_inter, coff_dis, (w, h), 1, (w, h))

        # 对图像进行去畸变处理
        dst = cv2.undistort(img, mat_inter, coff_dis, None, newcameramtx)

        # 根据ROI裁剪图像
        # x, y, roi_w, roi_h = roi
        # dst = dst[y:y + roi_h, x:x + roi_w]

        # 保存去畸变后的图像到指定目录
        cv2.imwrite(os.path.join(save_dir, img_name), dst)

    # 打印成功信息
    print('Dedistorted images have been saved to: %s successfully.' % save_dir)
