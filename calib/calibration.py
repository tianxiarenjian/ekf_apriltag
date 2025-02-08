import cv2
import numpy as np
import glob
import os

def calib(inter_corner_shape, size_per_grid, img_dir, img_type):
    w, h = inter_corner_shape
    cp_int = np.zeros((w * h, 3), np.float32)
    cp_int[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
    cp_world = cp_int * size_per_grid

    obj_points = []
    img_points = []
    images = glob.glob(os.path.join(img_dir, f'*.{img_type}'))
    for fname in images:
        img = cv2.imread(fname)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, cp_img = cv2.findChessboardCorners(gray_img, (w, h), None)
        if ret:
            obj_points.append(cp_world)
            img_points.append(cp_img)
            cv2.drawChessboardCorners(img, (w, h), cp_img, ret)
            cv2.namedWindow('FoundCorners', cv2.WINDOW_NORMAL)
            cv2.resizeWindow("FoundCorners", 600, 600)
            cv2.imshow('FoundCorners', img)
            cv2.waitKey(1)
    cv2.destroyAllWindows()

    ret, mat_inter, coff_dis, v_rot, v_trans = cv2.calibrateCamera(obj_points, img_points, gray_img.shape[::-1], None, None)
    print("internal matrix:\n", mat_inter)
    print("------------------------------------------------------------------")
    print("distortion coefficients:\n", coff_dis)
    print("------------------------------------------------------------------")
    print("rotation vectors:\n", v_rot)
    print("------------------------------------------------------------------")
    print("translation vectors:\n", v_trans)
    print("------------------------------------------------------------------")

    total_error = 0
    for i in range(len(obj_points)):
        img_points_repro, _ = cv2.projectPoints(obj_points[i], v_rot[i], v_trans[i], mat_inter, coff_dis)
        error = cv2.norm(img_points[i], img_points_repro, cv2.NORM_L2) / len(img_points_repro)
        total_error += error
    print("Average Error of Reproject: ", total_error / len(obj_points))
    return mat_inter, coff_dis
