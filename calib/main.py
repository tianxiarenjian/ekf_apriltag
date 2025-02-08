from calibration import calib
from dedistortion import dedistortion
from utils import ensure_dir_exists

if __name__ == '__main__':
    inter_corner_shape = (11, 8)
    size_per_grid = 0.02
    img_dir = "../pic/photo_calib"
    img_type = "jpg"
    save_dir = "../pic/photo-dedistortion"

    ensure_dir_exists(save_dir)
    # calibrate the camera
    mat_inter, coff_dis = calib(inter_corner_shape, size_per_grid, img_dir, img_type)
    # dedistort and save the dedistortion result.(选用)
    #dedistortion(inter_corner_shape, img_dir, img_type, save_dir, mat_inter, coff_dis)
