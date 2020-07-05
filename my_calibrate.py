import cv2 as cv
import numpy as np
import os
from utils.homography import get_homography
from utils.intrinsics import get_intrinsics_param
from utils.extrinsics import get_extrinsics_param
from utils.distortion import get_distortion
from utils.refine_all import refinall_all_param
 
 
def calibrate(pic_points, real_points_x_y):
    H = get_homography(pic_points, real_points_x_y)
    # print(H.shape)
    intrinsics_param = get_intrinsics_param(H)
    # print(intrinsics_param)
 
    extrinsics_param = get_extrinsics_param(H, intrinsics_param)
    # print(extrinsics_param)
 
    k = get_distortion(intrinsics_param, extrinsics_param, pic_points, real_points_x_y)
    # print(k)
 
    [new_intrinsics_param, new_k, new_extrinsics_param]  = refinall_all_param(intrinsics_param,
                                                            k, extrinsics_param, real_points, pic_points)
 
    print("intrinsics_parm:\n", new_intrinsics_param)
    print("distortionk:\n", new_k)
    # print("extrinsics_parm:\t", new_extrinsics_param)
 
 
if __name__ == "__main__":
    file_dir = r'./left'
    pic_name = os.listdir(file_dir)
 
    # 由于棋盘为二维平面，设定世界坐标系在棋盘上，一个单位代表一个棋盘宽度，产生世界坐标系三维坐标
    cross_corners = [9, 6] #棋盘方块交界点排列
    real_coor = np.zeros((cross_corners[0] * cross_corners[1], 3), np.float32)
    real_coor[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
 
    real_points = []
    real_points_x_y = []
    pic_points = []
 
    for pic in pic_name:
        pic_path = os.path.join(file_dir, pic)
        pic_data = cv.imread(pic_path)
 
        # 寻找到棋盘角点
        succ, pic_coor = cv.findChessboardCorners(pic_data, (cross_corners[0], cross_corners[1]), None)
 
        if succ:
            # 添加每幅图的对应3D-2D坐标
            pic_coor = pic_coor.reshape(-1, 2)
            pic_points.append(pic_coor)
 
            real_points.append(real_coor)
            real_points_x_y.append(real_coor[:, :2])
    calibrate(pic_points, real_points_x_y)
