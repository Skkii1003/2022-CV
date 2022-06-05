import os

import numpy as np
import cv2

def cal_args(images,grid=(9,6)):
    obj_points = []
    img_points = []

    obj_point = np.zeros((grid[0] * grid[1], 3), np.float32)
    obj_point[:, :2] = np.mgrid[0:grid[0], 0:grid[1]].T.reshape(-1, 2)
    for img in images:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret,corners = cv2.findChessboardCorners(gray,grid,None)
        if ret:
            obj_points.append(obj_point)
            img_points.append(corners)
    return obj_points,img_points

def calibrate(img,obj_points,img_points):
    ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(obj_points,img_points,img.shape[1::-1],None,None)
    dst = cv2.undistort(img,mtx,dist,None,mtx)
    return dst

def get_images_by_dir(dirname):
    img_names = os.listdir(dirname)
    img_paths = [dirname+'/'+img_name for img_name in img_names]
    imgs = [cv2.imread(path) for path in img_paths]
    return imgs

cal_imgs = get_images_by_dir('camera_cal')
obj_points,img_points = cal_args(cal_imgs,grid=(9,6))
test_imgs = get_images_by_dir('../sample/image')
ret_imgs = []
for img in test_imgs:
    img = calibrate(img,obj_points,img_points)
    ret_imgs.append(img)

outpath = "output/"
for i in range(0,len(ret_imgs)):
    cv2.imwrite(outpath+str(i+1)+".png",ret_imgs[i])

print("相机矫正完成")