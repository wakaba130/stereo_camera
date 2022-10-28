##
# coding:utf-8
##

import os
import json
from typing import Tuple
import numpy as np
import cv2
from random import randint
from glob import glob


class MatchingSGBM:
    def __init__(self, max_disparity:int=256, window_size:int=15) -> None:
        self.stereoProcessor = cv2.StereoSGBM_create(
                                        minDisparity=0,
                                        numDisparities = max_disparity, # max_disp has to be dividable by 16 f. E. HH 192, 256
                                        blockSize=window_size,
                                        #P1=8 * window_size ** 2,       # 8*number_of_image_channels*SADWindowSize*SADWindowSize
                                        #P2=32 * window_size ** 2,      # 32*number_of_image_channels*SADWindowSize*SADWindowSize
                                        #disp12MaxDiff=1,
                                        #uniquenessRatio=15,
                                        #speckleWindowSize=0,
                                        #speckleRange=2,
                                        #preFilterCap=63,
                                        mode=cv2.STEREO_SGBM_MODE_HH
                                )
        self.max_scale = self.stereoProcessor.getNumDisparities() * cv2.StereoMatcher_DISP_SCALE
    
    def compute(self, imgL:np.ndarray, imgR:np.ndarray) -> np.ndarray:
        if len(imgL.shape) == 1:
            gimgL = imgL.copy()
            gimgR = imgR.copy()
        else:
            gimgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
            gimgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        disparity_UMat = self.stereoProcessor.compute(cv2.UMat(gimgL),cv2.UMat(gimgR))
        return cv2.UMat.get(disparity_UMat)

    def draw_color_disparity(self, disparity:np.ndarray) -> np.ndarray:
        disp_buf = disparity.astype(np.double) * 256 / self.max_scale
        disp_8u = disp_buf.astype(np.uint8)
        colored = cv2.applyColorMap(disp_8u, cv2.COLORMAP_JET)
        colored[disparity < 0] = np.array([0, 0, 0], dtype = np.uint8)
        return colored

class StereoParam:
    def __init__(self, calib_dir, img_size:Tuple) -> None:
        fp = open(os.path.join(calib_dir, 'stereo_params.json'), 'r')
        calib_params = json.load(fp)

        self.R = np.array(calib_params['R'], dtype=np.float64)
        self.T = np.array(calib_params['T'], dtype=np.float64)
        self.Q = np.array(calib_params['Q'], dtype=np.float64)
        self.dstT = np.linalg.norm(self.T, ord=2)
        self.camera_matrix_r = np.array(calib_params['camera_matrix_r'], dtype=np.float64)
        self.dist_coeffs_r = np.array(calib_params['dist_coeffs_r'], dtype=np.float64)
        self.camera_matrix_r = np.array(calib_params['camera_matrix_r'], dtype=np.float64)
        self.dist_coeffs_r = np.array(calib_params['dist_coeffs_r'], dtype=np.float64)

        map_xy = np.load(os.path.join(calib_dir, "stereo_map_xy.npz"))
        self.map1x, self.map1y = map_xy['map_left_x'], map_xy['map_left_y']
        self.map2x, self.map2y = map_xy['map_right_x'], map_xy['map_right_y']
    
        self.width_list = [randint(300, img_size[0]-10) for i in range(30)]
        self.height_list = [randint(10, img_size[1]-10) for i in range(30)]

    def remap(self, imgL:np.ndarray, imgR:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        re_left_img = cv2.remap(imgL, self.map1x, self.map1y, cv2.INTER_NEAREST)
        re_right_img = cv2.remap(imgR, self.map2x, self.map2y, cv2.INTER_NEAREST)
        return re_left_img, re_right_img

    def get3DPoint(self, disparity:np.ndarray) -> np.ndarray:
        point3d = cv2.reprojectImageTo3D(disparity, self.Q, handleMissingValues=False)
        point3d_buf = np.where(point3d == np.inf, 0, point3d)
        point3d = np.where(point3d_buf == -1*np.inf, 0, point3d_buf)
        return np.where(point3d < 0, 0, point3d)
    
    def getRandomPointDistance_3DPoint(self, img:np.ndarray, point3d:np.ndarray) -> np.ndarray:
        buf_img = img.copy()
        draw_color = (0, 0, 255)
        for i, j in zip(self.height_list, self.width_list):
            z = np.linalg.norm(point3d[i][j], ord=2) * 25.0
            #z = point3d[i][j][2] * 25.0
            if z > 0:
                cv2.circle(buf_img, (j,i), 3, draw_color, -1)
                cv2.putText(buf_img, f"{z:.2f}[mm]", (j,i), cv2.FONT_HERSHEY_COMPLEX, 0.6, draw_color)
        return buf_img
    
    def getRandomPointDistance_disp(self, img:np.ndarray, disparity:np.ndarray) -> np.ndarray:
        buf_img = img.copy()
        draw_color = (0, 255, 0)
        for i, j in zip(self.height_list, self.width_list):
            zd = self.camera_matrix_r[0][0]*self.dstT/disparity[i][j] * 25.0
            if zd > 0:
                cv2.circle(buf_img, (j,i), 3, draw_color, -1)
                cv2.putText(buf_img, f"{zd:.2f}[mm]", (j,i-15), cv2.FONT_HERSHEY_COMPLEX, 0.6, draw_color)
        return buf_img

def main():

    output_dir = "output"
    calib_dir = "output_stereo_camera"
    input_dir = "Capture"
    os.makedirs(os.path.join(output_dir, f'remap'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, f'disp'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, f'output'), exist_ok=True)

    left_imgs = glob(os.path.join(input_dir, 'capture_left/*.jpg'))
    right_imgs = glob(os.path.join(input_dir, 'capture_right/*.jpg'))
    left_imgs.sort()
    right_imgs.sort()

    if len(left_imgs) <= 0 and len(right_imgs) <= 0:
        print("img load error")
        exit()

    imgL = cv2.imread(left_imgs[0])
    img_height, img_width = imgL.shape[:2]
    img_size = (img_width, img_height)

    matching = MatchingSGBM()
    stereo_params = StereoParam('output_stereo_camera', img_size)

    counter = 1
    for img_name_l, img_name_r in zip(left_imgs, right_imgs):
        imgL = cv2.imread(img_name_l)
        imgR = cv2.imread(img_name_r)

        re_left_img, re_right_img = stereo_params.remap(imgL, imgR)
        disparity = matching.compute(re_left_img, re_right_img)
        colored = matching.draw_color_disparity(disparity)
        point3d = stereo_params.get3DPoint(disparity)
        vimg = stereo_params.getRandomPointDistance_3DPoint(imgL, point3d)
        vimg = stereo_params.getRandomPointDistance_disp(vimg, disparity)
        
        cv2.imwrite(os.path.join(output_dir, f'remap/d{counter:07d}_left.png'), imgL)
        cv2.imwrite(os.path.join(output_dir, f'remap/{counter:07d}_right.png'), imgR)
        cv2.imwrite(os.path.join(output_dir, f'output/{counter:07d}_right.png'), vimg)
        cv2.imwrite(os.path.join(output_dir, f'disp/{counter:07d}.png'), colored)
        counter += 1

        cv2.imshow("View", vimg)
        cv2.imshow("disp", colored)
        if cv2.waitKey(3) == 27:
            break

if __name__=="__main__":
    main()

