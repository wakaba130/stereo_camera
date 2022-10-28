##
# cording:utf-8
##

import os
import time
import numpy as np
import cv2
import json
import argparse
from collections import deque

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, default='output_stereo_camera')
    parser.add_argument('-s', '--square_size', type=float, default=25.0)
    parser.add_argument('-v', '--video_ids', type=str, default="0,2", help='exp) --video_ids 0,2')
    parser.add_argument('-l', '--input_left_json', type=str, default='output_single_left/calibrate.json')
    parser.add_argument('-r', '--input_right_json', type=str, default='output_single_right/calibrate.json')
    parser.add_argument('--pattern_size', type=str, default='10x7', help='exp) --pattern_size 10x7')
    parser.add_argument('--sleep', type=int, default=3)
    parser.add_argument('--capture_end', type=int, default=20)
    return parser.parse_args()

def main():

    args = argparser()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'left'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'right'), exist_ok=True)
    square_size = 25.0

    calib_params_dict = {}
    with open(os.path.join(args.input_left_json), 'r') as fp:
        camL_param = json.load(fp)
        camera_matrix_l = np.array(camL_param["camera_matrix"], dtype=np.float64)
        dist_coeffs_l = np.array(camL_param["distortion"], dtype=np.float64)
        calib_params_dict['camera_matrix_l'] = camera_matrix_l.tolist()
        calib_params_dict['dist_coeffs_l'] = dist_coeffs_l.tolist()

    with open(args.input_right_json, 'r') as fp:
        camR_param = json.load(fp)
        camera_matrix_r = np.array(camR_param["camera_matrix"], dtype=np.float64)
        dist_coeffs_r = np.array(camR_param["distortion"], dtype=np.float64)
        calib_params_dict['camera_matrix_r'] = camera_matrix_r.tolist()
        calib_params_dict['dist_coeffs_r'] = dist_coeffs_r.tolist()

    pattern_size = args.pattern_size
    ph, pw = pattern_size.split('x')
    pattern_size = (int(ph), int(pw))
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    video_ids = args.video_ids
    lp, rp = video_ids.split(',')
    capL = cv2.VideoCapture(int(lp))
    if not capL.isOpened():
        print("Capture Open Error: Left")
        exit()
    capR = cv2.VideoCapture(int(rp))
    if not capR.isOpened():
        print("Capture Open Error: Right")
        exit()

    capL.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    capL.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capR.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    capR.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

    sleep_time = int(capL.get(cv2.CAP_PROP_FPS) * args.sleep)

    retL, imgL = capL.read()
    img_height, img_width = imgL.shape[:2]
    img_size = (img_width, img_height)

    img_points_l = []
    img_points_r = []

    cap_flag_que = deque(maxlen=3)
    sleep_frame = 0
    counter = 0
    while(1):
        time_start = time.perf_counter()
        retL, imgL = capL.read()
        time_L = time.perf_counter()
        retR, imgR = capR.read()
        time_R = time.perf_counter()
        if not retR or not retL:
            break

        #print(f" left read time: {(time_L - time_start)*1000:.2f}[ms]")
        #print(f"right read time: {(time_R - time_L)*1000:.2f}[ms]")

        gimgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        gimgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        vizL = imgL.copy()
        vizR = imgR.copy()
        foundL, corners_l = cv2.findChessboardCorners(gimgL, pattern_size)
        foundR, corners_r = cv2.findChessboardCorners(gimgR, pattern_size)
        cv2.drawChessboardCorners(vizL, pattern_size, corners_l, foundL)
        cv2.drawChessboardCorners(vizR, pattern_size, corners_r, foundR)
        if foundL and foundR:
            cap_flag_que.append(True)
        else:
            cap_flag_que.append(False)

        if foundL and foundR and cap_flag_que.count(True) == 3 and sleep_frame >= sleep_time:
            print("Count:{}".format(counter))
            #corners_l2 = cv2.cornerSubPix(gimgL, corners_l, (11,11), (-1,-1), criteria)
            #corners_r2 = cv2.cornerSubPix(gimgR, corners_r, (11,11), (-1,-1), criteria)
            img_points_l.append(corners_l)
            img_points_r.append(corners_r)
            
            cv2.imwrite(os.path.join(os.path.join(output_dir, 'left'), '{:07d}.png'.format(counter)), imgL)
            cv2.imwrite(os.path.join(os.path.join(output_dir, 'right'), '{:07d}.png'.format(counter)), imgR)
            counter += 1
            sleep_frame = 0

        if counter > args.capture_end:
            break

        sleep_frame += 1
        cv2.imshow('imgL', vizL)
        cv2.imshow('imgR', vizR)
        if cv2.waitKey(3) == 27:
            break

    capL.release()
    capR.release()
    cv2.destroyAllWindows()

    obj_points = None
    if counter != 0:
        obj_points = [pattern_points] * counter
    else:
        print("there are no obj_points!")
        exit()


    ## キャリブレーション後の内部パラメータと歪み補正パラメータを取得
    ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(obj_points,
                                                      img_points_l, img_points_r,
                                                      camera_matrix_l, dist_coeffs_l,
                                                      camera_matrix_r, dist_coeffs_r,
                                                      img_size,
                                                      flags = cv2.CALIB_FIX_INTRINSIC)

    calib_params_dict['R'] = R.tolist()
    calib_params_dict['T'] = T.tolist()

    # キャリブレーション済みステレオの，それぞれのカメラの平行化変換を求める
    R1, R2, P1, P2, Q, ROI1, ROI2 = cv2.stereoRectify(camera_matrix_l,
                                                      dist_coeffs_l,
                                                      camera_matrix_r,
                                                      dist_coeffs_r,
                                                      img_size,
                                                      R, T,
                                                      flags=cv2.CALIB_ZERO_DISPARITY,
                                                      alpha=0)
    calib_params_dict['R1'] = R1.tolist()
    calib_params_dict['R2'] = R2.tolist()
    calib_params_dict['P1'] = P1.tolist()
    calib_params_dict['P2'] = P2.tolist()
    calib_params_dict['Q'] = Q.tolist()
    calib_params_dict['ROI1'] = ROI1
    calib_params_dict['ROI2'] = ROI2


    mapx1, mapy1 = cv2.initUndistortRectifyMap(camera_matrix_l, dist_coeffs_l,
                                               R1, P1, img_size, cv2.CV_32FC1)
    mapx2, mapy2 = cv2.initUndistortRectifyMap(camera_matrix_r, dist_coeffs_r,
                                               R2, P2, img_size, cv2.CV_32FC1)

    np.savez(os.path.join(output_dir, "stereo_map_xy.npz"),
             map_left_x=mapx1, map_left_y=mapy1, map_right_x=mapx2, map_right_y=mapy2)

    with open(os.path.join(output_dir, "stereo_params.json"), 'w') as fp:
        json.dump(calib_params_dict, fp, indent=4)

if __name__=="__main__":
    main()