##
# cording:utf-8
##

import os
import numpy as np
import cv2
import json
import argparse
from glob import glob
from collections import deque

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, default='output_single_left')
    parser.add_argument('-s', '--square_size', type=float, default=25.0)
    parser.add_argument('-v', '--video_id', type=int, default=None)
    parser.add_argument('-i', '--input_dir', type=str, default='output_single_left')
    parser.add_argument('--pattern_size', type=str, default='10x7')
    parser.add_argument('--sleep', type=int, default=2)
    parser.add_argument('--capture_end', type=int, default=20)
    parser.add_argument('--exe', type=str, default='png')
    return parser.parse_args()

class ImageReader:
    def __init__(self, img_path:str, exe:str) -> None:
        self.img_path = img_path
        self.exe = exe
        self.img_names = glob(os.path.join(img_path, f"*.{exe}"))
        self.img_names_size = len(self.img_names)
        self.frame_counter = 0
        self.width = None
        self.height = None
        if self.img_names_size <= 0:
            print("Warning: image read 0")
            return
        img = cv2.imread(self.img_names[0])
        h, w, c = img.shape
        self.width = w
        self.height = h

    def getSize(self):
        return (self.height, self.width)

    def isOpened(self):
        if self.img_names_size > 0:
            return True
        return False

    def read(self):
        if self.frame_counter >= self.img_names_size:
            return False, None

        img = cv2.imread(self.img_names[self.frame_counter])
        self.frame_counter += 1
        return True, img

def main():
    args = argparser()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    square_size = args.square_size

    ph, pw = args.pattern_size.split('x')
    pattern_size = (int(ph), int(pw))
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    obj_points = []
    img_points = []

    if type(args.input_dir) is str:
        cap = ImageReader(os.path.join(args.input_dir) , args.exe)
        h, w = cap.getSize()
        sleep_time = 0
    elif type(args.video_id) is int:
        cap = cv2.VideoCapture(args.video_id)
        if not cap.isOpened():
            print("Capture Open Error")
            return
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        ret, img = cap.read()
        h, w = img.shape[:2]
        sleep_time = cap.get(cv2.CAP_PROP_FPS) * args.sleep
    else:
        print("input_dir and video_id is None")
        exit()

    capEnd = args.capture_end

    cap_flag_que = deque(maxlen=3)
    sleep_frame = 0
    counter = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break

        viz = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern_size)
        cap_flag_que.append(found)
        if found:
            cv2.drawChessboardCorners(viz, pattern_size, corners, found)

        if found and cap_flag_que.count(True) == 3 and sleep_frame >= sleep_time:
            print("Count:{}".format(counter))
            img_points.append(corners)
            obj_points.append(pattern_points)
            cv2.imwrite(os.path.join(output_dir, '{:07d}.png'.format(counter)), img)
            counter += 1
            sleep_frame = 0

        sleep_frame += 1
        cv2.imshow('img', viz)
        if capEnd <= counter or cv2.waitKey(3) == 27:
            break

    if args.video_id:
        cap.release()

    print("estimate... {}".format(len(img_points)))
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)

    print("\nRMS:", rms)
    print("camera matrix:\n", camera_matrix)
    print("distortion coefficients: ", dist_coefs.ravel())

    with open(os.path.join(output_dir, 'calibrate.json'), 'w') as fp:
        param = {'RMS': rms,
                 'camera_matrix': camera_matrix.tolist(),
                 'distortion': dist_coefs.ravel().tolist()}
        json.dump(param, fp, indent=4)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    