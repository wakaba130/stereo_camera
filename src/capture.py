##
# cording:utf-8
##

import os
import time
import cv2

def main():

    output_dir = "Capture"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'capture_left'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'capture_right'), exist_ok=True)

    capL = cv2.VideoCapture(0)
    if not capL.isOpened():
        print("Capture Open Error: Left")
        exit()
    capR = cv2.VideoCapture(2)
    if not capR.isOpened():
        print("Capture Open Error: Right")
        exit()

    image_size = (1280, 720)
    capL.set(cv2.CAP_PROP_FRAME_WIDTH, image_size[0])
    capL.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size[1])
    capR.set(cv2.CAP_PROP_FRAME_WIDTH, image_size[0])
    capR.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size[1])

    counter = 1
    while(1):
        time_start = time.perf_counter()
        retL, imgL = capL.read()
        time_L = time.perf_counter()
        retR, imgR = capR.read()
        time_R = time.perf_counter()
        if not retR or not retL:
            break

        print(f"left read time:{(time_L - time_start)*1000.0:.2f}[ms]")
        print(f"right read time:{(time_R - time_L)*1000.0:.2f}[ms]")

        cv2.imwrite(os.path.join(output_dir, f'capture_left/{counter:07d}.jpg'), imgL)
        cv2.imwrite(os.path.join(output_dir, f'capture_right/{counter:07d}.jpg'), imgR)
        counter += 1

        cv2.imshow("imgL", imgL)
        cv2.imshow("imgR", imgR)
        if cv2.waitKey(3) == 27:
            break

    cv2.destroyAllWindows()        
    capL.release()
    capR.release()

if __name__=="__main__":
    main()