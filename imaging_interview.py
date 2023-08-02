import cv2
import numpy as np
import imutils
import glob
import os
import sys
import shutils

PATH = "./dataset"

def draw_color_mask(img, borders, color=(0, 0, 0)):
    h = img.shape[0]
    w = img.shape[1]

    x_min = int(borders[0] * w / 100)
    x_max = w - int(borders[2] * w / 100)
    y_min = int(borders[1] * h / 100)
    y_max = h - int(borders[3] * h / 100)

    img = cv2.rectangle(img, (0, 0), (x_min, h), color, -1)
    img = cv2.rectangle(img, (0, 0), (w, y_min), color, -1)
    img = cv2.rectangle(img, (x_max, 0), (w, h), color, -1)
    img = cv2.rectangle(img, (0, y_max), (w, h), color, -1)

    return img


def preprocess_image_change_detection(img, gaussian_blur_radius_list=None, black_mask=(5, 10, 5, 0)):
    gray = img.copy()
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if gaussian_blur_radius_list is not None:
        for radius in gaussian_blur_radius_list:
            gray = cv2.GaussianBlur(gray, (radius, radius), 0)

    gray = draw_color_mask(gray, black_mask)

    return gray


def compare_frames_change_detection(prev_frame, next_frame, min_contour_area):
    prev_frame = preprocess_image_change_detection(prev_frame, [5,5])
    next_frame = preprocess_image_change_detection(next_frame, [5,5])
    frame_delta = cv2.absdiff(prev_frame, next_frame)
    thresh = cv2.threshold(frame_delta, 127, 255, cv2.THRESH_BINARY)[1]

    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    score = 0
    res_cnts = []
    for c in cnts:
        if cv2.contourArea(c) < min_contour_area:
            continue
        res_cnts.append(c)
        score += cv2.contourArea(c)

    return score, res_cnts, thresh


if __name__ =='__main__':
    curr_dir = os.getcwd()
    image_files = glob.glob(os.path.join(curr_dir, "dataset/*"))
    start = 0
    stop = 10
    score_list = np.array([])
    files_to_delete= []
    for index_i in image_files:
        img = cv2.imread(index_i)
        if img is not None:
            img = cv2.resize(img, (640, 480))
            cv2.imwrite(index_i, img)
    for index_i in image_files[start:stop:]:
        for index_j in image_files[start+1:stop:]:
            print(index_j)
            prev_frame = cv2.imread(index_i)
            next_frame = cv2.imread(index_j)
            if(prev_frame is not None) and (next_frame is not None):
                score, res_cnts, thresh = compare_frames_change_detection(prev_frame, next_frame, 10)
                if score < 1000:#Images are similar
                    files_to_delete.append(index_j)
                else: #images are similar
                    score_list = np.append(score_list,score)
                    continue
    for files in files_to_delete:
        os.remove(files)
    print(score_list)

        
    
    