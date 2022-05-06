#================================================================
#
#   File name   : detection_custom.py
#   Author      : PyLessons
#   Created date: 2020-09-17
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : object detection image and video example
#
#================================================================
import os

from cv2 import CV_32F
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
##from yolov3.utils import load_yolo_weights, image_preprocess, draw_bbox, bboxes_iou, nms, postprocess_boxes,
##                    Predict_bbox_mp, postprocess_mp, Show_Image_mp, detect_realtime, detect_video, Load_Yolo_model, detect_video_realtime_mp
from yolov3.utils import load_yolo_weights, detect_image, Load_Yolo_model
from yolov3.configs import *
import measure_pippette as mp
import grabcut
from liquid_counter import LiquidCounter

import time

count = 0
video_path   = "../sample_videos/Video-2020_1008_185556-1.2_with_liquid.mp4"
##image_path = "C:/Users/Yukan/Desktop/YOLO/IMAGES/test4.png"
yolo = Load_Yolo_model()
coors = []
mask = cv2.imread("mask_copy_1.png", 0)
cap = cv2.VideoCapture(video_path)
reader = LiquidCounter()
start = time.time()
while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        break
    image_path = ("frame%d.jpg" % count)
    image_path = os.path.join('frames', image_path)
    #cv2.imwrite(image_path, frame)
    coors, image = detect_image(yolo, frame, "./IMAGES/pipette_detect.jpg", input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
    img = []
    if coors:
        img = image[coors[1]-5:coors[3]+5, coors[0]-5:coors[2]+5]
        img = cv2.resize(img, (75,375))
        #img = mp.equalise_histogram(img)
        img = grabcut.find_pipette(img,mask)
        if np.argwhere(img).size == 0:
            continue
        thresh = mp.thresh_image(img)
        contour_map = mp.find_horizontal_edges(thresh)
        #cv2.imshow('contour_map', contour_map)
        #cv2.imshow('img', img)
        liquid_level,num_white = mp.find_liquid_level(img)
        bottom_y = mp.lowest_corner(img)
        top_y = mp.highest_corner(img)
        if num_white > 2500:
            volume = mp.return_liquid_level(img, bottom_y, liquid_level)
            #draw line at liquid_level
            reader.update(volume)
            #cv2.line(img, (0, int(liquid_level)), (img.shape[1], int(liquid_level)), (0, 255, 0), 2)
            #cv2.putText(img, "%.2f" % volume, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
            #cv2.imwrite(image_path, img)
    count = count + 1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
end = time.time()
print(end-start)
cap.release()
cv2.destroyAllWindows() # destroy cd ../..all opened windows
cap = cv2.VideoCapture(video_path)

while not cap.isOpened():
    cap = cv2.VideoCapture(video_path)
    cv2.waitKey(1000)
    print ("Wait for the header")
