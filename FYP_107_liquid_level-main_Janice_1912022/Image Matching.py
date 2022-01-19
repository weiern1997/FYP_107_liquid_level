import cv2 as cv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def normalize_filled(img):
##    img = cv2.imread(img, 0)
##    image = img.copy()
####    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
##    cnt, hier = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
##    im, cnt, _ = cv2.findContours(canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # fill shape
##    cv2.fillPoly(img, pts=cnt, color=(255,255,255))
##    bounding_rect = cv2.boundingRect(cnt[0])
##    img_cropped_bounding_rect = img[bounding_rect[1]:bounding_rect[1] + bounding_rect[3], bounding_rect[0]:bounding_rect[0] + bounding_rect[2]]
    # resize all to same size
##    img_resized = cv2.resize(img_cropped_bounding_rect, (300, 300))
    #Download the videos and create a directory called sample_videos
    cap = cv.VideoCapture('sample_videos\Video-2020_1008_185556-1.2_with_liquid.mp4')
#read the pippette mask
    template = cv.imread("mask.png")
    template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
#get the contour of the pipette
    template_contour,hierachy = cv.findContours(template, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    img = cv.drawContours(template, template_contour, -1, (0,255,0), 2)
    cv.imshow("OpenCV Image Reading", img)
    cv.waitKey(0)
##    return img_resized
    
imgs = ["Pipette1.jpg", "Pipette_with_water.png"]
imgs = [normalize_filled(i) for i in imgs]

for i in range(1, 2):
    plt.subplot(2, 3, i), plt.imshow(imgs[i - 1], cmap='gray')
    print(cv2.matchShapes(imgs[0], imgs[i - 1], 1, 0.0))

##image = ("Pipette1.png")
####img = np.array(Image.open(image))
##img = cv2.imread(image, 0)
####img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
##cv2.imshow("OpenCV Image Reading", img)
##cv2.waitKey(0)

##EXTRA
##while not cap.isOpened():
##    cap = cv.VideoCapture('Video-2020_1008_185556-1.2_with_liquid.mp4')
##    cv.waitKey(1000)
##    print ("Wait for the header")

##pos_frame = cap.get(cv.CAP_PROP_POS_FRAMES)
##while True:
##    flag, frame = cap.read()
##    if flag:
##        # The frame is ready and already captured
####        cv.imshow('video', frame)
##        pos_frame = cap.get(cv.CAP_PROP_POS_FRAMES)
##        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
##        blur = cv.GaussianBlur(gray,(5,5),0)
##        ret, thresh = cv.threshold(blur, 150, 255, cv.THRESH_BINARY)
##        #get the contour of the pipette
##        template_contour,hierachy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
##        for c in template_contour:
##            cv.drawContours(frame, [c], -1, (0,255,0), 3)
##
##        # Display the resulting frame
##        cv.imshow('frame',frame)
##        if cv.waitKey(1) & 0xFF == ord('q'):
##            break
##
##    else:
##        # The next frame is not ready, so we try to read it again
##        cap.set(cv.CAP_PROP_POS_FRAMES, pos_frame-1)
##        print ("frame is not ready")
##        # It is better to wait for a while for the next frame to be ready
##        cv.waitKey(1000)
##
##    if cv.waitKey(10) == 27:
##        break
##    if cap.get(cv.CAP_PROP_POS_FRAMES) == cap.get(cv.CAP_PROP_FRAME_COUNT):
##        # If the number of captured frames is equal to the total number of frames,
##        # we stop
##        break
