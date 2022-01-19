import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import argparse

#Download the videos and create a directory called sample_videos
##cap = cv.VideoCapture('Video-2020_1008_185556-1.2_with_liquid.mp4')

#read the pippette mask
template = cv.imread("mask1.png")
template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

#define the input image for comparison
image = "Image_1.png"
img = cv.imread(image)

#TODO Get shape matching to work ie focus the computation on the pipette area
def normalize_filled(img):
    cnt, heir= cv.findContours(img.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # fill shape
    cv.fillPoly(img, pts=cnt, color=(255,255,255))
    bounding_rect = cv.boundingRect(cnt[0])
    img_cropped_bounding_rect = img[bounding_rect[1]:bounding_rect[1] + bounding_rect[3], bounding_rect[0]:bounding_rect[0] + bounding_rect[2]]
    # resize all to same size
    img_resized = cv.resize(img_cropped_bounding_rect, (300, 300))
    return img_resized

def measure_liquid_level():
        frame = cv.imread(image)
        #kernel for noise reduction
        morph_dilate_kernel_size = (7, 7)
        morph_rect_kernel_size = (6, 1)

##        crop_img = frame[585:700, 359:370]
##        cv.imshow("cropped", crop_img)
##        cv.waitKey(0)

        
        return_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
##        cv.imshow("Grayscale",return_image)
        
        # apply histogram equalization
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return_image = clahe.apply(return_image)

        ##Adaptive Threshold
        thresh = cv.adaptiveThreshold(return_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 21, 4)

        #find canny edges
        canny_threshold_1, canny_threshold_2 = find_parameters_for_canny_edge(return_image)
        return_image = cv.Canny(return_image, canny_threshold_1, canny_threshold_2)

        #highlight foreground
##        masked = cv.bitwise_and(frame, frame, mask=return_image)
##        cv.imshow("Output", masked)
##        cv.waitKey(0)


        contours, hierarchy = cv.findContours(return_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cont = contours[0]

        print(len(contours))
        return_image = cv.drawContours(return_image, cont, -1, (255,0,0), 3)
##        cv.imshow('return_image', return_image)

        #Run image matching here
        imgs = [return_image]
        imgs = [normalize_filled(i) for i in imgs]

        for i in range(1, 2):
            plt.subplot(2, 3, i), plt.imshow(imgs[i - 1], cmap='jet')
            print(cv.matchShapes(template, imgs[i - 1], 1, 0.0))
            cv.imshow('frame', return_image)

def find_parameters_for_canny_edge(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    median = np.median(image)
    # find bounds for Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    return lower, upper

##def click_event(event, x, y, flags, params):
## 
##    # checking for left mouse clicks
##    if event == cv.EVENT_LBUTTONDOWN:
## 
##        # displaying the coordinates
##        # on the Shell
##        print(x, ' ', y)
## 
##        # displaying the coordinates
##        # on the image window
##        font = cv.FONT_HERSHEY_SIMPLEX
##        cv.putText(img, str(x) + ',' +
##                    str(y), (x,y), font,
##                    1, (255, 0, 0), 2)
##        cv.imshow('image', img)


if __name__ == '__main__':
##    cv.imshow('image', img)
##    # setting mouse handler for the image
##    # and calling the click_event() function
##    cv.setMouseCallback('image', click_event)
    measure_liquid_level()

##images = ["Image_1.png","Image_2.png","Image_3.png","Image_4.png","Image_5.png"]
##for i in images:
##    measure_liquid_level(i)
    

