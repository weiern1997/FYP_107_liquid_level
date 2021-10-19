import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import argparse

#Download the videos and create a directory called sample_videos
cap = cv.VideoCapture('Video-2020_1008_185556-1.2_with_liquid.mp4')

#read the pippette mask
template = cv.imread("mask.png")
template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

#TODO Get clean threshhold and contour of pipette for contourmatching


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

imgs = [template]
imgs = [normalize_filled(i) for i in imgs]

for i in range(1, 2):
    plt.subplot(2, 3, i), plt.imshow(imgs[i - 1], cmap='gray')
    print(cv.matchShapes(imgs[0], imgs[i - 1], 1, 0.0))

def measure_liquid_level():

    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        #kernel for noise reduction
        morph_dilate_kernel_size = (7, 7)
        morph_rect_kernel_size = (6, 1)
        
        return_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # apply histogram equalization
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return_image = clahe.apply(return_image)

        #return_image = cv.adaptiveThreshold(return_image,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            #cv.THRESH_BINARY,21,10)
##        ret,return_image = cv.threshold(return_image,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        #return_image = cv.adaptiveThreshold(return_image,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\cv.THRESH_BINARY,11,2)

                ##Adaptive Threshold
        thresh = cv.adaptiveThreshold(return_image, 255,
	cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 21, 4)

        #find canny edges
        canny_threshold_1, canny_threshold_2 = find_parameters_for_canny_edge(return_image)
        return_image = cv.Canny(return_image, canny_threshold_1, canny_threshold_2)

##        #highlight foreground
##        masked = cv.bitwise_and(frame, frame, mask=return_image)
##        cv.imshow("Output", masked)
##        cv.waitKey(0)


        contours, hierarchy = cv.findContours(return_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # return_image = cv.morphologyEx(return_image, cv.MORPH_DILATE, morph_dilate_kernel_size)
        # # create a horizontal structural element;
        # horizontal_structure = cv.getStructuringElement(cv.MORPH_RECT, morph_rect_kernel_size)
        # # to the edges, apply morphological opening operation to remove vertical lines from the contour image
        # return_image = cv.morphologyEx(return_image, cv.MORPH_OPEN, horizontal_structure)
        cont = contours[0]
        # ret = 1
        # for shape in contours:
        #     temp = cv.matchShapes(template_contour[0],shape,1,0.0)
        #     if temp< ret:
        #         ret = temp
        #         cont = shape
        print(len(contours))
        return_image = cv.drawContours(return_image, cont, -1, (255,0,0), 3)

        cv.imshow('frame', return_image)
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

def find_parameters_for_canny_edge(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    median = np.median(image)
    # find bounds for Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    return lower, upper


if __name__ == '__main__':
    measure_liquid_level()
