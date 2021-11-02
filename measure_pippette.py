import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def find_contour(fill):
        """
        Given an image find a closed image that tries to eliminate non-horizontal lines

        :param fill: numpy.ndarray: image to find the contour of
        :return: closed: black and white image, where detected contours are in white
        """
        # get these values from the object's attributes
        morph_rect_kernel_size = np.ones((5,1),np.uint8)
        kernel = np.ones((9,9),np.uint8)

        return_image = fill
        return_image = cv.cvtColor(return_image, cv.COLOR_BGR2GRAY)
        #threshold to binary
        return_image = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)
        #remove noise
        return_image = cv.morphologyEx(return_image, cv.MORPH_OPEN, kernel)
        #get canny edges
        canny_threshold_1, canny_threshold_2 = find_parameters_for_canny_edge(return_image)
        return_image = cv.Canny(return_image, canny_threshold_1, canny_threshold_2)

        #merge lines
        return_image = cv.morphologyEx(return_image, cv.MORPH_CLOSE, kernel)

        # create a horizontal structural element;

        # to the edges, apply morphological opening operation to remove vertical lines from the contour image
        #return_image = cv.erode(return_image, morph_rect_kernel_size, iterations = 4)
        return_image = cv.morphologyEx(return_image, cv.MORPH_OPEN, morph_rect_kernel_size)
        #return_image = cv.dilate(return_image,horizontal_structure,iterations=4)
        return return_image

def find_parameters_for_canny_edge(image, sigma=0.33):
        # compute the median of the single channel pixel intensities
        median = np.median(image)
        # find bounds for Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * median))
        upper = int(min(255, (1.0 + sigma) * median))
        return lower, upper



#read the pippette mask
template = cv.imread("mask.png",0)
template = cv.resize(template,None,fx=0.2,fy=0.2,interpolation = cv.INTER_AREA)
t_cont,_ = cv.findContours(template, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


img = cv.imread("..\sample_images\photo_2021-10-27_00-59-26.jpg")

blur = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(blur,(5,5),0)
th3 = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)


#Find bottom of pipette using corner detection
corners = cv.goodFeaturesToTrack(th3,25,0.01,10)
corners = np.int0(corners)
x, y = 0,0
for i in corners:
    x1,y1 = i.ravel()
    if y1 > y:
            x,y = x1,y1
cv.circle(img,(x,y),3,255,-1)

cont,_ = cv.findContours(th3, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
hor = find_contour(img)
y,x = th3.shape
lines = cv.HoughLinesP(hor,1,np.pi/180,int(0.005*x),minLineLength=0.1*x,maxLineGap=0.05*x)
if lines is not None:
    for line in lines:
        x1,y1,x2,y2 = line[0]
        distance = np.sqrt((y2-y1)**2 + (x2-x1)**2)
        angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
        if abs(angle) < 4.5:
            cv.putText(img,str(distance),(x1,y1), cv.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv.LINE_AA)
            cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)


# best = (0,1e9)
#for i,c in enumerate(cont):
    # ret = cv.matchShapes(c,t_cont[0],1,0.0)
    # if ret < best[1]:
    #cv.drawContours(img,cont,i,(255,0,0),3)
    #best = (i,ret)
    #print(best)
    # cv.imshow('img',result)
    # k=cv.waitKey(0)

#cv.drawContours(img,cont,-1,(255,0,0),2)
cv.imshow('img',img)
cv.imshow('th3',th3)
cv.imshow('hor',hor)
k = cv.waitKey(0)

