import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pyimagesearch.shapedetector import ShapeDetector

img = cv.imread("../sample_images/photo_2021-10-27_00-59-28.jpg")

#Find all files in directory ../sample_images/ and store in an array
def find_files(directory):
    import os
    files = []
    for file in os.listdir(directory):
        if file.endswith(".jpg"):
            files.append(file)
    return files

def find_horizontal_edges(fill):
        """
        Given an image find a closed image that tries to eliminate non-horizontal lines

        :param fill: numpy.ndarray: image to find the contour of
        :return: closed: black and white image, where detected contours are in white
        """
        # get these values from the object's attributes
        morph_dilate_kernel_size = (6, 6)
        morph_rect_kernel_size = (6, 1)

        return_image = fill

        return_image = cv.cvtColor(return_image, cv.COLOR_BGR2GRAY)
        # apply histogram equalization
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return_image = clahe.apply(return_image)
        canny_threshold_1, canny_threshold_2 = find_parameters_for_canny_edge(return_image)
        return_image = cv.Canny(return_image, canny_threshold_1, canny_threshold_2)
        return_image = cv.morphologyEx(return_image, cv.MORPH_DILATE, morph_dilate_kernel_size)

        # create a horizontal structural element;
        horizontal_structure = cv.getStructuringElement(cv.MORPH_RECT, morph_rect_kernel_size)
        # to the edges, apply morphological opening operation to remove vertical lines from the contour image
        return_image = cv.morphologyEx(return_image, cv.MORPH_OPEN, horizontal_structure)

        return return_image

def find_contour(fill):
        """
        Given an image find a closed image that tries to eliminate non-horizontal lines

        :param fill: numpy.ndarray: image to find the contour of
        :return: closed: black and white image, where detected contours are in white
        """
        # get these values from the object's attributes
        morph_rect_kernel_size = np.ones((8,1),np.uint8)

        return_image = fill
        return_image = cv.cvtColor(return_image, cv.COLOR_BGR2GRAY)
        # apply histogram equalization
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return_image = clahe.apply(return_image)

        #remove noise
        kernel = np.ones((5,5),np.uint8)
        return_image = cv.morphologyEx(return_image, cv.MORPH_OPEN, kernel)

        #get canny edges
        canny_threshold_1, canny_threshold_2 = find_parameters_for_canny_edge(return_image)
        return_image = cv.Canny(return_image, canny_threshold_1, canny_threshold_2)



        #merge lines
        kernel = np.ones((3,3),np.uint8)
        return_image = cv.morphologyEx(return_image, cv.MORPH_CLOSE, kernel)

        # create a horizontal structural element;
        kernel = np.ones((1,5),np.uint8)
        return_image = cv.morphologyEx(return_image, cv.MORPH_OPEN, kernel)

        kernel = np.ones((3,5),np.uint8)
        return_image = cv.morphologyEx(return_image, cv.MORPH_CLOSE, kernel)
        # to the edges, apply morphological opening operation to remove vertical lines from the contour image
        #return_image = cv.erode(return_image, morph_rect_kernel_size, iterations = 3)
        #return_image = cv.dilate(return_image,horizontal_structure,iterations=4)
        return return_image

def find_parameters_for_canny_edge(image, sigma=0.33):
        # compute the median of the single channel pixel intensities
        median = np.median(image)
        # find bounds for Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * median))
        upper = int(min(255, (1.0 + sigma) * median))
        return lower, upper

def thresh_image(fill):
    """
    Given an color image, converts to grayscale and applies a threshold to the image with gaussian blurring
    """
    blur = cv.cvtColor(fill, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(blur,(5,5),0)
    img = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv.THRESH_BINARY,11,2)
    return img

def lowest_corner(fill):
    """
    Given an image, find the lowest corner of the pipette
    """
    global img
    #Find bottom of pipette using corner detection
    corners = cv.goodFeaturesToTrack(fill,25,0.01,10)
    corners = np.int0(corners)
    x, y = 0,0
    for i in corners:
        x1,y1 = i.ravel()
        if y1 > y:
                x,y = x1,y1
    cv.circle(img,(x,y),3,255,-1)

def find_liquid_level(edge_image):
    """
    Given image of horizontal edges.
    Split the image into sections 3 pixels high, and find the section with the highest white pixel count and return the height of the section
    """
    #Split the image into sections
    height, width = edge_image.shape
    sections = []
    for i in range(0,height,3):
        sections.append(edge_image[i:i+3,:])
    #Find the section with the highest white pixel count
    max_white = 0
    max_white_index = 0
    for i in range(len(sections)):
        white = np.sum(sections[i])
        if white > max_white:
            max_white = white
            max_white_index = i
    return max_white_index*3

def main():

    th3 = thresh_image(img)

    lowest_corner(th3)


    cont,_ = cv.findContours(th3, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    hor = find_horizontal_edges(img)

    height = find_liquid_level(hor)
    #find dimensions of img
    y,x = th3.shape
    #opencv draw a line on the image img at height=height
    cv.line(img,(0,height),(x,height),(0,0,255),2)

    cv.imshow('img',img)
    cv.imshow('th3',th3)
    cv.imshow('hor',hor)
    k = cv.waitKey(0)

if __name__ == '__main__':
    main()