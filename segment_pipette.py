import cv2 as cv
import numpy as np
import measure_pippette as mp
import grabcut
import os

mask = cv.imread("mask_copy.png", 0)
cv.imshow('mask', mask)
cv.waitKey(0)
cv.destroyAllWindows()

#Find all files in directory ../sample_images/ and store in an array
def find_files(directory):
    files = []
    for file in os.listdir(directory):
        if file.endswith(".jpg") or file.endswith(".png"):
            files.append(file)
    return files

def grabcut_pipette(image):
    """
    Given an image, find the pipette using grabcut
    """
    #Read image
    img = cv.imread(image)
    #Find the pipette using grabcut
    ROI = grabcut.find_pipette(img,mask)
    return ROI

def main():
    #Path of input input pictures
    path = "D:\Downloads\School\Y4S1\FYP\sample_images\\top_visible"
    os.chdir(path)
    images = find_files(path)
    for image_path in images:
        print(image_path)
        img = grabcut_pipette(image_path)
        cv.imshow("img", img)
        cv.waitKey(0)
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()