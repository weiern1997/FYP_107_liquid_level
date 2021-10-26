from skimage.metrics import structural_similarity as compare_ssim
import argparse
import imutils
import cv2 
import numpy as np

cap = cv2.VideoCapture('..\sample_videos\Video-2020_1008_185556-1.2_with_liquid.mp4')

def img_diff():
    old_frame = None
    while cap.isOpened():
        ret, frame = cap.read()
        if old_frame is not None:
            grayA = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grayB = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
            (score, diff) = compare_ssim(grayA, grayB, full=True)
            diff = (diff * 255).astype("uint8")
            thresh = cv2.threshold(diff, 0, 255,
                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            for c in cnts:
                # compute the bounding box of the contour and then draw the
                # bounding box on both input images to represent where the two
                # images differ
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.rectangle(old_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        old_frame = frame
        cv2.imshow('Old', old_frame)
        cv2.imshow('New', frame)
        key = cv2.waitKey(0)
        while key not in [ord('q'), ord('k')]:
            key = cv2.waitKey(0)
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    img_diff()