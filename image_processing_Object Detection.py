import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import argparse
from glob import glob

#Download the videos and create a directory called sample_videos
##cap = cv.VideoCapture('Video-2020_1008_185556-1.2_with_liquid.mp4')

video = '../sample_videos/Video-2020_1008_185556-1.2_with_liquid.mp4'

#get background
def get_background(file_path):
    cap = cv.VideoCapture(file_path)
    # we will randomly select 50 frames for the calculating the median
    frame_indices = cap.get(cv.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=50)
    # we will store the frames in array
    frames = []
    for idx in frame_indices:
        # set the frame id to read that particular frame
        cap.set(cv.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        frames.append(frame)
    # calculate the median
    median_frame = np.median(frames, axis=0).astype(np.uint8)
    return median_frame

def find_parameters_for_canny_edge(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    median = np.median(image)
    # find bounds for Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    return lower, upper

def normalize_filled(img):
    cnt, heir= cv.findContours(img.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # fill shape
    cv.fillPoly(img, pts=cnt, color=(255,255,255))
    bounding_rect = cv.boundingRect(cnt[0])
    img_cropped_bounding_rect = img[bounding_rect[1]:bounding_rect[1] + bounding_rect[3], bounding_rect[0]:bounding_rect[0] + bounding_rect[2]]
    # resize all to same size
    img_resized = cv.resize(img_cropped_bounding_rect, (300, 300))
    return img_resized


#Read the Video
cap = cv.VideoCapture(video)
# get the video frame height and width
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
save_name = 'new_video'
# define codec and create VideoWriter object
out = cv.VideoWriter(
    save_name,
    cv.VideoWriter_fourcc(*'mp4v'), 10, 
    (frame_width, frame_height)
)


def find_pipette():
    # get the background model
    background = get_background(video)
    # convert the background model to grayscale format
    background = cv.cvtColor(background, cv.COLOR_BGR2GRAY)
    frame_count = 0
    consecutive_frame = 4
    print("Printing x, y, w, h:")

    while (cap.isOpened()):
        i=1
        ret, frame = cap.read()
        if ret == True:
            frame_count += 1
            orig_frame = frame.copy()
            # IMPORTANT STEP: convert the frame to grayscale first
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            if frame_count % consecutive_frame == 0 or frame_count == 1:
                frame_diff_list = []
            # find the difference between current frame and base frame
            frame_diff = cv.absdiff(gray, background)
            # thresholding to convert the frame to binary
            ret, thres = cv.threshold(frame_diff, 50, 255, cv.THRESH_BINARY)
            # dilate the frame a bit to get some more white area...
            # ... makes the detection of contours a bit easier
            dilate_frame = cv.dilate(thres, None, iterations=2)
            # append the final result into the `frame_diff_list`
            frame_diff_list.append(dilate_frame)
            # if we have reached `consecutive_frame` number of frames
            if len(frame_diff_list) == consecutive_frame:
                # add all the frames in the `frame_diff_list`
                sum_frames = sum(frame_diff_list)
                # find the contours around the white segmented areas
                contours, hierarchy = cv.findContours(sum_frames, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                # draw the contours, not strictly necessary
                for i, cnt in enumerate(contours):
                    cv.drawContours(frame, contours, i, (0, 0, 255), 3)
                for contour in contours:
                    # continue through the loop if contour area is less than 500...
                    # ... helps in removing noise detection
                    if cv.contourArea(contour) < 400:
                        continue
                    # get the xmin, ymin, width, and height coordinates from the contours
                    (x, y, w, h) = cv.boundingRect(contour)

                    # draw the bounding boxes
                    if x in range (200, 400):
                        if y in range (300,400):
                            if h in range (400,800):
                                if w in range (60,150):
##                                    cv.rectangle(orig_frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
                                    print(x,y,w,h)

            if cv.waitKey(100) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv.destroyAllWindows()



if __name__ == '__main__':
    find_pipette()

