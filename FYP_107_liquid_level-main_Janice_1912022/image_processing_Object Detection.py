import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import argparse
from glob import glob

#Download the videos and create a directory called sample_videos
##cap = cv.VideoCapture('Video-2020_1008_185556-1.2_with_liquid.mp4')

#read the pippette mask
template = cv.imread("mask1.png")
template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

video = 'Video-2020_1008_185556-1.2_with_liquid.mp4'

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

# get the background model
background = get_background(video)
# convert the background model to grayscale format
background = cv.cvtColor(background, cv.COLOR_BGR2GRAY)
frame_count = 0
consecutive_frame = 4


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

                # draw the bounding boxes + cropped images
                if x in range (200, 400):
                    if y in range (300,400):
                        if h in range (400,800):
                            if w in range (60,150):
                                cv.rectangle(orig_frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
                                cropped = orig_frame.copy()
                                cropped = cropped[y:y+h, x:x+w]
                                cv.imwrite("pipette"+str(i)+".png",cropped)
                                i=i+1

                                #compare cropped image to wanted image
                                return_image = cropped

                                #kernel for noise reduction
                                morph_dilate_kernel_size = (7, 7)
                                morph_rect_kernel_size = (6, 1)
                
                                return_image = cv.cvtColor(return_image, cv.COLOR_BGR2GRAY)
                
                                # apply histogram equalization
                                clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                                return_image = clahe.apply(return_image)

                                #return_image = cv.adaptiveThreshold(return_image,255,cv.ADAPTIVE_THRESH_MEAN_C,\
                                #cv.THRESH_BINARY,21,10)
                                ##  ret,return_image = cv.threshold(return_image,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
                                #return_image = cv.adaptiveThreshold(return_image,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\cv.THRESH_BINARY,11,2)

                                ##Adaptive Threshold
                                thresh = cv.adaptiveThreshold(return_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 21, 4)

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
                                cv.imshow('Detected Objects', return_image)
                                out.write(return_image)

                                imgs = [template,return_image]
                                imgs = [normalize_filled(i) for i in imgs]

                                for i in range(2, 3):
                                    plt.subplot(2, 3, i), plt.imshow(imgs[i - 1], cmap='gray')
                                    print(cv.matchShapes(imgs[0], imgs[i - 1], 1, 0.0))
        ##                            cv.imshow('frame', return_image)


            if cv.waitKey(100) & 0xFF == ord('q'):
                break
    else:
        break
cap.release()
cv.destroyAllWindows()

#Read Multiple Images
##frame = cv.VideoCapture("C:\Users\Yukan\Desktop\FYP\FYP_107_liquid_level-main\FYP_107_liquid_level-main\pipette%2d.png")

###set a mask
##    mask = np.zeros((frame.shape[0], frame.shape[1]))
##    cv.fillConvexPoly(mask, orig_frame, 1)
##    mask = mask.astype(np.bool)
##    out = np.zeros_like(frame)
##    out[mask] = frame[mask]

#TODO Get shape matching to work ie focus the computation on the pipette area

##
##def measure_liquid_level():
##
####    frame = cv.VideoCapture("C:/Users/Yukan/Desktop/FYP/FYP_107_liquid_level-main/FYP_107_liquid_level-main/pipette%2d.png")
####    cv.imshow("Frame", frame)
##
##    for fn in glob('*.png'):
##        frame = cv.imread(fn)
##        cv.imshow("Frame", frame)
##        ret, frame = cap.read()
##        # if frame is read correctly ret is True
##        if not ret:
##            print("Can't receive frame (stream end?). Exiting ...")
##            break
        #kernel for noise reduction
##        morph_dilate_kernel_size = (7, 7)
##        morph_rect_kernel_size = (6, 1)
##        
##        return_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
##        
##        # apply histogram equalization
##        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
##        return_image = clahe.apply(return_image)
##
##        #return_image = cv.adaptiveThreshold(return_image,255,cv.ADAPTIVE_THRESH_MEAN_C,\
##            #cv.THRESH_BINARY,21,10)
####        ret,return_image = cv.threshold(return_image,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
##        #return_image = cv.adaptiveThreshold(return_image,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\cv.THRESH_BINARY,11,2)
##
##                ##Adaptive Threshold
##        thresh = cv.adaptiveThreshold(return_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 21, 4)
##
##        #find canny edges
##        canny_threshold_1, canny_threshold_2 = find_parameters_for_canny_edge(return_image)
##        return_image = cv.Canny(return_image, canny_threshold_1, canny_threshold_2)
##
####        #highlight foreground
####        masked = cv.bitwise_and(frame, frame, mask=return_image)
####        cv.imshow("Output", masked)
####        cv.waitKey(0)
##
##
##        contours, hierarchy = cv.findContours(return_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
##
##        # return_image = cv.morphologyEx(return_image, cv.MORPH_DILATE, morph_dilate_kernel_size)
##        # # create a horizontal structural element;
##        # horizontal_structure = cv.getStructuringElement(cv.MORPH_RECT, morph_rect_kernel_size)
##        # # to the edges, apply morphological opening operation to remove vertical lines from the contour image
##        # return_image = cv.morphologyEx(return_image, cv.MORPH_OPEN, horizontal_structure)
##        cont = contours[0]
##        # ret = 1
##        # for shape in contours:
##        #     temp = cv.matchShapes(template_contour[0],shape,1,0.0)
##        #     if temp< ret:
##        #         ret = temp
##        #         cont = shape
##        print(len(contours))
##        return_image = cv.drawContours(return_image, cont, -1, (255,0,0), 3)
##
##        imgs = [template,return_image]
##        imgs = [normalize_filled(i) for i in imgs]
##
##        for i in range(2, 3):
##            plt.subplot(2, 3, i), plt.imshow(imgs[i - 1], cmap='gray')
##            print(cv.matchShapes(imgs[0], imgs[i - 1], 1, 0.0))
##
##        cv.imshow('frame', return_image)
##        if cv.waitKey(1) == ord('q'):
##            break
##    cap.release()
##    cv.destroyAllWindows()
##
##
##
##if __name__ == '__main__':
##    measure_liquid_level()

