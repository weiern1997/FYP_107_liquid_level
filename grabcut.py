import numpy as np
import cv2


def find_pipette(img, mask):
    height, width = img.shape[:2]
    mask = cv2.resize(mask, (width,height))

    #set mask to foreground and background
    mask[mask>0] = cv2.GC_FGD
    mask[mask==0] = cv2.GC_BGD


    # allocate memory for two arrays that the GrabCut algorithm internally
    # uses when segmenting the foreground from the background
    # Taken from docs just do it
    fgModel = np.zeros((1, 65), dtype="float")
    bgModel = np.zeros((1, 65), dtype="float")

    mask, bgModel, fgModel = cv2.grabCut(img, mask, None, bgModel, fgModel, iterCount=5, mode=cv2.GC_INIT_WITH_MASK)

    outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD),
        0, 1)
    outputMask = (outputMask * 255).astype("uint8")

    return outputMask