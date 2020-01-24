#
# faceplant.py
#
# Transfer faces between two images
#
# Author:  Christopher Minson  www.christopherminson.com
#
#
import os
import sys
import random
import math
import numpy as np
from PIL import Image
import skimage.io
import cv2
from skimage import exposure
from skimage.exposure import match_histograms


# return origin coordinates and dimensions of image (these are encoded in image name)
def getRegionAttributes(image_region):

    image_region = os.path.basename(image_region)
    (x, y, w, h)  = image_region.split('.')[3].split('_');
    return (int(x), int(y), int(w), int(h))


if __name__ == '__main__':

    if len(sys.argv) != 6:
        print('usage: srcImage srcMask dstImage dstMask outputImage ')
        exit()

    PATH_SRC_IMAGE = sys.argv[1] 
    PATH_SRC_REGION = sys.argv[2]
    PATH_DST_IMAGE = sys.argv[3]
    PATH_DST_REGION = sys.argv[4]
    PATH_OUTPUT_IMAGE = sys.argv[5]

    src_image = cv2.imread(PATH_SRC_IMAGE)
    src_region = cv2.imread(PATH_SRC_REGION)
    dst_image = cv2.imread(PATH_DST_IMAGE)
    dst_region = cv2.imread(PATH_DST_REGION)

    #crop out the source image and region
    (x, y, w, h) = getRegionAttributes(PATH_SRC_REGION)
    cropped_src_image = src_image[y:y+h, x:x+w]
    cropped_src_region = src_region[y:y+h, x:x+w]

    # resize the cropped source image and region to destination dimensions
    (x, y, w, h)  = getRegionAttributes(PATH_DST_REGION)
    resized_cropped_src_image = cv2.resize(cropped_src_image, (h, w))
    resized_cropped_src_region = cv2.resize(cropped_src_region, (h, w))

    center_x = int(x + (w / 2))
    center_y = int(y + (h / 2))
    dst_center = (center_x, center_y)

    result_image = cv2.seamlessClone(
                        resized_cropped_src_image, 
                        dst_image, 
                        resized_cropped_src_region, 
                        dst_center, 
                        cv2.NORMAL_CLONE)

    cv2.imwrite(PATH_OUTPUT_IMAGE, result_image);




