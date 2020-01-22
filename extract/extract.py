#
# extract.py
#
# Given regions of interest identified by analyze.py, extract
# those regions from a given image.
#
# Author:  Christopher Minson  www.christopherminson.com
#
#
import os
import sys
import time
import numpy as np
import cv2

# return origin coordinates and dimensions of image (these are encoded in image name)
def getRegionAttributes(image_region):

    image_region = os.path.basename(image_region)
    (x, y, w, h)  = image_region.split('.')[3].split('_');
    return (int(x), int(y), int(w), int(h))


# extract specified region within image
def extract_region(image, region):

    extracted_image = np.copy(region)
    rows = region.shape[0]
    cols = region.shape[1]
    for row in range(rows):
        for col in range(cols):
            if region[row, col][0] == 255:
                extracted_image[row, col] = image[row, col]
            else:
                extracted_image[row, col] = (255, 255, 255)

    return extracted_image


if __name__ == '__main__':

    if len(sys.argv) != 4:
        print('usage: python extract.py path_input_image path_region path_output_image')
        exit()

    PATH_INPUT_IMAGE = sys.argv[1] 
    PATH_REGION = sys.argv[2]
    PATH_OUTPUT_IMAGE = sys.argv[3]
    time_start = time.time()


    image_input = cv2.imread(PATH_INPUT_IMAGE)
    region_input = cv2.imread(PATH_REGION)

    # extract region, crop it match the region mask 
    (x, y, w, h) = getRegionAttributes(PATH_REGION)
    extracted_image = extract_region(image_input, region_input)
    cropped_extracted_image = extracted_image[y:y+h, x:x+w]
    result_image = cv2.resize(cropped_extracted_image, (w, h))

    cv2.imwrite(PATH_OUTPUT_IMAGE, result_image);
    print(PATH_OUTPUT_IMAGE)
    time_end = time.time()
    elapsed_time = round((time_end - time_start), 2)
    print(f'elapsed time: {elapsed_time} seconds')




