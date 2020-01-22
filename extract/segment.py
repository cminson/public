#
# segment.py
#
# Identify regions of interest (image segments) for a given image.
# Store these regions as black/white masks in the PATH_CONVERSION_DIR directory
#
# Author:  Christopher Minson  www.christopherminson.com
#
#
import os
import sys
import time
import skimage
import numpy as np
from PIL import Image

# quiet a ton of warnings about future deprecations
import warnings
warnings.filterwarnings("ignore")

# where the DNN model and COCO weights are stored
MODEL_PATH = '/home/ubuntu/projects/MODELS/mask_rcnn/mask_rcnn_coco.hy'
MODEL_WEIGHTS_PATH = '/home/ubuntu/projects/MODELS/mask_rcnn/mask_rcnn_coco.h5'
MODEL_DIR = '/home/ubuntu/projects/MODELS/Mask_RCNN'
COCO_DIR = '/home/ubuntu/projects/MODELS/Mask_RCNN/samples/coco'
sys.path.append(MODEL_DIR)
sys.path.append(COCO_DIR)
from mrcnn import utils
import mrcnn.model as modellib
import coco

# where all images and masks are stored
PATH_CONVERSION_DIR ='./'

# All the image categories  we can identify
COCO_CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# generate the background region mask
# this is the negative of the sum of all other region masks
def computeBackgroundRegion(file_name,regionFileList):

    if len(regionFileList) == 0: return

    image_region = Image.open(regionFileList[0])
    background_bitmap = np.array(image_region)
    width, height = image_region.size
    region_label = 'background'
    score = '99'
    x = y = 0

    # generate the or union of all region bitmaps
    for regionFile in regionFileList:
        image_region = Image.open(regionFile)
        bitmap = np.array(image_region)
        background_bitmap = np.logical_or(background_bitmap, bitmap).astype(np.uint8)

    # our background is the negative of that
    background_bitmap[background_bitmap == 1] = 255
    background_inverted_bitmap = np.invert(background_bitmap, dtype=np.uint8)

    mask_file_name =  f'{PATH_CONVERSION_DIR}/m{file_name}.{score}.{region_label}.{x}_{y}_{width}_{height}.png'
    image_background = Image.fromarray(background_inverted_bitmap, 'L')
    image_background.save(mask_file_name, 'PNG')
    print(mask_file_name)


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print('usage: python analayze.py path_input_image path_conversion_dir', end='')
        exit()

    PATH_INPUT_IMAGE = sys.argv[1]
    PATH_CONVERSION_DIR = sys.argv[2]
    time_start = time.time()

    file_name =  os.path.basename(PATH_INPUT_IMAGE).split('.')[0]

    # create an inference instance of DNN config object
    class InferenceConfig(coco.CocoConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1  
    config = InferenceConfig()

    # Create model object in inference mode, fold in weights
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_PATH, config=config)
    model.load_weights(MODEL_WEIGHTS_PATH, by_name=True)

    # run the model on the input image
    image_input = skimage.io.imread(PATH_INPUT_IMAGE)
    results = model.detect([image_input], verbose=1)

    # unpack all results  
    result = results[0]
    class_ids = result['class_ids']
    masks = result['masks'].astype(np.uint8)
    scores = result['scores']
    rois = result['rois']

    # for each region identified, get the score, label, dimensions and mask
    # modify the mask so that all active pixels are white, with background black
    # save the mask off bitmap `to the conversion directory
    regionFileList = []
    for index, class_id in enumerate(class_ids):

        region_label = COCO_CLASS_NAMES[class_id].replace(' ', '_')
        score = int(scores[index] * 100)
        (y1, x1, y2, x2) = rois[index] # bounding box for the max
        width = x2 - x1
        height = y2 - y1

        bitmap = masks[:,:,index]   # slice off the bitmap for this object
        bitmap[bitmap > 0] = 255    # make positive mask pixels white

        path_output_image = f'{PATH_CONVERSION_DIR}/m{file_name}.{score}.{region_label}.{x1}_{y1}_{width}_{height}.png'
        image_region = Image.fromarray(bitmap, 'L')
        image_region.save(path_output_image, 'PNG')
        print(path_output_image)

        regionFileList.append(path_output_image)

    # lastly, compute the background region (negative of all other regions)
    computeBackgroundRegion(file_name, regionFileList)
    time_end = time.time()
    elapsed_time = round((time_end - time_start), 2)
    print(f'elapsed time: {elapsed_time} seconds')




