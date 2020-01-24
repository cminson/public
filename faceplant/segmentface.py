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
import skimage.io
import numpy as np
from PIL import Image

# quiet a ton of warnings about future deprecations
import warnings
warnings.filterwarnings("ignore")

# where all images and masks are stored
PATH_CONVERSION_DIR ='./'

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print('usage: python analayze.py path_input_image path_conversion_dir', end='')
        exit()

    PATH_INPUT_IMAGE = sys.argv[1]
    PATH_CONVERSION_DIR = sys.argv[2]
    file_name =  os.path.basename(PATH_INPUT_IMAGE).split('.')[0]

    image_input = Image.open(PATH_INPUT_IMAGE)
    input_width, input_height = image_input.size
    image_input = skimage.io.imread(PATH_INPUT_IMAGE)

    import cv2
    cascade = '/usr/local/lib/python3.6/dist-packages/cv2/data'
    cascPath = '/usr/local/lib/python3.6/dist-packages/cv2/data/haarcascade_frontalface_default.xml'
    eyePath = '/usr/local/lib/python3.6/dist-packages/cv2/data/haarcascade_eye.xml'
    smilePath = '/usr/local/lib/python3.6/dist-packages/cv2/data/haarcascade_smile.xml'
    faceCascade = cv2.CascadeClassifier(cascPath)
    eyeCascade = cv2.CascadeClassifier(eyePath)
    smileCascade = cv2.CascadeClassifier(smilePath)


    faces = faceCascade.detectMultiScale(
    image_input,
    scaleFactor=1.1,
    minNeighbors=5,
    flags=cv2.CASCADE_SCALE_IMAGE
    )

    object_name = 'face'
    score = 100
    print(f'Face Count: {len(faces)}')
    for (x, y, w, h) in faces:
        score -= 1
        outputFilePath = f'{PATH_CONVERSION_DIR}/m{file_name}.{score}.{object_name}.{x}_{y}_{w}_{h}.png'
        bitmap = np.zeros([input_height, input_width], dtype = np.uint8)
        bitmap[y:y+h, x:x+w] = 255

        face_bitmap = bitmap
        face_bitmap[face_bitmap > 0] = 255
        image_face = Image.fromarray(face_bitmap, 'L')
        image_face.save(outputFilePath, 'PNG')
        print(outputFilePath)







