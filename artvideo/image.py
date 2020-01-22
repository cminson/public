#
# Author:  Christopher Minson
# Article: https://www.christopherminson.com/articles/artvideo.html
#
# Interpret an image with given style input 
#
#
import os
import sys

import numpy as np
import matplotlib.image 
import tensorflow as tf
import tensorflow_hub as hub


PATH_IMAGES = './images/'
PATH_STYLES = './styles/'
PATH_OUTPUTS = './outputs/'
MAX_IMAGE_DIM = 1024


#
# normalize an image for usage by dnn
#
def load_image(path_image):

    image = tf.io.read_file(path_image)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)

    shape = tf.cast(tf.shape(image)[:-1], tf.float32)
    long_dim = max(shape)
    scale = MAX_IMAGE_DIM / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    image = tf.image.resize(image, new_shape)
    image = image[tf.newaxis, :]
    return image


if __name__ == '__main__':

    count = len(sys.argv)
    if count != 3:
        print('usage: python image.py image style')
        exit()

    name_original = sys.argv[1]
    name_style = sys.argv[2]
    print('converting: %s %s' % (name_original, name_style))


    print("TF Version: ", tf.__version__)
    print("TF-Hub version: ", hub.__version__)
    print("Eager mode enabled: ", tf.executing_eagerly())
    print("GPU available: ", tf.test.is_gpu_available())

    hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1'
    hub_module = hub.load(hub_handle)

    image_original = load_image(PATH_IMAGES + name_original);
    image_style = load_image(PATH_STYLES + name_style);

    results = hub_module(tf.constant(image_original), tf.constant(image_style))

    image = tf.squeeze(results[0], axis=0)

    output_name = os.path.splitext(name_original)[0] + '.' + os.path.splitext(name_style)[0] + '.jpg'
    matplotlib.image.imsave(PATH_OUTPUTS + output_name, image)

    print('result: %s' % (PATH_OUTPUTS + output_name))

