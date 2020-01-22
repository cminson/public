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


if __name__ == '__main__':

    print("TF Version: ", tf.__version__)
    print("TF-Hub version: ", hub.__version__)
    print("Eager mode enabled: ", tf.executing_eagerly())
    print("GPU available: ", tf.test.is_gpu_available())
    version = tf.__version__
    f = open("venv.txt", "w+")
    f.write(version)
    f.close()
