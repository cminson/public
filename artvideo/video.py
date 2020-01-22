#
# Author:  Christopher Minson
# Article: https://www.christopherminson.com/articles/artvideo.html
#
# Interpret a video with given style input 
#
#
import os
import sys
import cv2
import time
import subprocess

import numpy as np
import matplotlib.image
import tensorflow as tf
import tensorflow_hub as hub

# the dnn we use, courtesy of google
HUB_URL = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1'

PATH_VIDEOS = './videos/'
PATH_STYLES = './styles/'
PATH_OUTPUTS = './output/'
PATH_TMP = './tmp/'

PATH_TMP_MP3 = PATH_TMP + 'tmp.mp3'

MAX_IMAGE_DIM = 1024
MAX_FRAMES = 30 * 60
VIDEO_FPS = 30.0

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

#
# for the given input video, generate the video's frames and store in tmp
#
def generate_frames(path_input_video, path_image_style):

    video_capture = cv2.VideoCapture(path_input_video) 
    image_style = load_image(path_image_style);
  
    for count in range(MAX_FRAMES):
  
        success, image = video_capture.read() 
        if success == False: break

        path_frame = PATH_TMP + (str(count).zfill(5)) + '.jpg'
        path_converted_frame = PATH_TMP + 'x' + (str(count).zfill(5)) + '.jpg'

        cv2.imwrite(path_frame, image) 

        image = load_image(path_frame)
        results = hub_module(tf.constant(image), tf.constant(image_style))

        image = tf.squeeze(results[0], axis=0)
        matplotlib.image.imsave(path_converted_frame, image) 
        print(count, path_frame, path_converted_frame)

#
# iterate through the frames in tmp and generate a video 
#
def generate_video(path_output_video):

    image_list = []
    count = 0
    path_converted_frame = PATH_TMP + 'x' + (str(count).zfill(5)) + '.jpg'

    image = cv2.imread(path_converted_frame)
    height, width, layers = image.shape
    size = (width,height)
    print('size: ', size)

    converted_files = [file_name for file_name in os.listdir(PATH_TMP) if 'x' in file_name]
    converted_files.sort()

    for file_name in converted_files:

        path_converted_frame = PATH_TMP + file_name
        image = cv2.imread(path_converted_frame)
        print(path_converted_frame)
        image_list.append(image)

    video_writer = cv2.VideoWriter(path_output_video, cv2.VideoWriter_fourcc(*'mp4v'), VIDEO_FPS, size)
    for i in range(len(image_list)):
        video_writer.write(image_list[i])

    video_writer.release()
    print('video generated: ', path_output_video)

#
# extract the mp3 from the video
# executing a subprocess appears to be the only way to do this, as
# there is no python binding
#
def extract_mp3(path_video):

    print('Extracting audio: ', path_video, PATH_TMP_MP3)
    command = 'ffmpeg -i {0} -f mp3 -ab 192000 -vn {1}'.format(path_video, PATH_TMP_MP3)
    subprocess.call(command, shell=True)

#
# for the given input video, add the mp3 in PATH_TMP_MP3 from the fand generate a video with audio
#
def add_mp3(path_input_video, path_output_video):

    print('Adding audio: ', PATH_TMP_MP3, path_input_video, path_output_video)
    command = 'ffmpeg -i {0} -i {1} -c:v copy -c:a aac -strict experimental {2} '.format(path_input_video, PATH_TMP_MP3, path_output_video)
    subprocess.call(command, shell=True)

#
# clear working tmp directory
#
def clear_tmp():

    print('clearing working tmp directory')
    for file_name in os.listdir(PATH_TMP):
        file_path = os.path.join(PATH_TMP, file_name)

        print('deleting: ', file_path)
        os.unlink(file_path)


#
# MAIN
# Interpret a video based off the given style
#
if __name__ == '__main__':

    if len(sys.argv) != 3:
        print('usage: video.py videeo style')
        exit()

    name_original = sys.argv[1]
    name_style = sys.argv[2]

    time_start = time.time()
    print(f'converting video: {name_original} {name_style}')
    print("TF Version: ", tf.__version__)
    print("TF-Hub version: ", hub.__version__)
    print("Eager mode enabled: ", tf.executing_eagerly())
    print("GPU available: ", tf.test.is_gpu_available())

    # clear working tmp directoruy
    clear_tmp()

    # load and cache the styling dnn
    hub_module = hub.load(HUB_URL)

    # extract audio from the video
    extract_mp3(PATH_VIDEOS + name_original)

    # extract all frames from the video, style them, and put results into tmp
    generate_frames(PATH_VIDEOS + name_original, PATH_STYLES + name_style)

    # regenerate the video from the styled frames
    output_name = os.path.splitext(name_original)[0] + '.' + os.path.splitext(name_style)[0] + '.mp4'
    generate_video(PATH_OUTPUTS + output_name)

    # recombine the extracted audio into the newly-styled video
    input_name = output_name
    output_name = os.path.splitext(name_original)[0] + '.' + os.path.splitext(name_style)[0] + '.audio.mp4'
    add_mp3(PATH_OUTPUTS + input_name, PATH_OUTPUTS + output_name)

    time_end = time.time()
    elapsed_time = int(time_end - time_start)

    minutes = int(elapsed_time / 60)
    seconds = int(elapsed_time % 60)
    print(f'completed: {PATH_OUTPUTS + output_name}')
    print(f'elapsed time: {minutes} minutes {seconds} seconds')
    print(f'elapsed time: {elapsed_time}')


