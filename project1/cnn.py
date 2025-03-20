import shutil
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy import asarray
import tensorflow as tf
from keras.utils import to_categorical
from keras.layers import Convolution2D, MaxPooling2D, Activation, Flatten, Dense
from keras.models import Sequential


def run():
    # read a random image
    image_path = "/content/drive/MyDrive/MLDL Spring 2025/MLDL_Data_Face-1/MLDL_Data_Face/Class 3/15/10.jpg"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    plt.imshow(image, cmap="gray")
    plt.show()

    # image.shape
    #
    # string = 'Class 3/15'
    # string[8:]
    # # not part of standard preprocessing
    # # copying images for following empty directores in drive
    #
    # empty_dirs = ['Class 3/16', 'Class 3/3', 'Class 3/20', 'Class 3/1', 'Class 3/13', 'Class 3/5', 'Class 3/19', 'Class 3/17', 'Class 3/8']
    #
    # base_photos_retrieval_path = "/content/drive/MyDrive/MLDL Spring 2025"
    # base_photos_copy_path = "/content/drive/MyDrive/MLDL Spring 2025/MLDL_Data_Face-1/MLDL_Data_Face"
    #
    # for empty_dir in empty_dirs:
    # files = os.listdir(f"{base_photos_retrieval_path}/{empty_dir[8:]}")
    # shutil.copytree(f"{base_photos_retrieval_path}/{empty_dir[8:]}", f"{base_photos_copy_path}/{empty_dir}")
    # print(f"done copying {empty_dir}")
    #
    # x_train, y_train, x_test, y_test = None
    #
    # # adding a 3rd dimension for channel(grayscale) since keras expects batches of images
    # image_batch = image.reshape(1, image.shape[0], image.shape[1], 1)
    # image_batch.shape
    #
    # num_filters = 8 # number of conv. filters
    # conv_filter_size1 = 3 # conv. filter size
    # pool_size1 = 2 # pooling filter size
    #
    # cnn_model = Sequential()
    # cnn_model.add(Convolution2D(10, (7,7), padding='same', input_shape=image_batch.shape[1:])) # random weights initialized
    # # input shape would be (224, 224, 1)
    # cnn_model.add(MaxPooling2D(pool_size=pool_size1))
    # cnn_model.add(Flatten())
    # cnn_model.add(Dense(10, activation = 'softmax'))
    # cnn_model.summary()
