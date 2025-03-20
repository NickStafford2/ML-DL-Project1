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

from project1.split_data import Datasets


def run(datasets: Datasets):
    x_train, y_train, x_test, y_test = None

    # adding a 3rd dimension for channel(grayscale) since keras expects batches of images
    image_batch = image.reshape(1, image.shape[0], image.shape[1], 1)
    image_batch.shape

    num_filters = 8  # number of conv. filters
    conv_filter_size1 = 3  # conv. filter size
    pool_size1 = 2  # pooling filter size

    cnn_model = Sequential()
    cnn_model.add(
        Convolution2D(10, (7, 7), padding="same", input_shape=image_batch.shape[1:])
    )  # random weights initialized
    # input shape would be (224, 224, 1)
    cnn_model.add(MaxPooling2D(pool_size=pool_size1))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(10, activation="softmax"))
    cnn_model.summary()
    # compile the model
    cnn_model.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"])
    # train the model
    cnn_model.fit(
        x_train,
        to_categorical(y_train),
        epochs=3,
        verbose=1,
        validation_data=(x_test, to_categorical(y_test)),
    )
    # task is to create a numpy list of tuples
    # each tuple contains a tensor matrix representing the image and a string label
    # RGB images
    import numpy as np
    import cv2
    from google.colab.patches import cv2_imshow

    train_image_names = ["class1_10_4.jpg", "class2_3_5.jpg", "class3_12_2.jpg"]
    valid_image_names = []
    test_image_names = []

    base_image_location = "/content"

    train_data_list = []
    valid_data_list = []
    test_data_list = []

    for image_name in train_image_names:
        PATH = f"{base_image_location}/{image_name}"
        image = cv2.imread(PATH)

        image_tensor = np.asarray(image)
        image_tensor = image_tensor / 255

        class_name = image_name[:6]  # obtain class name as substring
        curr_tuple = (image_tensor, class_name)
        train_data_list.append(curr_tuple)

    train_data = np.array(train_data_list, dtype=object)
    valid_data = ...
    test_data = ...
