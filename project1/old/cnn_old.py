# https://keras.io/examples/vision/image_classification_from_scratch/
# import shutil
# import os
import keras
from keras import layers
from tensorflow import data as tf_data
from typing import Any

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from numpy import asarray
# import tensorflow as tf
from keras.utils import to_categorical
from keras.layers import Convolution2D, MaxPooling2D, Activation, Flatten, Dense

# from keras.models import Sequential

from project1.split_data import Datasets
from .constants import (
    data_folder_path,
)


def get_shape(dataset: list[Any] | Any):
    for images, labels in dataset.take(1):
        return images.shape

    # adding a 3rd dimension for channel(grayscale) since keras expects batches of images
    # image = datasets.training[0][0]
    # print(image.shape)
    # return image.shape
    # image_batch = image.reshape(1, image.shape[0], image.shape[1], 1)
    # image_batch.shape
    # return image_batch.shape[1:]


def format_data(datasets: Datasets):

    x_train, y_train, x_test, y_test = None
    return (x_train, y_train, x_test, y_test)


data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]
# data_augmentation_layers = []


def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images


def preprocess(dataset: list[Any] | Any):
    shape = get_shape(dataset)
    inputs = keras.Input(shape=shape)
    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255)(x)
    return x


def configure_for_performance(train_ds, val_ds):
    # Apply `data_augmentation` to the training images.
    train_ds = train_ds.map(
        lambda img, label: (data_augmentation(img), label),
        num_parallel_calls=tf_data.AUTOTUNE,
    )
    # Prefetching samples in GPU memory helps maximize GPU utilization.
    train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf_data.AUTOTUNE)


def run_old(train_ds, val_ds):
    # x = preprocess(dataset)

    train_ds = train_ds.map(
        lambda img, label: (data_augmentation(img), label),
        num_parallel_calls=tf_data.AUTOTUNE,
    )
    # Prefetching samples in GPU memory helps maximize GPU utilization.
    train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf_data.AUTOTUNE)

    x_train, y_train, x_test, y_test = format_data(datasets)

    num_filters = 8  # number of conv. filters
    conv_filter_size1 = 3  # conv. filter size
    pool_size1 = 2  # pooling filter size

    cnn_model = Sequential()
    cnn_model.add(
        Convolution2D(10, (7, 7), padding="same", input_shape=get_shape(datasets))
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


def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        units = 1
    else:
        units = num_classes

    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(units, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def run():
    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        data_folder_path,
        validation_split=0.2,
        subset="both",
        seed=1337,
        image_size=(244, 244),
        batch_size=32,
    )
    train_ds = train_ds.map(
        lambda img, label: (
            data_augmentation(img),
            to_categorical(label, num_classes=3),
        ),
        num_parallel_calls=tf_data.AUTOTUNE,
    )

    val_ds = val_ds.map(
        lambda img, label: (
            data_augmentation(img),
            to_categorical(label, num_classes=3),
        ),
        num_parallel_calls=tf_data.AUTOTUNE,
    )
    # Prefetching samples in GPU memory helps maximize GPU utilization.
    train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf_data.AUTOTUNE)
    image_size = (244, 244)
    model = make_model(input_shape=image_size + (3,), num_classes=3)
    keras.utils.plot_model(model, show_shapes=True)

    epochs = 25

    callbacks = [
        keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
    ]
    model.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
    )
    model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_ds,
    )
