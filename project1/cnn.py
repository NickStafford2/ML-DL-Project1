import keras
from keras import layers
from tensorflow import data as tf_data
from typing import Any
from keras.utils import to_categorical

from project1.split_data import Datasets
from .constants import (
    data_folder_path,
)


data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]
# use this to increase reduce size of dataset for more speed
# data_augmentation_layers = []


def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images


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
    keras.utils.plot_model(model, show_shapes=True, to_file="./docs/model.png")

    model.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
    )

    # callbacks = [
    #     keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
    # ]
    # epochs = 25
    # model.fit(
    #     train_ds,
    #     epochs=epochs,
    #     callbacks=callbacks,
    #     validation_data=val_ds,
    # )
