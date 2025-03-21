import keras
from keras import layers
from tensorflow import data as tf_data
from keras.utils import to_categorical


from .constants import (
    data_folder_path,
)

# use this to increase reduce size of dataset for more speed
# data_augmentation_layers = []

# use this to increase performance. (but slower)
data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]


def _data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images


def create_datasets():
    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        data_folder_path,
        validation_split=0.2,
        subset="both",
        seed=1337,
        image_size=(244, 244),
        batch_size=16,
    )

    train_ds = train_ds.map(
        lambda img, label: (
            _data_augmentation(img),
            to_categorical(label, num_classes=3),
        ),
        num_parallel_calls=tf_data.AUTOTUNE,
    )
    val_ds = val_ds.map(
        lambda img, label: (
            _data_augmentation(img),
            to_categorical(label, num_classes=3),
        ),
        num_parallel_calls=tf_data.AUTOTUNE,
    )

    # Prefetching samples in GPU memory helps maximize GPU utilization.
    train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf_data.AUTOTUNE)

    return (train_ds, val_ds)
