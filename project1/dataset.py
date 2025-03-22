import keras
from keras import layers
from tensorflow import data as tf_data
from keras.utils import to_categorical


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


def create_datasets(
    training_data_path: str,
    num_classes: int,
    image_size: tuple[int, int],
    color_mode: str = "rgb",
):
    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        training_data_path,
        validation_split=0.2,
        subset="both",
        seed=1337,
        image_size=image_size,
        batch_size=24,
        color_mode=color_mode,
    )

    # print(f"size train_ds: {len(train_ds)}")
    # print(f"train_ds: {train_ds[0]}")
    train_ds = train_ds.map(
        lambda img, label: (
            _data_augmentation(img),
            to_categorical(label, num_classes),
        ),
        num_parallel_calls=tf_data.AUTOTUNE,
    )
    val_ds = val_ds.map(
        lambda img, label: (
            _data_augmentation(img),
            to_categorical(label, num_classes),
        ),
        num_parallel_calls=tf_data.AUTOTUNE,
    )

    # for images, labels in train_ds.take(1):  # Taking 1 batch from the train dataset
    #     print(f"Batch of images shape: {images.shape}")
    #     print(f"Batch of labels shape: {labels.shape}")
    #     print(f"First image (flattened): {images[0].numpy()}")
    #     print(f"First label (one-hot encoded): {labels[0].numpy()}")
    #
    # for images, labels in val_ds.take(1):  # Taking 1 batch from the val dataset
    #     print(f"Batch of images shape: {images.shape}")
    #     print(f"Batch of labels shape: {labels.shape}")
    #     print(f"First image (flattened): {images[0].numpy()}")
    #     print(f"First label (one-hot encoded): {labels[0].numpy()}")
    #
    # Prefetching samples in GPU memory helps maximize GPU utilization.
    train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf_data.AUTOTUNE)

    return (train_ds, val_ds)
