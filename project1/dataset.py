import keras


from .constants import (
    temp_folder_path,
)


def create_dataset():
    return keras.utils.image_dataset_from_directory(
        temp_folder_path,
        validation_split=0.2,
        subset="both",
        seed=1337,
        image_size=(244, 244),
        batch_size=32,
    )
