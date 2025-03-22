import os
import pickle
from keras.models import load_model
import keras_tuner
import tensorflow as tf

from .model import create_model
from .dataset import create_datasets


class MyHyperModel(keras_tuner.HyperModel):
    def __init__(self, model_name, input_channels, num_classes, image_size) -> None:
        self._model_name = model_name
        self._input_channels = input_channels
        self._num_classes = num_classes
        self._image_size = image_size
        super().__init__()

    def build(self, hp):
        # Tune hyperparameters inside `create_model` using Keras Tuner's `HyperParameters` object
        model = create_model(
            hp,
            input_shape=self._image_size + (self._input_channels,),
            num_classes=self._num_classes,
            model_name=self._model_name,
        )
        return model


# Function to save the best hyperparameters to cache
def save_best_hyperparameters(best_hyperparameters, cache_dir: str):
    best_hyperparameters_path = f"{cache_dir}/best_hyperparameters.pkl"
    with open(best_hyperparameters_path, "wb") as f:
        pickle.dump(best_hyperparameters, f)
    print("Best hyperparameters saved to cache.")


# Function to load the best hyperparameters from cache
def load_best_hyperparameters(cache_dir: str):
    best_hyperparameters_path = f"{cache_dir}/best_hyperparameters.pkl"
    if os.path.exists(best_hyperparameters_path):
        with open(best_hyperparameters_path, "rb") as f:
            best_hyperparameters = pickle.load(f)
        print("Loaded best hyperparameters from cache.")
        return best_hyperparameters
    else:
        print("No cached hyperparameters found.")
        return None


def train(
    model_name: str,
    training_folder_path: str,
    image_size: tuple[int, int],
    num_classes: int,
    input_channels: int = 3,
    color_mode: str = "rgb",
    use_cache: bool = True,
):
    cache_dir = f"./tuner_results/{model_name}"

    tuner = keras_tuner.RandomSearch(
        hypermodel=MyHyperModel(model_name, input_channels, num_classes, image_size),
        objective="val_accuracy",
        max_trials=3,  # 10,
        executions_per_trial=3,
        overwrite=True,
        directory="./tuner_results",
        project_name=model_name,
    )

    if use_cache:
        best_hyperparameters = load_best_hyperparameters(cache_dir)
        if best_hyperparameters is not None:
            if tuner.hypermodel:
                return tuner.hypermodel.build(best_hyperparameters)

    print("No cached hyperparameters. Generating new ones \n")
    (train_ds, val_ds) = create_datasets(
        training_folder_path, num_classes, image_size, color_mode=color_mode
    )
    tuner.search(train_ds, epochs=3, validation_data=val_ds)
    best_hyperparameters = tuner.get_best_hyperparameters()[0]
    save_best_hyperparameters(best_hyperparameters, cache_dir)
    if tuner.hypermodel:
        return tuner.hypermodel.build(best_hyperparameters)

    raise Exception("hypermodel does not exist")
    # best_model = tuner.get_best_models(1)[0]
    # best_model_file_path = f"./keras_cache/{model_name}/best_model.keras"
    # best_model.save(best_model_file_path)

    # return best_model
