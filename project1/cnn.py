import keras
from keras.models import load_model
import keras_tuner
import tensorflow as tf
import matplotlib.pyplot as plt

from .dataset import create_datasets
from .model import create_model


def plot_training_history(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()


def _get_best_model(tuner, train_ds, val_ds, use_cache: bool):
    cache_file_path = "keras_cache/best_model.keras"
    if use_cache:
        return load_model(cache_file_path)

    tuner.search(train_ds, epochs=3, validation_data=val_ds)

    best_model = tuner.get_best_models(1)[0]
    best_model.summary()
    best_model.save(cache_file_path)
    return best_model


def _train(model, train_ds, val_ds):
    callbacks = [
        keras.callbacks.ModelCheckpoint("keras_cache/save_at_{epoch}.keras"),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        keras.callbacks.TensorBoard(log_dir="./logs", profile_batch=50),
    ]
    epochs = 20  # 25
    with tf.profiler.experimental.Profile("./logs"):
        history = model.fit(
            train_ds,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=val_ds,
        )
    return history


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


def run(
    model_name: str,
    training_folder_path: str,
    image_size: tuple[int, int],
    num_classes: int,
    input_channels: int = 3,
    use_cache: bool = False,
):

    tuner = keras_tuner.RandomSearch(
        hypermodel=MyHyperModel(model_name, input_channels, num_classes, image_size),
        objective="val_accuracy",
        max_trials=10,
        executions_per_trial=3,
        overwrite=True,
        directory="tuner_results",
        project_name=model_name,
    )
    (train_ds, val_ds) = create_datasets(training_folder_path, num_classes, image_size)

    best_model = _get_best_model(tuner, train_ds, val_ds, use_cache)

    print("Hyperparameters optimized. Training data with best model.")
    history = _train(best_model, train_ds, val_ds)
    plot_training_history(history)
