import os
import keras
from keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt

from .dataset import create_datasets
from . import hyperparameters


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


def _get_best_model(
    model_name: str,
    use_cache: bool = False,
):
    best_model_file_path = f"./keras_cache/{model_name}/best_model.keras"
    if use_cache and os.path.exists(best_model_file_path):
        print("Loading best model from cache...")
        return load_model(best_model_file_path)
    else:
        print("No cached model found.")
        return None

    return None
    # tuner.search(train_ds, epochs=3, validation_data=val_ds)
    #
    # best_model = tuner.get_best_models(1)[0]
    # best_model.summary()
    # best_model.save(best_model_file_path)
    # return best_model


def _get_model(
    model_name: str,
    training_folder_path: str,
    image_size: tuple[int, int],
    num_classes: int,
    input_channels: int = 3,
    use_cache: bool = False,
    use_hp_cache: bool = True,
    color_mode: str = "rgb",
):
    if use_cache:
        return _get_best_model(model_name)
    return hyperparameters.train(
        model_name,
        training_folder_path,
        image_size,
        num_classes,
        input_channels,
        color_mode,
        use_hp_cache,
    )


def _train(model, train_ds, val_ds, model_name: str):
    cache_dir = f"./keras_cache/{model_name}"
    log_dir = f"{cache_dir}/logs"
    callbacks = [
        keras.callbacks.ModelCheckpoint(f"{cache_dir}/save_at_{{epoch}}.keras"),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch=50),
    ]
    epochs = 5  # 25
    with tf.profiler.experimental.Profile(log_dir):
        history = model.fit(
            train_ds,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=val_ds,
        )
    return history


def run(
    model_name: str,
    training_folder_path: str,
    image_size: tuple[int, int],
    num_classes: int,
    input_channels: int = 3,
    use_cache: bool = False,
    use_hp_cache: bool = True,
    color_mode: str = "rgb",
):

    print("creating CNN model")
    model = _get_model(
        model_name,
        training_folder_path,
        image_size,
        num_classes,
        input_channels,
        use_cache,
        use_hp_cache,
        color_mode,
    )
    print("Hyperparameters optimized. Training data with best model.")

    if model:
        model.summary()

        (train_ds, val_ds) = create_datasets(
            training_folder_path, num_classes, image_size, color_mode=color_mode
        )

        history = _train(model, train_ds, val_ds, model_name)
        plot_training_history(history)
    print("Model could not be found")
