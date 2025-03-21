import keras
from keras.models import load_model
import keras_tuner
import matplotlib.pyplot as plt

from .dataset import create_datasets
from .model import create_model


def plot_training_history(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["acc"], label="Training Accuracy")
    plt.plot(history.history["val_acc"], label="Validation Accuracy")
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
    cache_file_path = "keras_cache/best_model.h5"
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
    ]
    epochs = 3  # 25
    history = model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_ds,
    )
    return history


class MyHyperModel(keras_tuner.HyperModel):
    def __init__(self, model_name, channels, num_classes, image_size) -> None:
        self.model_name = model_name
        self.channels = channels
        self.num_classes = num_classes
        self.image_size = image_size

        super().__init__()

    def build(self, hp):

        # Tune hyperparameters inside `create_model` using Keras Tuner's `HyperParameters` object
        model = create_model(
            hp,
            input_shape=self.image_size + (self.channels,),
            num_classes=self.num_classes,
            model_name=self.model_name,
        )
        return model


def run(
    model_name: str = "part2",
    channels: int = 3,
    image_size: tuple[int, int] = (244, 244),
    num_classes: int = 3,
    use_cache: bool = False,
):

    tuner = keras_tuner.RandomSearch(
        hypermodel=MyHyperModel(model_name, channels, num_classes, image_size),
        objective="val_accuracy",
        max_trials=3,
        executions_per_trial=2,
        overwrite=True,
        directory="tuner_results",
        project_name=model_name,
    )
    (train_ds, val_ds) = create_datasets()

    best_model = _get_best_model(tuner, train_ds, val_ds, use_cache)

    history = _train(best_model, train_ds, val_ds)
    plot_training_history(history)
