import keras

from .dataset import create_datasets
from .model import create_model


def _train(model, train_ds, val_ds):
    callbacks = [
        keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
    ]
    epochs = 25
    model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_ds,
    )


def run(
    model_name: str = "part2",
    channels: int = 3,
    image_size: tuple[int, int] = (244, 244),
    num_classes: int = 3,
):
    model = create_model(
        input_shape=image_size + (channels,),
        num_classes=num_classes,
        model_name=model_name,
    )
    (train_ds, val_ds) = create_datasets()
    _train(model, train_ds, val_ds)
