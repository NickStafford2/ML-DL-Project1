import keras
from keras import layers


def create_model(
    input_shape: tuple[int, int, int],
    num_classes: int = 3,
    model_name: str = "part2",
):
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
    model = keras.Model(inputs, outputs)

    return _configure_model(model, model_name)


def _configure_model(model, model_name):
    keras.utils.plot_model(model, show_shapes=True, to_file=f"./docs/{model_name}.png")
    model.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
    )
    return model
