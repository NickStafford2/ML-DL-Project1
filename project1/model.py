import keras
from keras import layers


def create_model(
    input_shape: tuple[int, int, int],
    num_classes: int = 3,
    model_name: str = "part2",
):
    # Define the input layer with the specified input shape
    inputs = keras.Input(shape=input_shape)

    # start with a rescaling layer. This normalizes the [0-255] vlaues to [0,1]. Simply
    # divides by 255.
    nn = layers.Rescaling(1.0 / 255)(inputs)

    # Convolutional layer. Reduces the spatial dimensions of the input image.
    nn = layers.Conv2D(128, 3, strides=2, padding="same")(nn)
    nn = layers.BatchNormalization()(nn)
    nn = layers.Activation("relu")(nn)

    """ This is a more advanced technique designed to make the network less linear. The 
    output of this convolution layer is used as an input for later layers. The idea is 
    that by using several convolutions in combination, the network can understand what 
    it is looking at better. 
    """
    first_convolution_reference = nn

    for size in [256, 512, 728]:
        nn = layers.Activation("relu")(nn)
        nn = layers.SeparableConv2D(size, 3, padding="same")(nn)
        nn = layers.BatchNormalization()(nn)

        nn = layers.Activation("relu")(nn)
        nn = layers.SeparableConv2D(size, 3, padding="same")(nn)
        nn = layers.BatchNormalization()(nn)

        nn = layers.MaxPooling2D(3, strides=2, padding="same")(nn)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            first_convolution_reference
        )
        nn = layers.add([nn, residual])  # Add back residual
        first_convolution_reference = nn  # Set aside next residual

    nn = layers.SeparableConv2D(1024, 3, padding="same")(nn)
    nn = layers.BatchNormalization()(nn)
    nn = layers.Activation("relu")(nn)

    nn = layers.GlobalAveragePooling2D()(nn)
    if num_classes == 2:
        units = 1
    else:
        units = num_classes

    nn = layers.Dropout(0.25)(nn)
    outputs = layers.Dense(units, activation="softmax")(nn)
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
