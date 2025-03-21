import keras
from keras import layers


def _add_separable_convolution(nn, num_filters: int):
    nn = layers.Activation("relu")(nn)
    """ This is a more efficient version of a standard convolution layer. It does 
    a convolution for each rgb channel independently, and then merges the three 
    outputs together for a single output. 
    Great explanation here: https://www.youtube.com/watch?v=T7o3xvJLuHk
    """
    nn = layers.SeparableConv2D(filters=num_filters, kernel_size=3, padding="same")(nn)
    nn = layers.BatchNormalization()(nn)
    return nn


def create_model(
    input_shape: tuple[int, int, int],
    num_classes: int = 3,
    model_name: str = "part2",
):
    # Define the input layer with the specified input shape
    inputs = keras.Input(shape=input_shape)

    """ Start with a rescaling layer. This normalizes the [0-255] vlaues to [0,1]. Simply
    divides by 255."""
    nn = layers.Rescaling(1.0 / 255)(inputs)

    """ Convolutional layer. Reduces the spatial dimensions of the input image. Since 
    this is the first layer, we use a normal convolution instead of a separable 
    convolution."""
    nn = layers.Conv2D(128, 3, strides=2, padding="same")(nn)

    """ We choose to do plenty of batch normalization. This is a reccomended practice, 
    and seems to help everywhere without hurting performance much. It should help 
    prevent overfitting and should improve generalization."""
    nn = layers.BatchNormalization()(nn)
    nn = layers.Activation("relu")(nn)

    """ This is a more advanced technique designed to make the network less linear. The 
    output of this convolution layer is used as an input for later layers. The idea is 
    that by using several convolutions in combination, the network can understand what 
    it is looking at better. """
    previous_convolution_reference = nn

    """ Feature refinement loop. As we progress through the network, we should be 
    identifying larger scale features of the image, shapes at the beginning and hopefully
    recognizing faces at the end. we will keep adding convolution layers which 
    increasingly identify more features, but also look at the image at a larger scale."""
    for num_filters in [256, 512, 728]:
        """These two convolution layers do the primary work of identifying features
        at each layer of resolution. This is the main code block. Everything else is
        just connnection layers."""
        nn = _add_separable_convolution(nn, num_filters)
        nn = _add_separable_convolution(nn, num_filters)

        """ After a few convolutions, shrink the layer size. Basically cut it in half to
        improve performance and identify larger scale features on the next layer."""
        nn = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(nn)

        # Project residual
        """ Use a 1x1 convolution so that the number of filters of the previous residual
        layer is the same as the current layer. 1x1 keeps the spacial dimensions the 
        same, while adding to the number of channels. 2 strides means the feature map is
        downsized by 1/2."""
        residual = layers.Conv2D(
            filters=num_filters, kernel_size=1, strides=2, padding="same"
        )(previous_convolution_reference)
        """ Residual connection. Allow the network to look at previous layers to improve
        performance. Then update the previous residual connection."""
        nn = layers.add([nn, residual])
        previous_convolution_reference = nn

    nn = layers.SeparableConv2D(filters=1024, kernel_size=3, padding="same")(nn)
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
