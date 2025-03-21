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


def _pooling_layers(nn, filter_sizes=[256, 512, 728]):
    previous_convolution_reference = nn

    """ Feature refinement loop. As we progress through the network, we should be 
    identifying larger scale features of the image, shapes at the beginning and hopefully
    recognizing faces at the end. we will keep adding convolution layers which 
    increasingly identify more features, but also look at the image at a larger scale.

    Use padding=same constantly to keep each block of the network from shrinking. We 
    need to keep the dimensions aligned for the add at the end of the block. """
    for num_filters in filter_sizes:
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

    """ One final pooling layer to extract a more features."""
    nn = layers.SeparableConv2D(filters=1024, kernel_size=3, padding="same")(nn)
    nn = layers.BatchNormalization()(nn)
    nn = layers.Activation("relu")(nn)
    return nn


def _initialization_layers(inputs):
    """Start with a rescaling layer. This normalizes the [0-255] vlaues to [0,1]. Simply
    divides by 255."""
    nn = layers.Rescaling(1.0 / 255)(inputs)

    """ Convolution layer. Use padding=same to keep dimensions consistent. Since this is 
    the first layer, we use a normal convolution instead of a separable convolution."""
    nn = layers.Conv2D(filters=128, kernel_size=3, strides=2, padding="same")(nn)

    """ We choose to do plenty of batch normalization. This is a reccomended practice, 
    and seems to help everywhere without hurting performance much. It should help 
    prevent overfitting and should improve generalization."""
    nn = layers.BatchNormalization()(nn)
    nn = layers.Activation("relu")(nn)
    return nn


def _classification_layers(nn, num_classes: int):
    """now that the features have been reduced, change the shape of the tensor from 4d to 2d:
    (batch_size, height, width, channels) -> (batch_size, channels)"""
    nn = layers.GlobalAveragePooling2D()(nn)

    """ If the number of classes (num_classes) is 2 (i.e., binary classification), then the 
    output layer should have a single unit, as there are only two possible outcomes 
    (e.g., "yes" or "no"). If num_classes is more than 2 (i.e., multi-class 
    classification), the output layer should have num_classes units, each representing a 
    class. """
    units = 1 if num_classes == 2 else num_classes

    """ Dropout to prevent any one feature from dominating the output during training. """
    nn = layers.Dropout(0.25)(nn)

    """ Classification layer. We should have the features mapped by now. It is now time 
    to categorize the image based on the features. """
    return layers.Dense(units, activation="softmax")(nn)


def create_model(
    input_shape: tuple[int, int, int],
    num_classes: int = 3,
    model_name: str = "part2",
):
    inputs = keras.Input(shape=input_shape)

    nn = _initialization_layers(inputs)
    nn = _pooling_layers(nn)
    outputs = _classification_layers(nn, num_classes)

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
