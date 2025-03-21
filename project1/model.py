import keras
import math
from tensorboard.data.provider import Hyperparameter
import keras_tuner
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


def _classification_layers(hp, nn, num_classes: int):
    """Flatten the tensor to prepare it for dense layers
    (batch_size, height, width, channels) -> (batch_size, features)"""
    nn = layers.Flatten()(nn)  # Flatten the 4D tensor to 2D (batch_size, features)

    nn = layers.Dense(
        120,
        activation="relu",
    )(nn)

    # Number of Dense layers will be decided by the hyperparameter
    # num_dense_layers = hp.Int("num_dense_layers", min_value=1, max_value=5)

    # Define the max size for the first dense layer based on the flattened image size
    flattened_size = nn.shape[1]

    # The first layer size starts with the max size, and then we scale down as we go deeper
    max_layer_size = flattened_size // 2

    """ Dropout to prevent any one feature from dominating the output during training. """
    if hp.Boolean("dropout"):
        nn = layers.Dropout(0.25)(nn)

    nn = layers.Dense(
        40,
        activation="relu",
    )(nn)
    nn = layers.Dense(
        10,
        activation="relu",
    )(nn)
    last_dense_layer_size = min(num_classes * 2, flattened_size - 1)
    # For each dense layer, create a new Dense layer with decreasing units
    # for i in range(num_dense_layers):
    #     layer_size = last_dense_layer_size + math.floor(
    #         (i + 1) / num_dense_layers * (max_layer_size - last_dense_layer_size)
    #     )
    #     # Calculate the number of units for this layer, decreasing progressively
    #     units = max(
    #         max_layer_size // (i + 1), num_classes
    #     )  # Make sure it doesn't go below num_classes
    #     nn = layers.Dense(
    #         layer_size,
    #         activation="relu",
    #     )(nn)
    """ If the number of classes (num_classes) is 2 (i.e., binary classification), then the 
    output layer should have a single unit, as there are only two possible outcomes 
    (e.g., "yes" or "no"). If num_classes is more than 2 (i.e., multi-class 
    classification), the output layer should have num_classes units, each representing a 
    class. """
    units = 1 if num_classes == 2 else num_classes

    """ Classification layer. We should have the features mapped by now. It is now time 
    to categorize the image based on the features. """
    return layers.Dense(units, activation="softmax")(nn)


def create_model(
    hp: Hyperparameter,
    input_shape: tuple[int, int, int],
    num_classes: int = 3,
    model_name: str = "part2",
):
    inputs = keras.Input(shape=input_shape)

    nn = _initialization_layers(inputs)
    nn = _pooling_layers(nn)
    outputs = _classification_layers(hp, nn, num_classes)

    model = keras.Model(inputs, outputs)

    return _configure_model(hp, model, model_name)


def _configure_model(hp, model, model_name):
    keras.utils.plot_model(model, show_shapes=True, to_file=f"./docs/{model_name}.png")
    learning_rate = 3e-4
    # learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
    )
    return model
