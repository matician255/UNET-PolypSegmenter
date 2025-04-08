import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Concatenate, UpSampling2D, Input
from tensorflow.keras.models import Model


def conv_block(x, num_filters):   # we create a custom reusable convolutional block
    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def build_model():
    size = 256
    num_filters = [16, 32, 48, 64]
    inputs = Input((size, size, 3))

    skip_x = []
    x = inputs
    # Encoder
    for f in num_filters:  # loop every filter in the convolutional block
        x = conv_block(x, f)
        skip_x.append(x)  # Store the output in a list
        x = MaxPool2D((2, 2))(x)

    # Bridge
    x = conv_block(x, num_filters[-1])  # this takes the last element in the list

    num_filters.reverse()  # ensures the decoder starts with large filters (to match deep layers).
    skip_x.reverse()  # ensures the skip connections line up properly in reverse order.

    # Decoder
    for i, f in enumerate(num_filters):  # looping through each number of filter f and its index i
        x = UpSampling2D((2, 2))(x)  # here we're reconstructing the image
        xs = skip_x[i]  # this pulls out the corresponding skip connection, the saved details from higher resolution img
        x = Concatenate()([x, xs])
        x = conv_block(x, f)

    # Output
    x = Conv2D(1, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)

    return Model(inputs, x)


if __name__ == "__main__":
    model = build_model()
    model.summary()
