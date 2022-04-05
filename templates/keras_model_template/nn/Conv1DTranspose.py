import tensorflow as tf
from tensorflow.keras.layers import Layer


class Conv1DTranspose(Layer):

    """
    Conv1DTranspose Layers exist in tensorflow >= 2.3?
    Here, definition for older versions.
    """

    def __init__(self, filters, kernel_size, strides=1, padding='valid', name=""):
        super().__init__(name=name)

        # definition of transposed layer
        self.conv2dtranspose = tf.keras.layers.Conv2DTranspose(
            filters, (kernel_size, 1), (strides, 1), padding
        )

    # TODO: check problem with call ???

    def call(self, x, **kwargs):
        x = tf.expand_dims(x, axis=2)
        x = self.conv2dtranspose(x)
        x = tf.squeeze(x, axis=2)

        # ----------------- return -----------------

        return x
