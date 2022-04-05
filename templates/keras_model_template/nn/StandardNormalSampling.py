import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


class StandardNormalSamplingLayer(Layer):

    """
    Get standard normal samples with shape of given input.
    """

    @staticmethod
    def call(inputs, **kwargs):
        return tf.keras.backend.random_normal(shape=K.shape(inputs))
