import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


class NormalSampling(Layer):

    """
    Sampling layer for multivariate Gauss distributions with specified
    """

    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        epsilon = tf.keras.backend.random_normal(shape=K.shape(z_mean))

        # ----------------- return -----------------

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
