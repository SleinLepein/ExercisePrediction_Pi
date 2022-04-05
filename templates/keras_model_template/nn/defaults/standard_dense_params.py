from dotdict import DotDict

standard_dense_params = DotDict(
    units=32,  # number of units / neurons / learnable weights / dimension of learnable affine transform / kernels
    activation=None,  # which activation to use after transformation, typically None and separate activation layer
    use_bias=True,  # whether to use learnable bias offset, typically True
    kernel_initializer='glorot_uniform',  # initializer method for random initialization of weight variables
    bias_initializer='zeros',  # initializer method for random initialization of bias variables
    kernel_regularizer=None,  # regularizer method for weight variables
    bias_regularizer=None,  # regularizer method for bias variables
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    trainable=True,  # trainable, i.e., affected by optimization, otherwise fixed
)
