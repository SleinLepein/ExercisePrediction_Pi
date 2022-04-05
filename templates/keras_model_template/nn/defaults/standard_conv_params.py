from dotdict import DotDict

standard_conv_params = DotDict(
        filters=32,  # number of learnable filter transformations
        kernel_size=(3, 3),  # kernel size (a,b) of each filter, i.e., total weights = a * b * filters
        strides=(1, 1),  # striding parameter
        padding="same",  # padding parameter
        dilation_rate=(1, 1),
        activation=None,  # which activation to use after transformation, typically None and separate activation layer
        use_bias=True,   # whether to use learnable bias offset, typically True
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True
    )