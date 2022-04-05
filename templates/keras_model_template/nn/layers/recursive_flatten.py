from tensorflow.keras.layers import Layer, Flatten, concatenate


class RecursiveFlatten(Layer):

    def __init__(self, name=None):
        """
        This layer implements recursive flattening of various input tensors. Given a list of tensors, this layer
        outputs a tensor of shape (batch_size, total_dim).
        """

        super().__init__(name=name)

    def call(self, inputs, *args, **kwargs):

        if not isinstance(inputs, list):
            raise TypeError("Input must be a list of tensors which should all be flattened and concatenated.")

        if len(inputs) == 0:
            raise ValueError("Input is empty.")

        outputs = Flatten()(inputs[0])
        for input_layer in inputs[1:]:
            outputs = concatenate([outputs, Flatten()(input_layer)])

        return outputs




