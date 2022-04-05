import tensorflow
from tensorflow.keras.layers import Layer, Reshape
import tensorflow.keras.backend as K
import numpy as np


class RecursiveReshape(Layer):

    def __init__(self, output_shapes, name=None):
        """
        This layer implements recursive de-flattening of one input tensor with ndim = 2, i.e., with
        shape (batch_size, total_dim). Provided a list of shapes for the output such that total dimensions
        agree with each other, the layer outputs a list of reshaped tensors.
        """

        super().__init__(name=name)

        self.output_shapes = output_shapes
        self.output_dims = []
        self.total_output_dim = 0

    def determine_sizes(self, inputs):
        input_dim = np.prod(K.int_shape(inputs)[1:])

        self.output_dims = []
        for shape in self.output_shapes:
            self.output_dims.append(np.prod(shape))

        self.total_output_dim = np.sum(self.output_dims)

        if input_dim != self.total_output_dim:
            raise ValueError("Total dimension of input and requested output does not match.")

    def call(self, inputs, *args, **kwargs):

        if not tensorflow.is_tensor(inputs):
            raise TypeError("Input must be a tensor.")

        self.determine_sizes(inputs)

        outputs = []
        current_dim = 0
        for i, shape in enumerate(self.output_shapes):
            next_dim = current_dim + self.output_dims[i]

            if len(shape) > 1:
                x = inputs[:, current_dim:next_dim]
                x = Reshape(shape, name=self.name + f"_reshape_{i}")(x)
            else:
                x = inputs[:, current_dim:next_dim]

            outputs.append(x)
            current_dim = next_dim

        return outputs




