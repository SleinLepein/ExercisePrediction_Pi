import tensorflow as tf
from tensorflow.keras.layers import Layer


class CombinatorMLP(Layer):
    def __init__(self, units=(2, 2), **kwargs):
        super().__init__(**kwargs)

        self.neurons = units
        self.size = None
        self.w = None
        self.b = None

    def wi(self, init, name):
        if init == 1:
            return self.add_weight(name='mlp_' + name,
                                   shape=(self.size,),
                                   initializer='ones',
                                   trainable=True)
        elif init == 0:
            return self.add_weight(name='mlp_' + name,
                                   shape=(self.size,),
                                   initializer='RandomNormal',
                                   trainable=True)
        elif init == "normal":
            return self.add_weight(name='mlp_' + name,
                                   shape=(self.size,),
                                   initializer='RandomNormal',
                                   trainable=True)
        else:
            raise ValueError("Invalid argument '%d' provided for init in Gauss layer" % init)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.size = input_shape[0][-1]

        param_count = 4 * self.neurons[0]
        for i, neuron in enumerate(self.neurons[1:]):
            param_count += (self.neurons[i - 1] + 1) * neuron

        param_count += self.neurons[-1] + 1

        init_values = ["normal" for _ in range(param_count)]
        self.w = [self.wi(v, 'a' + str(i + 1)) for i, v in enumerate(init_values)]

        super(CombinatorMLP, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, training=True, **kwargs):
        z_c, u = x

        def compute_interact(y, comp_1, comp_2):
            return tf.nn.leaky_relu(
                y[0] * comp_1 + y[1] * comp_2 + y[2] * comp_1 * comp_2 + y[3],
                alpha=0.1)

        def compute(y, input_layer):
            sum_tensor = 0
            for index in range(len(input_layer)):
                sum_tensor += y[index] * input_layer[index]

            return tf.nn.leaky_relu(sum_tensor + y[-1])

        start_index = 0
        output_tensor = []
        for i in range(self.neurons[0]):
            output_tensor.append(
                compute_interact(self.w[start_index: start_index + 4], u, z_c)
            )
            start_index += 4

        for i, neuron in enumerate(self.neurons[1:]):
            input_tensor = output_tensor
            output_tensor = []
            for k in range(neuron):
                output_tensor.append(
                    compute(self.w[start_index: start_index + len(input_tensor) + 1], input_tensor)
                )
                start_index += len(input_tensor) + 1

        z_est = compute(self.w[start_index:], output_tensor)

        return z_est
