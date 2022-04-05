from tensorflow.keras.layers import Layer, Activation


class ReluSign(Layer):

    def __init__(self, radius=0.1, name=None):
        """
        This layer implements a piecewise linear approximation to signum function. It will be -1 for inputs
        smaller than - radius, +1 for inputs greater than radius and linearly interpolating in between, i.e.,
        it has a slope of 1 / radius.

        Parameters
        ----------
        radius : float
            positive float indicating size of approximation region, i.e., the smaller the closer at true signum
            at the cost of higher gradients
        """

        super().__init__(name=name)

        assert radius > 0

        self.radius = radius

    def call(self, inputs, *args, **kwargs):
        x = 1 + inputs / self.radius
        x = Activation("relu")(x) - 1
        return 1 - Activation("relu")(1 - x)
