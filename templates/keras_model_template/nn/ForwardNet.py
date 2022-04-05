"""
This file implements a class for building neural networks in keras. In particular, the object allows to create
sequential NNs blocks (weight, normalization, activation) with skip connections and various normalization techniques.
Currently, only weight layers of type "dense" and one as well as two dimensional convolutional and
transposed convolutional layers are implemented. Further types, particularly attention modules and recurrent modules,
are needed in the future.
To build up more complex truly non sequential NNs, use various of these objects and connect them accordingly.
"""

import copy

from dotdict import DotDict

import tensorflow as tf
from tensorflow.keras.layers import add, Flatten, concatenate, Conv2DTranspose, Conv2D, Conv1D, Dense, \
    BatchNormalization, Activation, Reshape, Input

# older versions of Tensorflow / Keras do not have Conv1DTranspose layers
from tensorflow.keras.layers import Conv1DTranspose
# from templates.keras_model_template.nn.Conv1DTranspose import Conv1DTranspose

from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers

from templates.parameter_class_template.helpers.params_handling import get_updated_params

from templates.keras_model_template.nn.NormalSampling import NormalSampling
from templates.keras_model_template.nn.defaults.standard_conv_params import standard_conv_params
from templates.keras_model_template.nn.defaults.standard_dense_params import standard_dense_params
from templates.keras_model_template.nn.defaults.standard_block_params import standard_block_params
from templates.keras_model_template.nn.defaults.standard_params import standard_params
from templates.keras_model_template.nn.defaults.non_serializable_params import non_serializable_params

from templates.keras_model_template.nn.GroupNormalization import InstanceNormalizationOwn
from templates.keras_model_template.nn.ConditionalGroupNormalization import ConditionalNormalization


class ForwardNet:
    def __init__(self, **params_dict):
        """
        Initialize neural net class.

        Parameters
        ----------
        params_dict
        """

        # load params
        self.params = DotDict(params_dict)
        if "collect_shapes" not in self.params:
            self.params["collect_shapes"] = True

        # stores connections to intermediate layers that can be used in a model definition
        self.laterals = []

        # used for internal automatic reshaping
        self.layer_shapes_at_conditional_inputs = []
        self.shapes = []
        self.reshapes = []

    @staticmethod
    def resolve_non_serializable_params(**params) -> DotDict:
        """
        Resolve non serializable parameters, like regularizers, given in params.

        Parameters
        ----------
        params : dict

        Returns
        -------
        DotDict
        """

        # TODO: add more / smarter handling

        # ----------------- parameter loading -----------------

        default_params = copy.deepcopy(non_serializable_params)

        if params is not None:
            default_params.update(params)
        params = default_params

        # ----------------- parameter resolving -----------------

        additional_params = DotDict()
        if params.kernel_regularizer is not None:
            if params.kernel_regularizer.type == "L2":
                additional_params["kernel_regularizer"] = regularizers.l2(params.kernel_regularizer.c)

        # ----------------- return -----------------

        return additional_params

    def forward(self,
                inputs,
                conditional_inputs=None,
                params=None):
        """
        Defines a forward network based on inputs, conditional inputs and according to specified parameters.

        Parameters
        ----------
        inputs : data input tensors
        conditional_inputs : conditional input tensors
        params : parameters

        Returns
        -------
        tf.tensor
            output tensor of network
        """

        # overall standard params
        default_params = copy.deepcopy(standard_params)

        def _prepare_params(block_params: dict) -> (DotDict, DotDict):

            """
            Update default block parameters with given block parameters and determine weight layer parameters.

            Parameters
            ----------
            block_params : dict

            Returns
            -------
            (DotDict, DotDict)
                updated default parameters as dotted dictionary, weight layer parameters
            """

            # ----------------- update current block parameters -----------------

            # standard block params
            block_params = get_updated_params(block_params, standard_block_params)
            block_layer_type = block_params.layer_type

            if block_params.lateral_only:
                # self.laterals.append(x)
                weight_layer_params = None
            else:
                # ----------------- parameter preparation -----------------

                if block_params.laterals is None:
                    block_params.laterals = DotDict(weights=False, bn=False, act=False)

                # if current block is a dense block
                if block_layer_type == "Dense":
                    weight_layer_params = get_updated_params(block_params.layer_params, standard_dense_params)

                # else if current block is a convolutional block
                elif block_layer_type.startswith("Conv"):
                    weight_layer_params = get_updated_params(block_params.layer_params, standard_conv_params)

                    # adjustments for 1D version
                    # for any Conv1D layer (Conv1DTransposed too)
                    if block_layer_type.startswith("Conv1D"):
                        weight_layer_params.kernel_size = weight_layer_params.kernel_size[0]
                        weight_layer_params.strides = weight_layer_params.strides[0]
                        del weight_layer_params["dilation_rate"]

                    # just for Conv1DTranspose layers (own implementation lacks functionality)
                    if block_layer_type == "Conv1DTranspose":
                        old_weight_layer_params = copy.deepcopy(weight_layer_params)
                        weight_layer_params = DotDict(
                            filters=old_weight_layer_params.filters,
                            kernel_size=old_weight_layer_params.kernel_size,
                            strides=old_weight_layer_params.strides,
                            padding=old_weight_layer_params.padding
                        )
                else:
                    raise NotImplementedError(f"No standard parameters for choice {block_layer_type}.")

            return block_params, weight_layer_params

        # ----------------- parameter loading -----------------

        if params is None:
            params = self.params

        if params is not None:
            default_params.update(params)
        params = default_params

        # ----------------- building of neural network -----------------

        # remember nodes to be able to use them as part of the output or for skip connections
        nodes = DotDict(weights=[],
                        normalization=[],
                        activations=[])

        # remember layer shapes to be able to switch between dense and convolution layers
        x = inputs
        input_shape = K.int_shape(x)[1:]

        if params.collect_shapes:
            self.shapes.append(input_shape)

        # iterate over all blocks that define neural net structure

        last_input = None
        for i, current_block_params in enumerate(params.block_params):

            # prepare parameters
            current_block_params, layer_params = _prepare_params(current_block_params)
            layer_type = current_block_params.layer_type

            if layer_params is None:
                self.laterals.append(x)
            else:

                # ----------------- parameter preparation -----------------

                resolved_params = self.resolve_non_serializable_params(**layer_params)
                layer_params.update(resolved_params)

                # ----------------- residual connection to last block -----------------

                if last_input is None:
                    last_input = x
                else:
                    if K.int_shape(last_input) == K.int_shape(x):
                        if current_block_params.skip:
                            x = add([x, last_input])
                    last_input = x

                # ----------------- input adjustments for current block -----------------

                # reshaping if necessary
                if (len(input_shape) > 1) & (layer_type == "Dense"):
                    x = Flatten()(x)

                if params.collect_shapes:
                    self.reshapes.append(K.int_shape(x)[1:])

                # TODO: add further conditional modulation techniques

                # conditional preprocessing
                current_conditional_inputs = None
                if conditional_inputs is not None:
                    if current_block_params.conditional_params is not None:

                        conditional_params = current_block_params.conditional_params
                        if params.name_prefix == "":
                            conditional_name_prefix = f"cond_{i}_"
                        else:
                            conditional_name_prefix = f"{params.name_prefix}cond_{i}_"
                        conditional_params["name_prefix"] = conditional_name_prefix
                        conditional_params["collect_shapes"] = False

                        # recursive call of current method to build up conditional preprocessing network
                        conditional_model = self.forward_keras_model(inputs=conditional_inputs,
                                                                     params=conditional_params)
                        current_conditional_inputs = conditional_model(conditional_inputs)
                    else:
                        current_conditional_inputs = conditional_inputs

                # conditional inputs
                if current_block_params.conditional_input:
                    if params.collect_shapes:
                        self.layer_shapes_at_conditional_inputs.append(K.int_shape(x))
                    x = concatenate([x, current_conditional_inputs])

                # ----------------- reshape -----------------
                if current_block_params.reshape is not None:
                    x = Reshape(current_block_params.reshape,
                                name=f"{params.name_prefix}reshape_{layer_type}_block_{i}")(x)

                if current_block_params.interactions:
                    current_interactions = []
                    for xi in range(K.int_shape(x)[-1]):
                        for xj in range(K.int_shape(x)[-1]):
                            if xj >= xi:
                                interaction_tensor = tf.math.multiply(x[:, xi], x[:, xj])
                                interaction_tensor = tf.expand_dims(interaction_tensor, axis=-1)
                                current_interactions.append(interaction_tensor)
                    x = concatenate([x] + current_interactions, axis=-1)

                # ----------------- current block -----------------

                # if block is final sampling layer for use in e.g. VAE

                if params.final_sampling & (i == len(params.block_params) - 1):

                    # ----------------- return -----------------

                    return self._conditional_sampling(x, layer_params, layer_type, params)

                # if block is not sampling layer, build input -> weights -> batch normalization -> activation
                else:

                    x = self._weight_layer(x, layer_params, layer_type, params, str(i))
                    nodes.weights.append(x)
                    self._add_lateral_connection(x, "weights", current_block_params)

                    # ----------------- batch normalization -----------------

                    if current_block_params.normalization is not None:
                        if current_block_params.normalization == "batch_normalization":
                            x = BatchNormalization(name=f"{params.name_prefix}BN_{layer_type}_block_{i}")(x)

                        elif current_block_params.normalization == "instance_normalization":
                            x = InstanceNormalizationOwn(axis=-1,
                                                         center=True,
                                                         scale=True,
                                                         beta_initializer="zeros",
                                                         gamma_initializer="ones",
                                                         name=f"{params.name_prefix}IN_{layer_type}_block_{i}")(x)

                        elif current_block_params.normalization == "conditional_instance_normalization":

                            if conditional_inputs is None:
                                raise ValueError("conditional_inputs must not be None when using conditional instance"
                                                 "normalization")

                            y = [x, current_conditional_inputs]
                            x_shape = K.int_shape(x)
                            cond_shape = K.int_shape(current_conditional_inputs)

                            x = ConditionalNormalization(name=f"{params.name_prefix}CIN_{layer_type}_block_{i}",
                                                         groups=x_shape[-1],
                                                         center=True,
                                                         scale=True,
                                                         axis=-1,
                                                         beta_initializer="zeros",
                                                         gamma_initializer="ones",
                                                         cond_shape=cond_shape)(y)

                        nodes.normalization.append(x)
                        self._add_lateral_connection(x, "normalizations", current_block_params)

                    # ----------------- activation -----------------

                    x = Activation(current_block_params.activation,
                                   name=f"{params.name_prefix}activation_{layer_type}_block_{i}")(x)

                    nodes.activations.append(x)
                    self._add_lateral_connection(x, "act", current_block_params)

        # ----------------- return -----------------

        return x

    def _add_lateral_connection(self, x, connection_type, current_block_params):
        if connection_type in current_block_params.laterals:
            if current_block_params.laterals[connection_type]:
                self.laterals.append(x)

    def _weight_layer(self,
                      input_layer,
                      weight_layer_params: DotDict,
                      weight_layer_type: str,
                      net_params: DotDict,
                      name_suffix: str):

        """
        Apply weight layer as specified by given parameters to input layer.

        Parameters
        ----------
        input_layer : tf.tensor
        weight_layer_params : DotDict
        weight_layer_type : str
        net_params : DotDict
        name_suffix : str

        Returns
        -------
        tf.tensor
            output of learnable weight layer
        """

        # ----------------- weight layer -----------------

        if weight_layer_type == "Conv2DTranspose":
            output_layer = Conv2DTranspose(name=f"{net_params.name_prefix}{weight_layer_type}_block_{name_suffix}",
                                           **weight_layer_params)(input_layer)
        elif weight_layer_type == "Conv1DTranspose":
            output_layer = Conv1DTranspose(name=f"{net_params.name_prefix}{weight_layer_type}_block_{name_suffix}",
                                           **weight_layer_params)(input_layer)
        elif weight_layer_type == "Conv2D":
            output_layer = Conv2D(name=f"{net_params.name_prefix}{weight_layer_type}_block_{name_suffix}",
                                  **weight_layer_params)(input_layer)
        elif weight_layer_type == "Conv1D":
            output_layer = Conv1D(name=f"{net_params.name_prefix}{weight_layer_type}_block_{name_suffix}",
                                  **weight_layer_params)(input_layer)
        elif weight_layer_type == "Dense":
            output_layer = Dense(name=f"{net_params.name_prefix}{weight_layer_type}_block_{name_suffix}",
                                 **weight_layer_params)(input_layer)
        else:
            raise NotImplementedError(f"Choice {weight_layer_type} not implemented")

        if net_params.collect_shapes:
            self.shapes.append(K.int_shape(output_layer)[1:])

        return output_layer

    def _conditional_sampling(self,
                              input_layer,
                              weight_layer_params: DotDict,
                              weight_layer_type: str,
                              net_params: DotDict):

        """
        Add normal sampling mechanism, i.e., calculate means and log variances from input_layer according to specified
        parameters and sample from resulting normal distributions.

        Parameters
        ----------
        input_layer : tf.tensor
        weight_layer_params : DotDict
        weight_layer_type : str
        net_params : DotDict

        Returns
        -------
        tf.tensor
            samples
        tf.tensor
            means
        tf.tensor
            log variances
        """

        if weight_layer_type == "Conv2DTranspose":
            means = Conv2DTranspose(name=f"{net_params.name_prefix}{weight_layer_type}_means",
                                    **weight_layer_params)(input_layer)
            log_vars = Conv2DTranspose(name=f"{net_params.name_prefix}{weight_layer_type}_log_vars",
                                       **weight_layer_params)(input_layer)

        elif weight_layer_type == "Conv1DTranspose":
            means = Conv1DTranspose(**weight_layer_params)(input_layer)
            log_vars = Conv1DTranspose(**weight_layer_params)(input_layer)

        elif weight_layer_type == "Conv2D":
            means = Conv2D(name=f"{net_params.name_prefix}{weight_layer_type}_means",
                           **weight_layer_params)(input_layer)

            log_vars = Conv2D(name=f"{net_params.name_prefix}{weight_layer_type}_log_vars",
                              **weight_layer_params)(input_layer)

        elif weight_layer_type == "Conv1D":
            means = Conv1D(name=f"{net_params.name_prefix}{weight_layer_type}_means",
                           **weight_layer_params)(input_layer)

            log_vars = Conv1D(name=f"{net_params.name_prefix}{weight_layer_type}_log_vars",
                              **weight_layer_params)(input_layer)

        elif weight_layer_type == "Dense":
            means = Dense(name=f"{net_params.name_prefix}{weight_layer_type}_means",
                          **weight_layer_params)(input_layer)

            log_vars = Dense(name=f"{net_params.name_prefix}{weight_layer_type}_log_vars",
                             **weight_layer_params)(input_layer)
        else:
            raise NotImplementedError(f"Choice {weight_layer_type} for conv_dim not implemented")

        samples = NormalSampling(name=f"{net_params.name_prefix}latent_vector")((means, log_vars))

        self.shapes.append(K.int_shape(samples)[1:])
        self.reshapes.append(K.int_shape(samples)[1:])

        return samples, means, log_vars

    def forward_keras_model(self,
                            inputs,
                            conditional_inputs=None,
                            params=None) -> tf.keras.models.Model:

        """
        Define a Keras model by using forward method.

        Parameters
        ----------
        inputs
        conditional_inputs
        params

        Returns
        -------
        tf.keras.models.Model
            keras model
        """

        input_shape = K.int_shape(inputs)[1:]
        input_tensor = Input(input_shape, name=params.name_prefix + "data_input")
        model_input = [input_tensor]

        conditional_input_tensor = None
        if conditional_inputs is not None:
            conditional_shape = K.int_shape(conditional_inputs)[1:]
            conditional_input_tensor = Input(conditional_shape, name=params.name_prefix + "cond_input")
            model_input.append(conditional_input_tensor)

        outputs = self.forward(input_tensor, conditional_input_tensor, params=params)

        if len(model_input) == 1:
            model = Model(model_input[0], outputs, name=params.name_prefix + "model")
        else:
            model = Model(model_input, outputs, name=params.name_prefix + "model")

        return model
