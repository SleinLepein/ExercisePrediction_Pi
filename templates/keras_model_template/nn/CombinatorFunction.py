import copy

from dotdict import DotDict

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Conv1D, Conv2DTranspose, Dense, \
    BatchNormalization, Activation, concatenate, Multiply

from templates.keras_model_template.nn.CombinatorMLP import CombinatorMLP
from templates.keras_model_template.nn.Conv1DTranspose import Conv1DTranspose
from templates.keras_model_template.nn.defaults.standard_conv_params import standard_conv_params
from templates.keras_model_template.nn.defaults.standard_dense_params import standard_dense_params


def combinator(inputs, **params):

    """
    Combine different streams of data to calculate a mixture of all of them.

    :param inputs: list of input data
    :param params: parameter dictionary
    :return: output tensor
    """

    # ----------------- parameter loading and preparation -----------------

    standard_params = DotDict(
        layer_type="Dense",
        name_prefix="",
        activation="relu",
        combinator="extended_vanilla"
    )

    params = DotDict(params)

    if params is not None:
        standard_params.update(params)
    params = standard_params

    # ----------------- check input -----------------

    assert isinstance(inputs, list)
    assert len(inputs) > 1

    relevant_shapes = [K.int_shape(item)[1:] for item in inputs]
    assert len(set(relevant_shapes)) == 1

    shape = K.int_shape(inputs[0])[1:]

    # ----------------- mixture layer -----------------

    def temp_function(input_layer, **temp_params):
        temp_params = DotDict(temp_params)

        if temp_params.layer_type == "Conv2D":
            filters = shape[-1]
            conv_params = copy.deepcopy(standard_conv_params)
            conv_params.filters = filters
            conv_params.kernel_size = (1, 1)
            outputs = Conv2D(name=f"{temp_params.name_prefix}combinator_conv", **conv_params)(input_layer)

        elif temp_params.layer_type == "Conv1D":
            # TODO parameter corrections for 1D

            filters = shape[-1]
            conv_params = copy.deepcopy(standard_conv_params)
            conv_params.filters = filters
            conv_params.kernel_size = (1, 1)
            outputs = Conv1D(name=f"{temp_params.name_prefix}combinator_conv", **conv_params)(input_layer)

        elif temp_params.layer_type == "Conv1DTranspose":
            # TODO parameter corrections for 1D
            # filters, kernel_size, strides=1, padding='valid'

            filters = shape[-1]

            conv_params = DotDict(
                filters=filters,
                kernel_size=1,
                strides=standard_conv_params.strides[0],
                padding=standard_conv_params.padding
            )

            outputs = Conv1DTranspose(**conv_params)(input_layer)

        elif temp_params.layer_type == "Conv2DTranspose":

            filters = shape[-1]
            conv_params = copy.deepcopy(standard_conv_params)
            conv_params.filters = filters
            conv_params.kernel_size = (1, 1)
            outputs = Conv2DTranspose(name=f"{temp_params.name_prefix}combinator_deconv", **conv_params)(input_layer)

        elif temp_params.layer_type == "Dense":
            units = shape[-1]
            dense_params = copy.deepcopy(standard_dense_params)
            dense_params.units = units
            outputs = Dense(name=f"{temp_params.name_prefix}combinator_dense", **dense_params)(input_layer)

        else:
            raise NotImplementedError("Choice of type not implemented!")

        outputs = BatchNormalization(name=f"{temp_params.name_prefix}combinator_BN")(outputs)

        outputs = Activation(temp_params.activation,
                             name=f"{temp_params.name_prefix}combinator_act")(outputs)

        return outputs

    # ----------------- combinator definition -----------------

    # standard version that concatenates inputs and uses a dense or convolutional layer for the mixture
    if params.combinator == "vanilla":
        x = concatenate(inputs, axis=-1)
        x = temp_function(x, **params)

    # this version only works for two inputs but also uses it's element wise products
    elif params.combinator == "extended_vanilla":
        assert len(inputs) == 2

        x = concatenate([inputs[0], inputs[1], Multiply()([inputs[0], inputs[1]])], axis=-1)
        x = temp_function(x, **params)

    # this version also only works for two streams and is used in Ladder Networks
    elif params.combinator == "MLP":
        assert len(inputs) == 2
        x = CombinatorMLP(units=(2, 2))([inputs])

    else:
        raise NotImplementedError(f"Choice of combinator function {params.combinator} is not implemented.")

    # ----------------- return -----------------

    return x
