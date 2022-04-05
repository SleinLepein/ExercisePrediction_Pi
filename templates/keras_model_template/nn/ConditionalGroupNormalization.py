import tensorflow as tf
from templates.keras_model_template.nn.GroupNormalization import GroupNormalizationOwn


class ConditionalNormalization(GroupNormalizationOwn):

    def __init__(self, cond_shape=None, **kwargs):
        super().__init__(**kwargs)

        self.cond_shape = cond_shape
        self.agg_beta = None
        self.agg_gamma = None

    def _check_size_of_dimensions(self, input_shape):

        # dim = input_shape[self.axis]
        # if dim < self.groups:
        #     raise ValueError(
        #         "Number of groups (" + str(self.groups) + ") cannot be "
        #         "more than the number of channels (" + str(dim) + ")."
        #     )
        #
        # if dim % self.groups != 0:
        #     raise ValueError(
        #         "Number of groups (" + str(self.groups) + ") must be a "
        #         "multiple of the number of channels (" + str(dim) + ")."
        #     )

        pass

    def _aggregate_weights_by_condition(self, condition):
        self.agg_beta = tf.matmul(condition, self.beta)
        self.agg_gamma = tf.matmul(condition, self.gamma)

    @tf.function
    def call(self, inputs, **kwargs):

        if self.cond_shape is not None:
            inputs, conditions = inputs
            self._aggregate_weights_by_condition(conditions)

        input_shape = tf.keras.backend.int_shape(inputs)
        tensor_input_shape = tf.shape(inputs)

        reshaped_inputs, group_shape = self._reshape_into_groups(
            inputs, input_shape, tensor_input_shape
        )

        normalized_inputs = self._apply_normalization(reshaped_inputs, input_shape)

        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            outputs = tf.reshape(normalized_inputs, tensor_input_shape, name=f"{self.name}_reshape_outputs")
        else:
            outputs = normalized_inputs

        return outputs

    def _add_gamma_weight(self, input_shape):

        if self.cond_shape is None:
            dim = input_shape[self.axis]
            shape = (dim,)
        else:
            dim = input_shape[0][self.axis]
            shape = (self.cond_shape[-1], dim)

        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                name="gamma",
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
            )
        else:
            self.gamma = None

    def _add_beta_weight(self, input_shape):

        if self.cond_shape is None:
            dim = input_shape[self.axis]
            shape = (dim,)
        else:
            dim = input_shape[0][self.axis]
            shape = (self.cond_shape[-1], dim)

        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                name="beta",
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
            )
        else:
            self.beta = None

    def _create_input_spec(self, input_shape):

        if self.cond_shape is None:
            dim = input_shape[self.axis]
            self.input_spec = tf.keras.layers.InputSpec(
                ndim=len(input_shape), axes={self.axis: dim}
            )
        else:
            input_shape, cond_shape = input_shape
            dim = input_shape[self.axis]

            input_spec = tf.keras.layers.InputSpec(
                ndim=len(input_shape), axes={self.axis: dim}
            )

            cond_input_spec = tf.keras.layers.InputSpec(
                ndim=len(cond_shape)
            )

            self.input_spec = [input_spec, cond_input_spec]

    def _create_broadcast_shape(self, input_shape):
        broadcast_shape = [1] * len(input_shape)
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
            broadcast_shape.insert(self.axis, self.groups)
        else:
            broadcast_shape[self.axis] = self.groups
        return broadcast_shape

    def _get_reshaped_weights(self, input_shape):
        broadcast_shape = self._create_broadcast_shape(input_shape)
        broadcast_shape[0] = -1

        gamma = None
        beta = None

        if self.scale:
            gamma = tf.reshape(self.agg_gamma, broadcast_shape, name=f"reshape_gamma")

        if self.center:
            beta = tf.reshape(self.agg_beta, broadcast_shape, name=f"reshape_beta")

        return gamma, beta

    def _reshape_into_groups(self, inputs, input_shape, tensor_input_shape):

        group_shape = [tensor_input_shape[i] for i in range(len(input_shape))]
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            group_shape[self.axis] = input_shape[self.axis] // self.groups
            group_shape.insert(self.axis, self.groups)
            group_shape = tf.stack(group_shape)
            reshaped_inputs = tf.reshape(inputs, group_shape, name=f"reshape_inputs")
            return reshaped_inputs, group_shape
        else:
            return inputs, group_shape

    def _apply_normalization(self, reshaped_inputs, input_shape):

        group_shape = tf.keras.backend.int_shape(reshaped_inputs)
        group_reduction_axes = list(range(1, len(group_shape)))
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            axis = -2 if self.axis == -1 else self.axis - 1
        else:
            axis = -1 if self.axis == -1 else self.axis - 1
        group_reduction_axes.pop(axis)

        mean, variance = tf.nn.moments(
            reshaped_inputs, group_reduction_axes, keepdims=True
        )

        gamma, beta = self._get_reshaped_weights(input_shape)

        normalized_inputs = tf.nn.batch_normalization(
            reshaped_inputs,
            mean=mean,
            variance=variance,
            scale=gamma,
            offset=beta,
            variance_epsilon=self.epsilon,
        )
        return normalized_inputs
