import tensorflow as tf

from tensorflow.keras.layers import Layer, Lambda, Multiply

from templates.keras_model_template.nn.StandardNormalSampling import StandardNormalSamplingLayer
from templates.keras_model_template.nn.LossFunctions import get_kl_loss, compute_mmd, gaussian_log_density


class KLLoss(Layer):
    def __init__(self, aggregation=True):
        super().__init__()
        self.aggregation = aggregation

    def call(self, inputs, *args, **kwargs):
        z_means, z_log_vars = inputs

        return get_kl_loss(z_means, z_log_vars, aggregation=self.aggregation)


class TCLoss(Layer):
    def __init__(self, aggregation=True):
        super().__init__()
        self.aggregation = aggregation

    def call(self, inputs, *args, **kwargs):
        z, z_mean, z_log_squared_scale = inputs

        # Compute log(q(z(x_j)|x_i)) for every sample in the batch, which is a
        # tensor of size [batch_size, batch_size, num_latents]. In the following
        # comments, [batch_size, batch_size, num_latents] are indexed by [j, i, l].

        # log_qz_prob = gaussian_log_density(
        #     tf.expand_dims(z, 1),
        #     tf.expand_dims(z_mean, 0),
        #     tf.expand_dims(z_log_squared_scale, 0)
        # )

        log_qz_prob = Lambda(lambda x: gaussian_log_density(tf.expand_dims(x[0], 1),
                                                            tf.expand_dims(x[1], 0),
                                                            tf.expand_dims(x[2], 0)
                                                            )
                             )([z, z_mean, z_log_squared_scale])

        # Compute log prod_l p(z(x_j)_l) = sum_l(log(sum_i(q(z(z_j)_l|x_i)))
        # + constant) for each sample in the batch, which is a vector of size
        # [batch_size,].

        # log_qz_product = tf.math.reduce_sum(
        #     tf.math.reduce_logsumexp(log_qz_prob, axis=1, keepdims=False),
        #     axis=1,
        #     keepdims=False)

        log_qz_product = Lambda(lambda l: tf.math.reduce_sum(tf.math.reduce_logsumexp(l, axis=1, keepdims=False),
                                                             axis=1,
                                                             keepdims=False))(log_qz_prob)

        # Compute log(q(z(x_j))) as log(sum_i(q(z(x_j)|x_i))) + constant =
        # log(sum_i(prod_l q(z(x_j)_l|x_i))) + constant.

        # log_qz = tf.math.reduce_logsumexp(
        #     tf.math.reduce_sum(log_qz_prob, axis=2, keepdims=False),
        #     axis=1,
        #     keepdims=False)

        log_qz = Lambda(lambda l: tf.math.reduce_logsumexp(tf.math.reduce_sum(l, axis=2, keepdims=False),
                                                           axis=1,
                                                           keepdims=False))(log_qz_prob)

        # ----------------- return -----------------

        if self.aggregation:
            return tf.math.reduce_mean(log_qz_product - log_qz)
        else:
            return log_qz_product - log_qz


class MMDLoss(Layer):
    def __init__(self, aggregation=True):
        super().__init__()
        self.aggregation = aggregation

    def call(self, inputs, *args, **kwargs):
        true_samples = StandardNormalSamplingLayer()(inputs)
        return compute_mmd(true_samples, inputs, self.aggregation)


# class WeightedVAELosses(Layer):
#     def __init__(self, loss_weights: dict = None):
#         super().__init__()
#
#         if loss_weights is None:
#             loss_weights = {}
#
#         self.loss_weights = loss_weights
#
#     def call(self, inputs, *args, **kwargs):
#         z, z_means, z_log_vars = inputs
#
#         losses = {}
#         scaled_losses = {}
#         for str_loss in sorted(self.loss_weights):
#             if str_loss == "kl_loss":
#                 loss = KLLoss()([z_means, z_log_vars])
#             elif str_loss == "tc_loss":
#                 loss = TCLoss()([z, z_means, z_log_vars])
#             elif str_loss == "mmd_loss":
#                 loss = MMDLoss()(z)
#             else:
#                 loss = None
#
#             losses[str_loss] = loss
#
#             if tf.is_tensor(self.loss_weights[str_loss]):
#                 weight = Lambda(lambda x: tf.reduce_mean(x))(self.loss_weights[str_loss])
#                 loss_scaled = Multiply()([weight, loss])
#             else:
#                 loss_scaled = self.loss_weights[str_loss] * loss
#
#             # if tf.is_tensor(self.loss_weights[str_loss]):
#             #     self.training_model.add_metric(weight, f"{str_loss}_weight")
#
#             scaled_losses[str_loss] = loss_scaled
#
#             # self.training_model.add_loss(loss_scaled)
#             # self.training_model.add_metric(loss, name=str_loss)
#
#         return list(losses.values()) + list(scaled_losses.values())
