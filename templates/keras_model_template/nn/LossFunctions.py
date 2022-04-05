import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from tensorflow.keras.layers import Lambda


def get_reduction_function(reduction_string):
    if reduction_string.lower() == "sum":
        return tf.keras.losses.Reduction.SUM
    elif reduction_string.lower() == "auto":
        return tf.keras.losses.Reduction.AUTO
    elif reduction_string.lower() == "sum_over_batch":
        return tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
    else:
        return None


def continuous_bernoulli_constant(pred, uniform_part=False):
    """
    Approximate normalization factor in continuous Bernoulli distribution.

    :param pred:
    :param uniform_part:
    :return:
    """

    def cbc(x):
        # Use Taylor approximation for normalization factor up to additive constant
        output = 4 / 3 * tf.keras.backend.pow(x - 1 / 2, 2) + \
                 104 / 45 * tf.keras.backend.pow(x - 1 / 2, 4)

        # add constant if specified to do so
        if uniform_part:
            output += tf.keras.backend.log(2.0)

        # ----------------- return -----------------

        return output

    # ----------------- return -----------------

    # apply function per element in pred
    return tf.map_fn(cbc, pred)


def continuous_bernoulli_correction(pred, reduction="auto"):
    """
    Compute continuous Bernoulli "correction", i.e., offset of both log likelihoods

    :param pred:
    :param reduction:
    :return: correction term
    """

    if reduction.lower() == "sum":
        return K.sum(continuous_bernoulli_constant(pred))
    elif reduction.lower() == "auto":
        return K.mean(continuous_bernoulli_constant(pred))


def binary_crossentropy(true, pred, reduction="auto"):
    """
    Binary crossentropy.

    :param true: true data
    :param pred: predicted data
    :param reduction:
    :return: loss
    """

    true = K.flatten(true)
    pred = K.flatten(pred)

    bce = tf.keras.losses.BinaryCrossentropy(reduction=get_reduction_function(reduction))
    return bce(true, pred)

    # if reduction.lower() == "auto":
    #     return K.mean(tf.keras.losses.binary_crossentropy(true, pred), axis=-1)
    # elif reduction.lower() == "sum":
    #     return K.sum(tf.keras.losses.binary_crossentropy(true, pred), axis=-1)


def weighted_binary_cross_entropy(weights: dict, from_logits: bool = False, axis=None):

    assert 0 in weights
    assert 1 in weights

    weights[0] = float(weights[0])
    weights[1] = float(weights[1])

    def weighted_cross_entropy_fn(y_true, y_pred):
        tf_y_true = tf.cast(y_true, dtype=y_pred.dtype)
        tf_y_pred = tf.cast(y_pred, dtype=y_pred.dtype)

        weights_v = tf.where(tf.equal(tf_y_true, 1), weights[1], weights[0])
        ce = K.binary_crossentropy(tf_y_true, tf_y_pred, from_logits=from_logits)
        loss = K.mean(tf.multiply(ce, weights_v))
        return loss

    return weighted_cross_entropy_fn


def loss_axis_wrapper(loss, length, reduction="auto"):
    assert isinstance(length, int)
    assert length > 0

    def agg_loss(y_true, y_pred):
        total_loss = 0
        for i in range(length):
            total_loss += loss(y_true[:, i], y_pred[:, i])

        if reduction == "auto":
            return total_loss / length
        elif reduction == "sum":
            return total_loss

    return agg_loss


def categorical_crossentropy(true, pred, reduction="auto", flatten=False):
    """

    Parameters
    ----------
    true
    pred
    reduction
    flatten

    Returns
    -------

    """

    if flatten:
        true = K.flatten(true)
        pred = K.flatten(pred)

    bce = tf.keras.losses.CategoricalCrossentropy(reduction=get_reduction_function(reduction))
    return bce(true, pred)

    # if reduction.lower() == "auto":
    #     return K.mean(tf.keras.losses.binary_crossentropy(true, pred), axis=-1)
    # elif reduction.lower() == "sum":
    #     return K.sum(tf.keras.losses.binary_crossentropy(true, pred), axis=-1)


def continuous_crossentropy(true, pred, reduction="auto"):
    """
    Loss function for continuous Bernoulli distribution.

    :param true: true data
    :param pred: predicted data
    :param reduction:
    :return: loss
    """

    loss = binary_crossentropy(true, pred, reduction)
    loss -= continuous_bernoulli_correction(pred, reduction)

    return loss


def poisson(true, pred, reduction="auto"):
    """
    Loss for Poisson distribution.

    :param true: true data
    :param pred: predicted data
    :return: loss
    """

    return tf.keras.backend.mean(tf.keras.losses.poisson(K.flatten(true), K.flatten(pred)))


def mse(true, pred, reduction="auto"):
    return tf.math.reduce_mean(tf.math.square(true - pred))

# TODO: Add further losses


def get_loss(loss_as_string: str,
             reduction: str = "auto"):

    """
    Get loss function for Keras corresponding to a string identifier.

    Parameters
    ----------
    loss_as_string : str
        name of loss
    reduction : str
        type of reduction, "auto"

    Returns
    -------
    loss function
    """

    loss = None

    if isinstance(loss_as_string, str):
        if loss_as_string == "mse":
            loss = mse
            # loss = tf.keras.losses.MeanSquaredError()
        elif loss_as_string == "binary_crossentropy":
            loss = binary_crossentropy
        elif loss_as_string == "continuous_crossentropy":
            loss = continuous_crossentropy
        elif loss_as_string == "poisson":
            loss = poisson
        elif loss_as_string == "categorical_crossentropy":
            loss = categorical_crossentropy
        elif loss_as_string == "normal_loss":
            loss = get_normal_loss(eps=0.00001)
        else:
            raise NotImplementedError(f"Choice of loss {loss_as_string} not implemented.")

    assert loss is not None

    def loss_function(true, pred):
        return loss(true, pred, reduction)

    loss_function.__name__ = f'{loss_as_string}_loss'
    return loss_function


def get_normal_loss(eps=0.00001):

    def normal_loss(y_true, y_pred, reduction="auto"):
        means = y_pred[0]
        log_vars = y_pred[1]
        loss = tf.keras.losses.mse(y_true, means) / (eps + tf.keras.backend.exp(log_vars)) + log_vars

        if reduction == "auto":
            loss = tf.reduce_mean(loss)

        return loss

    return normal_loss


def get_kl_loss(z_means, z_log_vars, aggregation=True):
    """
    Compute KL loss for given latent means and latent log variances, assuming a standard normal prior.

    Parameters
    ----------
    z_means
    z_log_vars
    aggregation

    Returns
    -------
    KL Loss
    """

    # kl_loss = - 0.5 * (z_log_vars - tf.square(z_means) - tf.exp(z_log_vars) + 1)
    kl_loss = Lambda(lambda x:
                     - 0.5 * tf.reduce_sum((x[1] - tf.square(x[0]) - tf.exp(x[1]) + 1), axis=-1)
                     )([z_means, z_log_vars])

    if aggregation:
        kl_loss = tf.reduce_mean(kl_loss)

    # ----------------- return -----------------

    return kl_loss


def gaussian_log_density(samples, mean, log_squared_scale):
    """
    Compute log density for a sample of a multivariate normal distribution specified by summary statistics. Used
    to estimate total correlation of latents.

    Parameters
    ----------
    samples
    mean
    log_squared_scale

    Returns
    -------
    log density
    """

    pi = tf.constant(np.pi)
    normalization = tf.math.log(2. * pi)
    inv_sigma = tf.math.exp(-log_squared_scale)
    tmp = (samples - mean)

    # ----------------- return -----------------

    return -0.5 * (tmp * tmp * inv_sigma + log_squared_scale + normalization)


def total_correlation(z, z_mean, z_log_squared_scale, aggregation=True):
    """
    Compute total correlation.

    Parameters
    ----------
    z
    z_mean
    z_log_squared_scale
    aggregation

    Returns
    -------
    total correlation
    """

    # TODO: check carefully

    # Compute log(q(z(x_j)|x_i)) for every sample in the batch, which is a
    # tensor of size [batch_size, batch_size, num_latents]. In the following
    # comments, [batch_size, batch_size, num_latents] are indexed by [j, i, l].

    log_qz_prob = gaussian_log_density(
        tf.expand_dims(z, 1),
        tf.expand_dims(z_mean, 0),
        tf.expand_dims(z_log_squared_scale, 0)
    )

    # Compute log prod_l p(z(x_j)_l) = sum_l(log(sum_i(q(z(z_j)_l|x_i)))
    # + constant) for each sample in the batch, which is a vector of size
    # [batch_size,].

    log_qz_product = tf.math.reduce_sum(
        tf.math.reduce_logsumexp(log_qz_prob, axis=1, keepdims=False),
        axis=1,
        keepdims=False)

    # Compute log(q(z(x_j))) as log(sum_i(q(z(x_j)|x_i))) + constant =
    # log(sum_i(prod_l q(z(x_j)_l|x_i))) + constant.

    log_qz = tf.math.reduce_logsumexp(
        tf.math.reduce_sum(log_qz_prob, axis=2, keepdims=False),
        axis=1,
        keepdims=False)

    # ----------------- return -----------------

    if aggregation:
        return tf.math.reduce_mean(log_qz - log_qz_product)
    else:
        return log_qz - log_qz_product


def compute_kernel(x, y, name="mmd_kernel"):

    x_size = K.shape(x)[0]

    y_size = K.shape(y)[0]

    dim = K.shape(x)[1]

    new_shape_x = tf.stack([x_size, 1, dim], name=f"{name}_stack_shape_x")
    new_shape_y = tf.stack([1, y_size, dim], name=f"{name}_stack_shape_y")

    tile_x = tf.stack([1, y_size, 1], name=f"{name}_stack_tile_x")
    tile_y = tf.stack([x_size, 1, 1], name=f"{name}_stack_tile_y")

    tiled_x = tf.tile(tf.reshape(x, new_shape_x), tile_x)
    tiled_y = tf.tile(tf.reshape(y, new_shape_y), tile_y)

    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))


def compute_mmd(x, y, aggregation=True):

    x_kernel = compute_kernel(x, x, name="mmd_kernel_x")
    y_kernel = compute_kernel(y, y, name="mmd_kernel_y")
    xy_kernel = compute_kernel(x, y, name="mmd_kernel_xy")

    if aggregation:
        return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)
    else:
        # TODO: check dimensions

        return x_kernel, y_kernel, xy_kernel


