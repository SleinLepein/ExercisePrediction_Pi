from tensorflow.python.keras import Input, Model

from templates.keras_model_template.nn.layers.losses import KLLoss, TCLoss, MMDLoss


def get_kl_loss_model(latent_shape,
                      model_name="kl_loss_model",
                      aggregation=True):
    """
    Get model to calculate KL loss in VAE (for Gaussians) for given shape of latents.

    Parameters
    ----------
    latent_shape
    model_name
    aggregation

    Returns
    -------
    KL loss model
    """

    inputs = []
    for i, shape in enumerate([latent_shape, latent_shape]):
        inputs.append(Input(shape[1:], name=f"{model_name}_input_{i}"))

    z_means = inputs[0]
    z_log_vars = inputs[1]

    kl_loss = KLLoss()([z_means, z_log_vars])

    # ----------------- return -----------------

    return Model(inputs, kl_loss, name=model_name)


def get_tc_loss_model(latent_shape,
                      model_name="tc_loss_model",
                      aggregation=True):

    """
    Get model to compute total correlation loss in TCVAE for given shape of latents.

    Parameters
    ----------
    latent_shape
    model_name
    aggregation

    Returns
    -------
    total correlation model
    """

    inputs = []
    for i, shape in enumerate([latent_shape, latent_shape, latent_shape]):
        inputs.append(Input(shape[1:], name=f"{model_name}_input_{i}"))

    z = inputs[0]
    z_means = inputs[1]
    z_log_vars = inputs[2]

    # tc = - total_correlation(z, z_means, z_log_vars, aggregation)
    tc = - TCLoss()([z, z_means, z_log_vars])

    # ----------------- return -----------------

    return Model(inputs, tc, name=model_name)


def get_mmd_loss_model(latent_shape,
                       model_name="mmd_loss_model",
                       aggregation=True):

    inputs = Input(latent_shape[1:], name=f"{model_name}_sample_input")

    outputs = MMDLoss()(inputs)

    return Model(inputs, outputs, name=model_name)
