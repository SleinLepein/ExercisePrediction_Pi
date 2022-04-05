from dotdict import DotDict
import tensorflow as tf
from tensorflow.keras.layers import Dense

from templates.vae_model_template.basic_vae_model_class import VariationalAutoencoderModel
from templates.keras_model_template.nn.NormalSampling import NormalSampling

DEFAULT_MODELS_PATH = "../"


class ExampleVAE(VariationalAutoencoderModel):

    standard_params_example_vae = DotDict(
        # model params
        input_shapes=((10,),),

        # losses
        loss="normal_loss",
        loss_weight=1,

        use_kl_loss=True,

        model_summary=True,
    )

    def __init__(self, **params_dict):

        super().__init__(**params_dict)

        # update standard parameters with given parameters
        self.update_params(self.input_params, self.standard_params_example_vae)

        # initialize model architecture
        self._initialize()

        # build model according to specified parameters
        self._build()

        # compile model with specified parameters
        self.compile()

    # ------------------------------------------------------------------------------------------------------------------
    # --------------- initialization and preparation methods -----------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def _specialize_init(self):
        """
        Implement in child class.

        Returns
        -------

        """
        pass

    @staticmethod
    def _prepare_params(**params):
        """
        Implement in child class.

        Returns
        -------

        """
        return DotDict(params)

    # ------------------------------------------------------------------------------------------------------------------
    # --------------- build methods for models -------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def _compute_latents(self,
                         inputs: list,
                         **params: dict) -> (list, list, list):

        """
        Must be implemented in child class.
        Essentially, this is the encoder model.

        Parameters
        ----------
        inputs : list
        params : dict

        Returns
        -------
        (list, list, list)
            list of latent codes, list of latent means, list of latent log variances
        """

        self._prepare_params(**params)

        data_input = inputs[0]

        x = Dense(units=5, activation="relu")(data_input)

        latents_means = Dense(units=2)(x)
        latents_log_vars = Dense(units=2)(x)

        latents = NormalSampling(name=f"latent_vector")((latents_means, latents_log_vars))

        return [latents], [latents_means], [latents_log_vars]

    def _compute_reconstructions(self,
                                 inputs: list,
                                 **params: dict) -> list:

        """
        Must be implemented in child class.
        Essentially, this is the decoder model.

        Parameters
        ----------
        inputs : list
            list of input tensors
        params : dict

        Returns
        -------
        list
            list of tensors needed for reconstruction
        """

        self._prepare_params(**params)

        latents_input = inputs[0]

        x = Dense(units=5, activation="relu")(latents_input)

        reconstructions = [Dense(units=10)(x),
                           Dense(units=10, activation="softplus")(x)]

        return reconstructions

    def _add_further_metrics(self,
                             latents,
                             lat_means,
                             lat_log_vars,
                             rec_loss,
                             regularization_weights,
                             reg_losses,
                             **params):
        """
        Implement in child class.

        Returns
        -------

        """
        self._prepare_params(**params)
        params = DotDict(params)

        if params.loss == "binary_crossentropy":
            rec_val = tf.keras.metrics.BinaryAccuracy()
        elif params.loss == "categorical_crossentropy":
            rec_val = tf.keras.metrics.CategoricalAccuracy()
        else:
            rec_val = rec_loss

        # unweighted sum metric for callbacks
        def sum_metric(kl_loss_layer, tc_loss_layer, mmd_loss_layer):
            def metric(true, pred):
                loss = rec_val(true, pred)

                if kl_loss_layer is not None:
                    loss += kl_loss_layer

                if tc_loss_layer is not None:
                    loss += tc_loss_layer

                if mmd_loss_layer is not None:
                    loss += mmd_loss_layer

                return loss

            return metric

        kl_loss = None
        tc_loss = None
        mmd_loss = None

        for str_loss in ["kl_loss", "tc_loss", "mmd_loss"]:
            if str_loss in reg_losses:
                if str_loss == "kl_loss":
                    kl_loss = reg_losses[str_loss]
                elif str_loss == "tc_loss":
                    tc_loss = reg_losses[str_loss]
                elif str_loss == "mmd_loss":
                    mmd_loss = reg_losses[str_loss]

        self.append_metric(sum_metric(kl_loss, tc_loss, mmd_loss), "unweighted_loss")


if __name__ == '__main__':
    example_VAE = ExampleVAE()
