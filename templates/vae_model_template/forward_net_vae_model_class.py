from dotdict import DotDict

import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Add
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from templates.vae_model_template.basic_vae_model_class import VariationalAutoencoderModel
from templates.keras_model_template.nn.CombinatorFunction import combinator
from templates.keras_model_template.nn.ForwardNet import ForwardNet
from templates.vae_model_template.defaults.forward_net_vae_model_defaults import standard_params_vae

DEFAULT_MODELS_PATH = "../"


class ForwardNetVAE(VariationalAutoencoderModel):

    def __init__(self, **params_dict):

        super().__init__(**params_dict)

        # update standard parameters with given parameters
        self.update_params(self.input_params, standard_params_vae)

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
        self.feature_extractor = None
        self.feature_reconstructors = []

        self.lateral_extractors = []
        self.lateral_reconstructors = []

        self.conditional_preprocessor = None

        # TODO: check usage

        # further stuff
        self.input_shape = self.params.input_shapes[0]
        if len(self.input_shape) > 1:
            self.input_dimension = self.input_shape[-1]
        else:
            self.input_dimension = self.input_shape

        self.latent_shapes = []
        self.latent_layer_count = 0
        self.pre_flatten_shapes = {}

    @staticmethod
    def _prepare_params(**params) -> DotDict:
        """
        Implement in child class.

        Returns
        -------

        """

        params = DotDict(params)

        return params

    # ------------------------------------------------------------------------------------------------------------------
    # --------------- build methods for models -------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def _build_conditional_preprocessor(self, conditional_input_shape, params):

        params["name_prefix"] = "cond_pre_"

        inputs = Input(shape=conditional_input_shape, name=f"conditional_pre_input")
        forward_net = ForwardNet(**params)
        outputs = forward_net.forward(inputs)

        self.conditional_preprocessor = Model(inputs, outputs, name="conditional_preprocessor")

        if self.params.model_summary:
            self.save_summary_to_file(self.conditional_preprocessor, "conditional_extractor")

        if self.params.plot_model:
            self.save_model_plot_to_file(self.conditional_preprocessor, "conditional_extractor", expand_nested=False)

    def _add_lateral_connection(self,
                                lateral,
                                conditional_inputs,
                                laterals,
                                lat_means,
                                lat_log_vars,
                                lateral_params):

        """
        Add a lateral connection according to lateral_params from lateral and conditional inputs and append outputs
        to corresponding lists.

        Parameters
        ----------
        lateral : Tensorflow tensor
            lateral sample input
        conditional_inputs : Tensorflow tensor
            conditional inputs
        laterals : list of Tensorflow tensors
            current sample outputs of previous lateral connections
        lat_means : list of Tensorflow tensors
            current mean outputs of previous lateral connections
        lat_log_vars : list of Tensorflow tensors
            current log variance outputs of previous lateral connections
        lateral_params : dict
            definition for current lateral connection

        Returns
        -------
        list of Tensorflow tensors
            updated lateral samples
        list of Tensorflow tensors
            updated lateral means
        list of Tensorflow tensors
            updated lateral log variances
        """

        # build forward net according to specifications in lateral_params (including conditional preprocessing)
        self.lateral_extractors.append(ForwardNet(**lateral_params))

        # apply forward net to lateral and conditional input
        lat, lat_mean, lat_log_var = self.lateral_extractors[-1].forward(lateral, conditional_inputs)

        print("lateral reshapes")
        print(self.lateral_extractors[-1].reshapes)
        print(self.lateral_extractors[-1].shapes)

        # remember shapes for decoder definition
        self.latent_shapes.append(K.int_shape(lat))

        # remember lateral outputs
        laterals.append(lat)
        lat_means.append(lat_mean)
        lat_log_vars.append(lat_log_var)

        # ----------------- return -----------------

        return laterals, lat_means, lat_log_vars

    def _build_lateral_networks(self,
                                conditional_inputs,
                                params):

        """
        Build lateral networks that take features from feature extractor as input and produce samples in the
        latent space possibly taking in conditional inputs.

        Parameters
        ----------
        conditional_inputs : None
            conditional inputs
        params : DotDict
            definitions of lateral connections

        Returns
        -------
        tuple
            tuple of lists of Tensorflow tensors for samples, means and log variances
        """

        # feature extractor must be already defined
        assert self.feature_extractor is not None

        # initialize lists for different outputs
        laterals = []
        lat_means = []
        lat_log_vars = []

        # conditional preprocessing
        if params.conditional_extractor is not None:
            assert self.conditional_preprocessor is not None

            conditional_inputs_pre = self.conditional_preprocessor(conditional_inputs)
        else:
            conditional_inputs_pre = conditional_inputs

        # build lateral connections at defined features of feature extractor
        for i, lateral in enumerate(self.feature_extractor.laterals):

            self.logger.info(f"Building lateral connection {i}")

            # for more than one lateral connection at current feature
            if isinstance(params.latent_extractors[i], tuple) or isinstance(params.latent_extractors[i], list):
                for k, sub_params in enumerate(params.latent_extractors[i]):

                    # add lateral connection
                    laterals, lat_means, lat_log_vars = self._add_lateral_connection(lateral,
                                                                                     conditional_inputs_pre,
                                                                                     laterals,
                                                                                     lat_means,
                                                                                     lat_log_vars,
                                                                                     sub_params)

            # for only one lateral connection at current feature
            else:

                # add lateral connection
                laterals, lat_means, lat_log_vars = self._add_lateral_connection(lateral,
                                                                                 conditional_inputs_pre,
                                                                                 laterals,
                                                                                 lat_means,
                                                                                 lat_log_vars,
                                                                                 params.latent_extractors[i])

        # ----------------- return -----------------

        return laterals, lat_means, lat_log_vars

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
        params = DotDict(params)
        self._prepare_params(**params)

        # ----------------- define input -----------------

        data_inputs = inputs[0]

        # TODO: change to multiple conditional inputs
        
        if len(inputs) > 1:
            conditional_inputs = inputs[1]
        else:
            conditional_inputs = None

        # ----------------- build data feature extractor network -----------------

        self.feature_extractor = ForwardNet(**params.feature_extractor)

        # build global conditional extractor
        if params.conditional_extractor is not None:
            self._build_conditional_preprocessor(conditional_input_shape=params.input_shapes[1],
                                                 params=params.conditional_extractor)

            conditional_inputs_pre = self.conditional_preprocessor(conditional_inputs)
        else:
            conditional_inputs_pre = conditional_inputs

        self.feature_extractor.forward(data_inputs, conditional_inputs_pre)

        print("feature extractor reshapes")
        print(self.feature_extractor.reshapes)
        print(self.feature_extractor.shapes)

        # ----------------- build lateral networks / latents -----------------

        latents, latents_means, latents_log_vars = self._build_lateral_networks(conditional_inputs, params)

        if not isinstance(latents, list):
            latents = [latents]
        if not isinstance(latents, list):
            latents_means = [latents_means]
        if not isinstance(latents, list):
            latents_log_vars = [latents_log_vars]

        return latents, latents_means, latents_log_vars

    def _compute_reconstructions(self,
                                 inputs: list,
                                 conditional_inputs: list,
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

        params = DotDict(params)
        self._prepare_params(**params)

        # ----------------- define input -----------------

        # latent_inputs = inputs[0]
        #
        # if len(inputs) > 1:
        #     conditional_inputs = inputs[1]
        # else:
        #     conditional_inputs = None
        #
        # separation_model = self._get_separation_model(f"separation_model")
        # laterals = separation_model(latent_inputs)
        #
        # if not isinstance(laterals, list):
        #     laterals = [laterals]

        # ----------------- conditional preprocessing -----------------

        laterals = inputs

        if params.conditional_extractor is not None:
            assert self.conditional_preprocessor is not None

            conditional_inputs = self.conditional_preprocessor(conditional_inputs)

        # determine reverse params for lateral and feature extractor models
        # rev_lateral_params = self._compute_rev_lateral_params(**params)
        decoder_params = params.feature_reconstructors
        rev_laterals = []

        lateral_counter = 0
        for i, temp_params in enumerate(params.latent_reconstructors):

            if isinstance(temp_params, tuple) or isinstance(temp_params, list):
                temp_params = list(temp_params)
                sub_rev_laterals = []

                # model part for current lateral connection might consist of multiple parallel submodels

                for k, sub_temp_params in enumerate(temp_params):

                    # build reverse lateral submodel branch
                    self.lateral_reconstructors.append(ForwardNet(**sub_temp_params))

                    # define input
                    lateral_input = laterals[::-1][lateral_counter]

                    # remember output of reverse lateral submodel branch
                    sub_rev_laterals.append(self.lateral_reconstructors[-1].forward(lateral_input, conditional_inputs))

                    print("lateral recons reshapes")
                    print(self.lateral_reconstructors[-1].reshapes)
                    print(self.lateral_reconstructors[-1].shapes)

                    lateral_counter += 1

                if len(temp_params) > 1:
                    # combine parallel submodels for current lateral connection, if needed
                    combined_sub_laterals = combinator(sub_rev_laterals,
                                                       name_prefix=f"lat_comb_{i}_",
                                                       combinator="vanilla",
                                                       layer_type=decoder_params[i].block_params[-1].layer_type, )

                    # remember output of reverse lateral submodel
                    rev_laterals.append(combined_sub_laterals)
                else:
                    # remember output of reverse lateral submodel
                    rev_laterals.extend(sub_rev_laterals)

            else:
                # build reverse lateral submodel

                self.lateral_reconstructors.append(ForwardNet(**temp_params))

                # define input
                lateral_input = laterals[::-1][lateral_counter]

                # remember output of reverse lateral submodel
                rev_laterals.append(self.lateral_reconstructors[-1].forward(lateral_input, conditional_inputs))

                print("lateral recons reshapes")
                print(self.lateral_reconstructors[-1].reshapes)
                print(self.lateral_reconstructors[-1].shapes)

                lateral_counter += 1

        # ----------------- build top down reconstruction networks and combine with laterals -----------------

        # first top down only connected to first reversed lateral
        self.feature_reconstructors.append(ForwardNet(**decoder_params[0]))
        x = self.feature_reconstructors[0].forward(rev_laterals[0], conditional_inputs)

        # combine top down signal and lateral signal for subsequent steps
        for i, rev_lateral in enumerate(rev_laterals[1:]):

            # combine signals
            combined_signal = combinator([rev_lateral, x],
                                         name_prefix=f"comb_{i}_",
                                         combinator="extended_vanilla",
                                         layer_type=decoder_params[i].block_params[-1].layer_type,
                                         )

            # further reconstruction from combined signal
            skip_reconstruction = False
            if "laterals_only" in decoder_params[i + 1].block_params[-1]:
                if decoder_params[i + 1].block_params[-1].laterals_only:
                    skip_reconstruction = True

            if not skip_reconstruction:
                self.feature_reconstructors.append(ForwardNet(**decoder_params[i + 1]))
                x = self.feature_reconstructors[i + 1].forward(combined_signal, conditional_inputs)

        if not isinstance(x, list):
            reconstructions = [x]
        else:
            reconstructions = x

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
        def sum_metric(layer):
            def metric(true, pred):
                loss = rec_val(true, pred)
                loss = Add()([loss, layer])

                return loss

            return metric

        kl_loss = None
        tc_loss = None
        mmd_loss = None

        # regularization_losses = []
        #
        # for str_loss in ["kl_loss", "tc_loss", "mmd_loss"]:
        #     if str_loss in reg_losses:
        #         if str_loss == "kl_loss":
        #             kl_loss = reg_losses[str_loss]
        #             regularization_losses.append(kl_loss)
        #         elif str_loss == "tc_loss":
        #             tc_loss = reg_losses[str_loss]
        #             regularization_losses.append(tc_loss)
        #         elif str_loss == "mmd_loss":
        #             mmd_loss = reg_losses[str_loss]
        #             regularization_losses.append(mmd_loss)

        list_reg_losses = list(reg_losses.values())
        if len(list_reg_losses) > 1:
            sum_reg_loss = Add()(list_reg_losses)
            # self.metrics.append(sum_metric(sum_reg_loss))

            self.training_model.add_metric(sum_reg_loss, "unweighted_reg_loss")

        # unweighted_sum_metric = Lambda(lambda x: tf.math.reduce_sum(x))(regularization_losses)
        # unweighted_sum_metric = Add()(regularization_losses)
        # self.training_model.add_metric(unweighted_sum_metric, "unweighted_loss")

        # self.append_metric(self._custom_metric(unweighted_sum_metric), "unweighted_loss")

        # self.append_metric(sum_metric(kl_loss, tc_loss, mmd_loss), "unweighted_loss")


if __name__ == '__main__':
    params_vae = DotDict(
        # save / load
        save_folder=DEFAULT_MODELS_PATH,  # where to save stuff
        name="vae",  # name to be used
        # TODO: check
        model_load_name=None,

        # model params
        input_shapes=((28, 10),),  # tuple of tuple indicating inout shapes

        feature_extractor=DotDict(final_sampling=False,
                                  name_prefix="feat_",
                                  block_params=(DotDict(layer_params={"filters": 32,
                                                                      "kernel_regularizer": {"type": "L2",
                                                                                             "c": 0.0001}},
                                                        activation="relu",
                                                        layer_type="Conv1D",
                                                        normalization="batch_normalization",
                                                        ),
                                                DotDict(layer_params={"filters": 8,
                                                                      "kernel_regularizer": {"type": "L2",
                                                                                             "c": 0.0001}},
                                                        activation="relu",
                                                        layer_type="Conv1D",
                                                        normalization="batch_normalization",
                                                        laterals=DotDict(act=True),  # important
                                                        ),
                                                )
                                  ),  # dictionary providing definition of feature extractor
        feature_reconstructors=(DotDict(final_sampling=False,
                                        name_prefix="feat_re_",
                                        block_params=(DotDict(layer_params={"filters": 8,
                                                                            "kernel_regularizer": {"type": "L2",
                                                                                                   "c": 0.0001}},
                                                              activation="relu",
                                                              layer_type="Conv1DTranspose",
                                                              normalization="batch_normalization",
                                                              reshape=(28, 8)
                                                              ),
                                                      DotDict(layer_params={"filters": 32,
                                                                            "kernel_regularizer": {"type": "L2",
                                                                                                   "c": 0.0001}},
                                                              activation="relu",
                                                              layer_type="Conv1DTranspose",
                                                              normalization="batch_normalization",
                                                              ),
                                                      DotDict(layer_params={"filters": 10,
                                                                            "kernel_regularizer": {"type": "L2",
                                                                                                   "c": 0.0001}},
                                                              activation="relu",
                                                              layer_type="Conv1DTranspose",
                                                              normalization="batch_normalization",
                                                              ),
                                                      )
                                        ),
                                ),  # dictionary providing definition of feature extractor

        latent_extractors=(DotDict(final_sampling=True,
                                   name_prefix="lat_",
                                   block_params=(DotDict(layer_params={"units": 2,
                                                                       "kernel_regularizer": {"type": "L2",
                                                                                              "c": 0.0001}},
                                                         activation="relu",
                                                         layer_type="Dense",
                                                         normalization="batch_normalization",
                                                         ),
                                                 ),
                                   ),
                           ),  # list of dictionaries providing definitions of lateral models for latents
        latent_reconstructors=(DotDict(final_sampling=False,
                                       name_prefix="lat_re_",
                                       block_params=(DotDict(layer_params={"units": 4,
                                                                           "kernel_regularizer": {"type": "L2",
                                                                                                  "c": 0.0001}},
                                                             activation="relu",
                                                             layer_type="Dense",
                                                             normalization="batch_normalization",
                                                             ),
                                                     DotDict(layer_params={"units": 224,
                                                                           "kernel_regularizer": {"type": "L2",
                                                                                                  "c": 0.0001}},
                                                             activation="relu",
                                                             layer_type="Dense",
                                                             normalization="batch_normalization",
                                                             ),
                                                     ),
                                       ),
                               ),  # list of dictionaries providing definitions of lateral models for latents
        conditional_extractor=None,  # dictionary with definition of global conditional pre processor

        # TODO: add noise
        # encoder_noise=False,
        # noise_std=0.1,

        output_activation="sigmoid",  # activation for reconstructions

        # losses
        loss="mse",  # reconstruction loss / model family
        loss_weight=1,  # weight of reconstruction loss

        use_kl_loss=True,  # whether to use KL loss (always true in VAE)

        kl_loss_weight=1.0,  # weight of KL loss if no annealing is used
        schedule_kl_loss=True,  # whether to use annealing for KL weight
        kl_annealing_params=DotDict(start_epoch=0,
                                    annealing_epochs=5,
                                    start_value=0.0,
                                    end_value=1.0,
                                    method="linear"),  # annealing parameters

        use_tc_loss=True,  # whether to use KL loss (always true in TC-VAE, TC = Total Correlation)
        tc_loss_weight=1.0,  # weight of TC loss if no annealing is used
        schedule_tc_loss=True,  # whether to use annealing for TC weight
        tc_annealing_params=DotDict(start_epoch=0,
                                    annealing_epochs=5,
                                    start_value=0.0,
                                    end_value=1.0,
                                    method="linear"),  # annealing parameters

        use_mmd_loss=True,  # whether to use MMD loss (always true in info-VAE, MMD = Maximum Mean Discrepancy)
        mmd_loss_weight=1.0,  # weight of MMD loss if no annealing is used
        schedule_mmd_loss=True,  # whether to use annealing for MMD weight
        mmd_annealing_params=DotDict(start_epoch=0,
                                     annealing_epochs=5,
                                     start_value=0.0,
                                     end_value=1.0,
                                     method="linear"),  # annealing parameters
    )

    example_VAE = ForwardNetVAE(**params_vae)

    example_VAE.encoder.summary()
    example_VAE.decoder.summary()
