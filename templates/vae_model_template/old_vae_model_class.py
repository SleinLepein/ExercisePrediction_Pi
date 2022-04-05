"""
This file implements a Variational Autoencoder class with symmetric architecture and various regularization losses."
"""

import os
import copy
from dotdict import DotDict

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Flatten, concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from templates.keras_model_template.keras_model_class import KerasModel
from templates.keras_model_template.data_generators.annealing_data_generator import AnnealingDataGenerator
from templates.keras_model_template.nn.CombinatorFunction import combinator
from templates.keras_model_template.nn.ForwardNet import ForwardNet
from templates.keras_model_template.nn.StandardNormalSampling import StandardNormalSamplingLayer
from templates.vae_model_template.defaults.symm_vae_model_defaults import standard_params_vae

DEFAULT_MODELS_PATH = "../"


class VariationalAutoencoderModel(KerasModel):

    """
    This is a Tensorflow / Keras implementation of a Variational Autoencoder architecture. It includes
    various previous results. In particular, the architecture allows for stacked / hierarchical encoder - decoder
    models as well as dynamic weighting of different loss terms like "KL Loss" (from original VAE definition),
    Total Correlation Loss and MMD Loss. Currently, dense and convolutional layers are supported for
    feature extractor and lateral networks. At the moment, only encoder definitions are needed since decoder
    architecture is inferred automatically to produce a more or less symmetric setup of encoder and decoder. Due
    to combining top-down and lateral signals in stacked setups, the decoder model is typically a bit larger
    compared to the encoder model.

    Notes on losses:

    KL loss refers here to KL divergence between data dependent encoder distribution and data independent
    Gaussian prior (see derivation of ELBO estimate). We want encoder distribution to be informative about the
    data input, i.e., should produce specific latents instead of generic latents from prior. The term is in
    tension with reconstruction error and therefore will not optimize towards 0 during training but towards
    a data dependent positive constant involving mutual information between data and latents.

    TC loss measures total correlation of latents and can be optimized towards 0. If it is 0, then the latents
    are independent of each other. This kind of regularization is used to enforce more disentangled factors
    of variation.

    MMD loss is a kernel based similarity measure between two distributions that can be optimized easily. In the
    case of VAEs it measures the similarity between aggregate encoder distribution and standard normal prior.
    In contrast to KL loss above, this term can be optimized towards 0 since in general (over all data points) we
    want the latent samples to follow the chosen latent prior.

    ----------------------------------------------------------------------------------------------------------------
    ------------------------------------- IMPORTANT MEMBERS --------------------------------------------------------
    ----------------------------------------------------------------------------------------------------------------

    -------------------------- models --------------------------
    training_model: holds Keras model with all relevant losses etc. to be used during training
    encoder: holds Keras model for encoder of VAE training model
    encoder_flattened: same as encoder but for convenience with flattened / unstacked outputs
    decoder: holds Keras model for decoder of VAE training model
    decoder_flattened: same as decoder but for convenience with flattened / unstacked inputs

    ----------------------------------------------------------------------------------------------------------------
    ------------------------------------- SOME PAPERS --------------------------------------------------------------
    ----------------------------------------------------------------------------------------------------------------

    Auto-Encoding Variational Bayes
    https://arxiv.org/pdf/1312.6114.pdf

    An Introduction to Variational Autoencoders
    https://arxiv.org/pdf/1906.02691.pdf

    Ladder Variational Autoencoders
    https://papers.nips.cc/paper/2016/file/6ae07dcb33ec3b7c814df797cbda0f87-Paper.pdf

    ELBO surgery: yet another way to carve up the variational evidence lower bound
    http://approximateinference.org/accepted/HoffmanJohnson2016.pdf

    Understanding disentangling in $\beta$-VAE
    https://arxiv.org/pdf/1804.03599.pdf

    A closer look at Disentangling in $\beta$-VAE
    https://arxiv.org/pdf/1912.05127.pdf

    Isolating Sources of Disentanglement in VAEs
    https://arxiv.org/pdf/1802.04942.pdf

    Disentangling by Factorising
    https://arxiv.org/pdf/1802.05983.pdf

    Info VAE: Balancing Learning and Inference in Variational Autoencoders
    https://arxiv.org/pdf/1706.02262.pdf

    Fixing a broken ELBO
    https://arxiv.org/pdf/1711.00464.pdf

    ControlVAE: Controllable Variational Autoencoder
    https://arxiv.org/pdf/2004.05988.pdf
    """

    def __init__(self, **params_dict):

        """
        Initializes Variational Autoencoder model based on provided keyword parameters.

        Parameters
        ----------
        input_shape : tuple
            data input shape, default: (28, 28, 1)
        feature_extractor : dict
            dictionary with net definitions (see ForwardNet), default: None
        laterals : list of dicts
            list of net definitions for lateral connections, default: None
        output_activation : str
            name of output activation for reconstructions, default: "sigmoid",
        loss : str
            name of loss function to use for reconstructions, default: "continuous_crossentropy",
        loss_weight : float
            weight for reconstruction loss, default: 1.0
        use_kl_loss : bool
            whether to use a KL regularization which is usually the case for VAEs, default: True
        kl_importance_sampling : bool
            whether to scale KL loss by latent_dim / input_dim, default: False
        kl_annealing_params : dict
            dictionary with definitions for parameter annealing of weight for KL loss,
                                 default:   DotDict(start_epoch=0,
                                                    annealing_epochs=5,
                                                    start_value=0.0,
                                                    end_value=1.0,
                                                    method="linear"),
        use_tc_loss : bool
            whether to use a total correlation regularization, default: True
        tc_annealing_params : dict
            dictionary with definitions for parameter annealing of weight for TC loss,
                                 default:   DotDict(start_epoch=0,
                                                    annealing_epochs=5,
                                                    start_value=0.0,
                                                    end_value=1.0,
                                                    method="linear"),
        use_mmd_loss : bool
            whether to use mmd regularization, default: True
        mmd_annealing_params : dict
            dictionary with definitions for parameter annealing of weight for MMD loss,
                                 default:   DotDict(start_epoch=0,
                                                    annealing_epochs=5,
                                                    start_value=0.0,
                                                    end_value=1.0,
                                                    method="linear"),
        conditional_input_shape : tuple or None
            shape of conditional inputs (labels), default: None,

        ----------------------------------------------------------------------------------------------------------------
        ------------------------------------- PARAMETER DETAILS --------------------------------------------------------
        ----------------------------------------------------------------------------------------------------------------

        # save / load
        build_from_submodels: default: True (not implemented otherwise yet)
        model_load_name=None (not working correctly)

        encoder_noise=False (not implemented),
        noise_std=0.1, (not implemented),
        """

        super().__init__(**params_dict)

        self.standard_params_vae = standard_params_vae

        if self.params.params_file_path is not None:
            # load saved params_dict from file and update default values
            folder = os.path.dirname(self.params.params_file_path)
            file_name = os.path.basename(self.params.params_file_path)
            self.params = self.load_params(folder, file_name)

            # update default parameters with loaded parameters
            self.update_params(self.params, self.standard_params_vae)

            # logging messages
            self.logger.info(f"Loading parameters from file {file_name} in folder {folder}. \n "
                             f"Loaded parameters are: \n")
            self.logger.info(self.params)

        else:
            # load params_dict and update default values
            self.update_params(self.input_params, self.standard_params_vae)

            # save params
            self.save_params(self.params.save_folder, self.params.name)
            self.print_params_to_file(self.params.save_folder, self.params.name)

            # logging messages
            self.logger.info(f"Parameters are not loaded from file. "
                             f"Current parameters are saved in folder {self.params.save_folder}.")

        # initialize model architecture
        self._initialize()

        # build model according to specified parameters
        self._build()

        # compile model with specified parameters
        self.compile()

        # load weights / model from file if specified to do so
        if self.params.model_load_name is not None:
            self.load(folder_path=self.params.save_folder, model_name=self.params.model_load_name)

    # ------------------------------------------------------------------------------------------------------------------
    # --------------- initialization and preparation methods -----------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def _initialize(self):

        """
        Initialize class members.

        Returns
        -------
        None
        """

        # possible submodels
        self.encoder = None
        self.encoder_flattened = None
        self.decoder = None
        self.decoder_flattened = None

        self.feature_extractor = None
        self.feature_reconstructors = []
        self.lateral_extractors = []
        self.lateral_reconstructors = []
        self.conditional_preprocessor = None

        # model inputs and model outputs
        self.encoder_inputs = []
        self.encoder_outputs = []
        self.encoder_means = []
        self.encoder_log_vars = []

        self.decoder_inputs = []
        self.training_model_inputs = []
        self.training_model_outputs = []

        # further stuff
        self.input_shape = self.params.input_shape
        if len(self.input_shape) > 1:
            self.input_dimension = self.input_shape[-1]
        else:
            self.input_dimension = self.input_shape

        self.latent_shapes = []
        self.latent_layer_count = 0
        self.pre_flatten_shapes = {}

    @staticmethod
    def _prepare_params(**params):

        """
        Prepare input parameters for further use in this class.

        Parameters
        ----------
        params : dict or DotDict

        Returns
        -------
        DotDict
            prepared params
        """

        params = DotDict(params)

        # ----------------- adjustments for lateral models -----------------

        for i, lateral in enumerate(params.laterals):

            if isinstance(lateral, tuple) or isinstance(lateral, list):
                lateral = list(lateral)
                params.laterals[i] = lateral

                for k, sub_lateral in enumerate(lateral):
                    params.laterals[i][k].name_prefix = f"lat_{i}_{k}_"

                    # TODO: add parameter for conditions
            else:
                params.laterals[i].name_prefix = f"lat_{i}_"

        # ----------------- adjustments for parameter annealing ------------------

        if "kl_annealing_params" not in params:
            params["kl_annealing_params"] = {}

        params["kl_annealing_params"]["name"] = "kl_annealing"

        if "tc_annealing_params" not in params:
            params["tc_annealing_params"] = {}

        params["tc_annealing_params"]["name"] = "tc_annealing"

        if "mmd_annealing_params" not in params:
            params["mmd_annealing_params"] = {}

        params["mmd_annealing_params"]["name"] = "mmd_annealing"

        # ----------------- return -----------------

        return params

    # ------------------------------------------------------------------------------------------------------------------
    # --------------- methods to define symmetric decoder from given encoder params ------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _reverse_block_params(input_block_params):

        """
        Reverse order of block parameters for automatic decoder definitions. Essentially, convolution layers
        are switched to transposed convolution layers.

        Parameters
        ----------
        input_block_params : DotDict

        Returns
        -------
        DotDict
            reversed blocked params
        """

        block_params = copy.deepcopy(input_block_params)

        for i, params in enumerate(block_params):
            if params.layer_type == "Conv2D":
                block_params[i].layer_type = "Conv2DTranspose"
            elif params.layer_type == "Conv1D":
                block_params[i].layer_type = "Conv1DTranspose"

        # ----------------- return -----------------

        return block_params

    # TODO: check

    @staticmethod
    def _shift_block_params(input_block_params, shapes, reshapes):
        """

        Parameters
        ----------
        input_block_params
        shapes
        reshapes

        Returns
        -------

        """

        block_params = copy.deepcopy(input_block_params)

        for i, _ in enumerate(block_params):

            # shift units / filters
            if len(reshapes[i]) == 1:
                # init if missing
                if "layer_params" not in block_params[i]:
                    block_params[i].layer_params = DotDict(units=0)

                # get number of units
                block_params[i].layer_params.units = reshapes[i][-1]
            else:
                # init if missing
                if "layer_params" not in block_params[i]:
                    block_params[i].layer_params = DotDict(filters=0)

                # get number of filters
                block_params[i].layer_params.filters = reshapes[i][-1]

            # reshape if necessary
            if i > 0:
                # reshape if necessary (e.g. Conv -> Dense or Dense -> Conv)
                if shapes[i-1] != reshapes[i-1]:
                    block_params[i].reshape = shapes[i-1]

        # ----------------- return -----------------

        return block_params

    @staticmethod
    def _split_decoder_params(input_block_params):
        block_params = copy.deepcopy(input_block_params)
        lateral_conditions = []

        for i, params in enumerate(block_params):
            if params.laterals is not None:
                lateral_conditions.append(True)
            else:
                lateral_conditions.append(False)
            block_params[i].laterals = None

        lateral_conditions = lateral_conditions[1:] + [True]

        new_blocks = []

        indices = np.argwhere(np.array(lateral_conditions) == 1).reshape(-1)
        old_index = indices[0] + 1
        new_blocks.append(block_params[:old_index])

        for index in indices[1:-1]:
            index = index + 1
            new_blocks.append(block_params[old_index: index])
            old_index = index

        new_blocks.append(block_params[old_index:])

        # ----------------- return -----------------

        return new_blocks

    def _compute_rev_lateral_params(self, **params):
        if params is None:
            params = self.params
        else:
            params = DotDict(params)

        # reverse lateral params for reversed direction in decoder
        rev_laterals = copy.deepcopy(params.laterals[::-1])

        extractor_index = 0
        for i, params in enumerate(rev_laterals):

            # in case of more than one lateral connection at a node
            if isinstance(params, tuple) or isinstance(params, list):
                rev_laterals[i] = list(rev_laterals[i])[::-1]
                params = list(params)[::-1]
                for k, sub_params in enumerate(params):
                    # symmetrically reverse lateral nets
                    rev_laterals[i][k].block_params = sub_params.block_params[::-1]

                    # change down sampling layers to up sampling layers
                    rev_laterals[i][k].block_params = self._reverse_block_params(rev_laterals[i][k].block_params)

                    shapes = self.lateral_extractors[::-1][extractor_index].shapes[::-1][1:]
                    reshapes = self.lateral_extractors[::-1][extractor_index].reshapes[::-1][1:]

                    rev_laterals[i][k].block_params = self._shift_block_params(rev_laterals[i][k].block_params,
                                                                               shapes,
                                                                               reshapes)
                    rev_laterals[i][k].final_sampling = False
                    rev_laterals[i][k].name_prefix = "rev_" + rev_laterals[i][k].name_prefix

                    extractor_index = extractor_index + 1
            else:
                # symmetrically reverse lateral nets
                rev_laterals[i].block_params = params.block_params[::-1]

                # change down sampling layers to up sampling layers
                rev_laterals[i].block_params = self._reverse_block_params(rev_laterals[i].block_params)

                shapes = self.lateral_extractors[::-1][extractor_index].shapes[::-1][1:]
                reshapes = self.lateral_extractors[::-1][extractor_index].reshapes[::-1][1:]

                rev_laterals[i].block_params = self._shift_block_params(rev_laterals[i].block_params,
                                                                        shapes,
                                                                        reshapes)

                rev_laterals[i].final_sampling = False
                rev_laterals[i].name_prefix = "rev_" + rev_laterals[i].name_prefix

                for block_id, current_block_params in enumerate(rev_laterals[i].block_params):
                    if "conditional_params" in current_block_params:
                        if rev_laterals[i].block_params[block_id].conditional_params is not None:
                            if "name_prefix" in rev_laterals[i].block_params[block_id].conditional_params:
                                rev_laterals[i].block_params[block_id].conditional_params.pop("name_prefix")

                extractor_index = extractor_index + 1

        # ----------------- return -----------------

        return rev_laterals

    def _compute_decoder_params(self):

        """
        Determines decoder parameters according to encoder definition given in internal params.

        Returns
        -------
        DotDict
            decoder params
        """

        decoder_params = copy.deepcopy(self.params.feature_extractor)
        decoder_params.block_params = self._reverse_block_params(decoder_params.block_params[::-1])

        shapes = self.feature_extractor.shapes[::-1][1:]
        reshapes = self.feature_extractor.reshapes[::-1][1:]

        # TODO check more carefully
        if len(shapes) > len(reshapes):
            reshapes = [shapes[0]] + reshapes

        decoder_params.block_params = self._shift_block_params(decoder_params.block_params, shapes, reshapes)
        split_block_params = self._split_decoder_params(decoder_params.block_params)

        temp_params = copy.deepcopy(decoder_params)
        decoder_params = []

        for i, params in enumerate(split_block_params):
            temp_params = copy.deepcopy(temp_params)
            temp_params.block_params = params
            temp_params.name_prefix = f"dec_{i}_"
            decoder_params.append(temp_params)

        if len(decoder_params[-1].block_params) > 0:
            decoder_params[-1].block_params[-1].activation = self.params.output_activation
        else:
            decoder_params[-1].block_params = (DotDict(activation=self.params.output_activation),)

        # ----------------- return -----------------

        return decoder_params

    # ------------------------------------------------------------------------------------------------------------------
    # --------------- build methods for models -------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def _compute_latent_dimension(self):

        """
        Compute latent dimension from collected latent shapes.

        Returns
        -------
        int
            latent dimension
        """

        assert self.decoder is not None

        dimension = 0

        for shape in self.latent_shapes:
            dimension += shape[1]

        return dimension

    def _compute_input_dimension(self):

        """
        Compute dimension of input space.

        Returns
        -------
        int
            dimension of input space
        """

        shape = self.params.input_shape

        if shape is None:
            return None
        else:
            if len(shape) == 1:
                return shape[0]
            else:
                dimension = 1
                for sub_dim in shape:
                    dimension *= sub_dim
                return dimension

    @staticmethod
    def _build_encoder_inputs(input_shape, conditional_input_shape=None):

        """
        Build encoder inputs.

        Parameters
        ----------
        input_shape : tuple
        conditional_input_shape : tuple

        Returns
        -------
        Keras Input Layer
            for input_shape
        Keras Input Layer or None
            for conditional_input_shape
        """

        # data input
        inputs = Input(shape=input_shape, name=f"encoder_input")

        # conditional input
        conditional_inputs = None
        if conditional_input_shape is not None:
            conditional_inputs = Input(shape=conditional_input_shape, name=f"encoder_conditional_input")

        return inputs, conditional_inputs

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

        # remember shapes for decoder definition
        self.latent_shapes.append(K.int_shape(lat))

        # remember lateral outputs
        laterals.append(lat)
        lat_means.append(lat_mean)
        lat_log_vars.append(lat_log_var)

        # ----------------- return -----------------

        return laterals, lat_means, lat_log_vars

    def _build_lateral_networks(self, conditional_inputs, params):

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

        assert self.feature_extractor is not None

        # initialize lists for different outputs
        laterals = []
        lat_means = []
        lat_log_vars = []

        if params.conditional_extractor is not None:
            assert self.conditional_preprocessor is not None

            conditional_inputs_pre = self.conditional_preprocessor(conditional_inputs)
        else:
            conditional_inputs_pre = conditional_inputs

        # build lateral connections at defined features of feature extractor
        for i, lateral in enumerate(self.feature_extractor.laterals):

            self.logger.info(f"Building lateral connection {i}")

            # for more than one lateral connection at current feature
            if isinstance(params.laterals[i], tuple) or isinstance(params.laterals[i], list):
                for k, sub_params in enumerate(params.laterals[i]):

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
                                                                                 params.laterals[i])

        # ----------------- return -----------------

        return laterals, lat_means, lat_log_vars

    def _build_encoder(self, **params):

        """
        Build encoder neural network based on parameters. Model will be available via member "encoder".
        Method also returns model inputs and model outputs that can be used to build the model as a whole
        instead of defining it via its sub models.

        Parameters
        ----------
        params : dict
            definition of encoder model

        Returns
        -------
        tuple
            encoder_inputs, samples, means, log variances
        """

        # ----------------- parameter preparation -----------------

        params = self._prepare_params(**params)

        # ----------------- define encoder input -----------------

        # TODO: add more data input branches

        inputs, conditional_inputs = self._build_encoder_inputs(input_shape=params.input_shape,
                                                                conditional_input_shape=params.conditional_input_shape)

        self.encoder_inputs.append(inputs)

        conditional_inputs_pre = conditional_inputs
        if conditional_inputs is not None:
            self.encoder_inputs.append(conditional_inputs)

            # build global conditional extractor
            if params.conditional_extractor is not None:
                self._build_conditional_preprocessor(conditional_input_shape=params.conditional_input_shape,
                                                     params=params.conditional_extractor)

                conditional_inputs_pre = self.conditional_preprocessor(conditional_inputs)

        # ----------------- build data feature extractor network -----------------

        self.feature_extractor = ForwardNet(**params.feature_extractor)
        self.feature_extractor.forward(inputs, conditional_inputs_pre)

        # ----------------- build lateral networks -----------------

        laterals, lat_means, lat_log_vars = self._build_lateral_networks(conditional_inputs, params)

        # ----------------- define encoder, show summary and plot model -----------------

        outputs = laterals + lat_means + lat_log_vars
        self.encoder = Model(self.encoder_inputs, outputs, name="encoder")

        if params.model_summary:
            self.save_summary_to_file(self.encoder, "encoder")

        if params.plot_model:
            self.save_model_plot_to_file(self.encoder, "encoder", expand_nested=False)
            self.save_model_plot_to_file(self.encoder, "encoder_expanded", expand_nested=True)

        # ----------------- return -----------------

        return self.encoder_inputs, laterals, lat_means, lat_log_vars

    def _get_concatenation_model(self, model_name):

        """
        Iteratively concatenate and flatten latent layers. Used for loss calculation and as output of a more
        convenient flattened encoder model.

        Parameters
        ----------
        model_name

        Returns
        -------
        concatenation model
        """

        inputs = []
        for i, shape in enumerate(self.latent_shapes):
            inputs.append(Input(shape[1:], name=f"{model_name}_input_{i}"))

        all_inputs = Flatten()(inputs[0])
        for input_layer in inputs[1:]:
            all_inputs = concatenate([all_inputs, Flatten()(input_layer)])

        # ----------------- return -----------------

        return Model(inputs, all_inputs, name=model_name)

    def _build_decoder(self, inputs=None, conditional_inputs=None, **params):

        """
        Build decoder neural network based on parameters.

        Parameters
        ----------
        inputs
        conditional_inputs
        params

        Returns
        -------
        decoder output (reconstruction)
        """

        # ----------------- parameter preparation -----------------

        params = self._prepare_params(**params)

        # ----------------- define decoder input -----------------

        if inputs is None:
            inputs = []

            for i, shape in enumerate(self.latent_shapes):
                inputs.append(Input(shape[1:], name=f"{params.name}_decoder_input_{i}"))

        laterals = inputs

        if conditional_inputs is None:
            if params.conditional_input_shape is not None:
                conditional_inputs = Input(params.conditional_input_shape,
                                           name=f"{params.name}_decoder_conditional_input")
                inputs = self.extend_data(inputs, conditional_inputs)
        else:
            inputs = self.extend_data(inputs, conditional_inputs)

        # ----------------- build lateral reconstruction networks -----------------

        if params.conditional_extractor is not None:
            assert self.conditional_preprocessor is not None

            conditional_inputs = self.conditional_preprocessor(conditional_inputs)

        # determine reverse params for lateral and feature extractor models
        rev_lateral_params = self._compute_rev_lateral_params(**params)
        decoder_params = self._compute_decoder_params()
        rev_laterals = []

        lateral_counter = 0
        for i, temp_params in enumerate(rev_lateral_params):

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

        # ----------------- define decoder, show summary and plot model -----------------

        self.decoder = Model(inputs, x, name="decoder")

        if params.model_summary:
            self.save_summary_to_file(self.decoder, "decoder")

        if self.params.plot_model:
            self.save_model_plot_to_file(self.decoder, "decoder", expand_nested=False)
            self.save_model_plot_to_file(self.decoder, "decoder_expanded", expand_nested=True)

        # ----------------- return -----------------

        return x

    def _get_separation_model(self, model_name):

        """
        Separates flattened and concatenated encoder outputs back to original hierarchy.

        Parameters
        ----------
        model_name

        Returns
        -------
        separation model
        """

        latent_dim = self._compute_latent_dimension()

        inputs = Input(shape=(latent_dim,), name=f"{model_name}_input")

        outputs = []
        current_dim = 0
        for i, shape in enumerate(self.latent_shapes):
            next_dim = current_dim + shape[1]
            outputs.append(inputs[:, current_dim:next_dim])
            current_dim = next_dim

        return Model(inputs, outputs, name=model_name)

    def _build_decoder_flattened(self, decoder=None):

        """
        Builds decoder that accepts flattened and concatenated inputs.

        Returns
        -------
        flattened decoder model
        """

        if decoder is None:
            assert self.decoder is not None
            decoder = self.decoder

        flattened_inputs = []
        separated_inputs = []

        flattened_inputs.append(
            Input(shape=(self._compute_latent_dimension()), name=f"decoder_flattened_sample_input")
        )

        separation_model = self._get_separation_model(f"separation_model")

        separated_inputs.append(separation_model(flattened_inputs[-1]))

        if self.params.conditional_input_shape is not None:
            conditional_inputs = Input(self.params.conditional_input_shape,
                                       name=f"decoder_flattened_conditional_input")
            separated_inputs = self.extend_data(separated_inputs, conditional_inputs)
            flattened_inputs = self.extend_data(flattened_inputs, conditional_inputs)

        if len(flattened_inputs) == 1:
            flattened_inputs = flattened_inputs[0]

        outputs = decoder(separated_inputs)

        return Model(flattened_inputs, outputs, name="decoder_flattened")

    @staticmethod
    def _get_kl_loss(z_means, z_log_vars, aggregation=True):
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

        kl_loss = - 0.5 * (z_log_vars - tf.square(z_means) - tf.exp(z_log_vars) + 1)

        if aggregation:
            kl_loss = tf.reduce_mean(kl_loss)

        # ----------------- return -----------------

        return kl_loss

    def _get_kl_loss_model(self,
                           latent_shape,
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

        kl_loss = self._get_kl_loss(z_means, z_log_vars, aggregation)

        # ----------------- return -----------------

        return Model(inputs, kl_loss, name=model_name)

    @staticmethod
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

    def total_correlation(self, z, z_mean, z_log_squared_scale, aggregation=True):
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

        log_qz_prob = self.gaussian_log_density(
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

    def _get_tc_loss_model(self,
                           latent_shape,
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

        tc = - self.total_correlation(z, z_means, z_log_vars, aggregation)

        # ----------------- return -----------------

        return Model(inputs, tc, name=model_name)

    @staticmethod
    def _compute_kernel(x, y, name="mmd_kernel"):

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

    def _compute_mmd(self, x, y, aggregation=True):

        x_kernel = self._compute_kernel(x, x, name="mmd_kernel_x")
        y_kernel = self._compute_kernel(y, y, name="mmd_kernel_y")
        xy_kernel = self._compute_kernel(x, y, name="mmd_kernel_xy")

        if aggregation:
            return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)
        else:
            # TODO: check dimensions

            return x_kernel, y_kernel, xy_kernel

    def _get_mmd_loss_model(self,
                            latent_shape,
                            model_name="mmd_loss_model",
                            aggregation=True):

        inputs = Input(latent_shape[1:], name=f"{model_name}_sample_input")
        true_samples = StandardNormalSamplingLayer()(inputs)

        'calculate mmd loss'
        outputs = self._compute_mmd(true_samples, inputs, aggregation)

        return Model(inputs, outputs, name=model_name)

    @staticmethod
    def _custom_metric(aggregated_layer, pseudo_condition=False):

        """
        Create custom metric from some aggregated layer output. Pseudo condition is only used to
        get rid of pyCharm warnings stating that function does not depend on inputs which is intended here.

        Parameters
        ----------
        aggregated_layer
        pseudo_condition

        Returns
        -------
        metric function
        """

        def metric(true, pred):
            if pseudo_condition:
                print(true - pred)

            return aggregated_layer

        return metric

    def _build_model(self, **params):

        """
        Build entire VAE training model and submodels from specified params.

        Parameters
        ----------
        params

        Returns
        -------
        None
        """

        # ----------------- parameter preparation -----------------

        params = DotDict(params)

        # ----------------- build encoder and decoder -----------------

        self.logger.info("Building encoder.")

        inputs, laterals, lat_means, lat_log_vars = self._build_encoder(**params)

        self.logger.info("Building decoder.")

        if params.build_from_submodels:
            inputs = Input(shape=params.input_shape, name=f"{params.name}_input")

            conditional_inputs = None
            if params.conditional_input_shape is not None:
                conditional_inputs = Input(shape=params.conditional_input_shape,
                                           name=f"{params.name}_conditional_input")
                inputs = [inputs, conditional_inputs]

            codes = self.encoder(inputs)

            laterals = codes[:len(laterals)]
            lat_means = codes[len(laterals):2 * len(laterals)]
            lat_log_vars = codes[2 * len(laterals):]

            self._build_decoder(inputs=None, conditional_inputs=None, **params)
            if params.conditional_input_shape is not None:
                reconstructions = self.decoder([laterals, conditional_inputs])
            else:
                reconstructions = self.decoder(laterals)
        else:
            reconstructions = self._build_decoder(inputs=laterals, **params)

        # ----------------- show summary and plot model without further loss terms -----------------

        self.logger.info("Building core VAE model.")

        self.model = Model(inputs, reconstructions, name=params.name)

        if self.params.plot_model:
            self.save_model_plot_to_file(self.model, f"{params.name}_no_regularizers", expand_nested=True)

        # ----------------- concatenation of decoder outputs to calculate loss terms -----------------

        concat_model_samples = self._get_concatenation_model(model_name="concat_model_samples")
        z = concat_model_samples(laterals)

        concat_model_means = self._get_concatenation_model(model_name="concat_model_means")
        z_means = concat_model_means(lat_means)

        concat_model_log_vars = self._get_concatenation_model(model_name="concat_model_log_vars")
        z_log_vars = concat_model_log_vars(lat_log_vars)

        # ----------------- add flattened encoder submodel for convenience -----------------

        self.logger.info("Building flattened encoder.")

        self.encoder_flattened = Model(inputs, [z, z_means, z_log_vars], name="encoder_flattened")

        # ----------------- add flattened decoder submodel for convenience -----------------

        self.logger.info("Building flattened decoder.")

        self.decoder_flattened = self._build_decoder_flattened()

        # ----------------- definition of training model -----------------

        loss_weights = []

        kl_weight_input = None
        if params.use_kl_loss:
            kl_weight_input = Input((1,), name=f"{params.name}_kl_weight")
            loss_weights.append(kl_weight_input)

        tc_weight_input = None
        if params.use_tc_loss:
            tc_weight_input = Input((1,), name=f"{params.name}_tc_weight")
            loss_weights.append(tc_weight_input)

        mmd_weight_input = None
        if params.use_mmd_loss:
            mmd_weight_input = Input((1,), name=f"{params.name}_mmd_weight")
            loss_weights.append(mmd_weight_input)

        input_dim = self._compute_input_dimension()
        latent_dim = self._compute_latent_dimension()

        assert input_dim > 0

        training_inputs = self.extend_data(inputs, loss_weights)

        self.logger.info("Building training model.")

        self.training_model = Model(training_inputs, reconstructions, name=params.name)

        # ----------------- definition of weighted regularization losses -----------------

        # reconstruction loss
        rec_loss = self.get_loss(self.params.loss, reduction="auto")
        self.append_loss(rec_loss, loss_weight=self.params.loss_weight, loss_name="rec_loss")

        kl_loss = None
        kl_weight = None
        if params.use_kl_loss:
            # kl loss
            kl_loss_model = self._get_kl_loss_model(latent_shape=K.int_shape(z_means), aggregation=True)
            kl_loss = kl_loss_model([z_means, z_log_vars])

            if self.params.kl_importance_sampling:
                kl_loss = latent_dim / input_dim * kl_loss

            kl_weight = tf.math.reduce_mean(kl_weight_input)
            kl_loss_scaled = kl_weight * kl_loss
            self.training_model.add_loss(kl_loss_scaled)
            self.append_metric(self._custom_metric(kl_loss), "kl_loss")

        tc_loss = None
        tc_weight = None
        if params.use_tc_loss:
            # tc loss
            tc_loss_model = self._get_tc_loss_model(latent_shape=K.int_shape(z_means), aggregation=True)
            tc_loss = tc_loss_model([z, z_means, z_log_vars])

            tc_weight = tf.math.reduce_mean(tc_weight_input)
            tc_loss_scaled = tc_weight * tc_loss
            self.training_model.add_loss(tc_loss_scaled)
            self.append_metric(self._custom_metric(tc_loss), "tc_loss")

        mmd_loss = None
        mmd_weight = None
        if params.use_mmd_loss:
            # mmd loss
            mmd_loss_model = self._get_mmd_loss_model(latent_shape=K.int_shape(z_means), aggregation=True)
            mmd_loss = mmd_loss_model(z)

            mmd_weight = tf.math.reduce_mean(mmd_weight_input)
            mmd_loss_scaled = mmd_weight * mmd_loss
            self.training_model.add_loss(mmd_loss_scaled)
            self.append_metric(self._custom_metric(mmd_loss), "mmd_loss")

        # ----------------- definition of other metrics -----------------

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

        self.append_metric(sum_metric(kl_loss, tc_loss, mmd_loss), "unweighted_loss")

        # weight metrics for easy traceback and comparisons
        if kl_weight is not None:
            self.append_metric(self._custom_metric(kl_weight), "kl_weight")

        if tc_weight is not None:
            self.append_metric(self._custom_metric(tc_weight), "tc_weight")

        if mmd_weight is not None:
            self.append_metric(self._custom_metric(mmd_weight), "mmd_weight")

        # ----------------- show summary and plot training model -----------------

        self.training_model.summary()

        if self.params.model_summary:
            self.save_summary_to_file(self.training_model, params.name)

        if self.params.plot_model:
            self.save_model_plot_to_file(self.training_model, params.name)
            self.save_model_plot_to_file(self.training_model, f"{params.name}_expanded", expand_nested=True)

    def _build(self):

        """
        Wrapper method to build VAE training model according to internal params.

        Returns
        -------
        None
        """

        self._build_model(**self.params)

    # -----------------------------------------------------------------------
    # ------- data preparation for annealing callbacks / data generator -----
    # -----------------------------------------------------------------------

    def _prepare_data(self, input_data=None, output_data=None):

        """
        Defines an annealing data generator to be able to deal with dynamic weights for different loss terms.

        Parameters
        ----------
        input_data
        output_data

        Returns
        -------
        annealing data generator
        """

        if isinstance(input_data, tf.keras.utils.Sequence) | ((input_data is not None) & (output_data is not None)):
            # define data generator
            annealing_params = []

            if self.params.use_kl_loss:
                annealing_params.append(self.params.kl_annealing_params)

            if self.params.use_tc_loss:
                annealing_params.append(self.params.tc_annealing_params)

            if self.params.use_mmd_loss:
                annealing_params.append(self.params.mmd_annealing_params)

            data = AnnealingDataGenerator(model_inputs=input_data,
                                          model_outputs=output_data,
                                          batch_size=self.params.batch_size,
                                          annealing_params=annealing_params)

            # append annealing callbacks to list of callbacks
            for callback in data.callbacks:
                self.callbacks.append(callback)

        else:
            data = None

        return data

    # -----------------------------------------------------------------------
    # ------- methods to define submodels from whole model -----------------
    # -----------------------------------------------------------------------

    # -------------------------------------------
    # ----------------- ENCODER -----------------
    # -------------------------------------------

    def _get_encoder_input_layer(self, model=None):

        """
        Get input layer for encoder from whole training model.

        Parameters
        ----------
        model

        Returns
        -------
        encoder inputs
        """

        if model is None:
            model = self.training_model

        inputs = []
        conditionals = []

        for layer in model.get_layer("encoder").layers:
            if "input" in layer.name:
                if "conditional" in layer.name:
                    conditionals.append(model.get_layer("encoder").get_layer(layer.name).input)
                else:
                    inputs.append(model.get_layer("encoder").get_layer(layer.name).input)

        inputs = self.extend_data(inputs, conditionals)

        # ----------------- return -----------------

        if len(inputs) == 1:
            return inputs[0]
        elif len(inputs) == 0:
            return None
        else:
            return inputs

    def _get_encoder_output_layers(self, model=None):

        """
        Get output layers for encoder from whole training model.

        Parameters
        ----------
        model

        Returns
        -------
        encoder outputs
        """

        if model is None:
            model = self.training_model

        latents = []
        means = []
        log_vars = []

        for layer in model.get_layer("encoder").layers:
            if "latent" in layer.name:
                latents.append(layer.name)
            if "means" in layer.name:
                means.append(layer.name)
            if "log_vars" in layer.name:
                log_vars.append(layer.name)

        # TODO: check sorting of latents

        latents = list(sorted(latents))
        means = list(sorted(means))
        log_vars = list(sorted(log_vars))

        latents_layers = [model.get_layer("encoder").get_layer(name).output for name in latents]
        means_layers = [model.get_layer("encoder").get_layer(name).output for name in means]
        log_vars_layers = [model.get_layer("encoder").get_layer(name).output for name in log_vars]

        # ----------------- return -----------------

        return latents_layers + means_layers + log_vars_layers

    def _get_conditional_preprocessor_model(self):
        """
        Get conditional preprocessor from whole model.

        Returns
        -------
        conditional preprocessor
        """

        inputs = self._get_encoder_input_layer()

        if not isinstance(inputs, list):
            return None
        else:
            inputs = self.model.get_layer("encoder").get_layer("conditional_preprocessor").layers[0].input
            outputs = self.model.get_layer("encoder").get_layer("conditional_preprocessor").layers[-1].output
            return Model(inputs, outputs, name="conditional_preprocessor")

    def _get_encoder_model(self):

        """
        Get encoder from whole training model.

        Returns
        -------
        encoder model
        """

        inputs = self._get_encoder_input_layer()
        outputs = self._get_encoder_output_layers()

        # ----------------- return -----------------

        return Model(inputs, outputs, name="encoder")

    # -----------------------------------------------------
    # ----------------- FLATTENED ENCODER -----------------
    # -----------------------------------------------------

    def _get_encoder_flattened_output_layers(self, model=None):

        """
        Get flattened and concatenated output layers for encoder from whole training model, i.e. three layers for
        samples, means and log variances.

        Parameters
        ----------
        model

        Returns
        -------
        flattened and concatenated output layers (unstacked)
        """

        if model is None:
            model = self.training_model

        latents = [model.get_layer("concat_model_samples").layers[-1].output]
        means = [model.get_layer("concat_model_means").layers[-1].output]
        log_vars = [model.get_layer("concat_model_log_vars").layers[-1].output]

        # ----------------- return -----------------

        return latents + means + log_vars

    def _get_encoder_flattened_model(self):

        """
        Get flattened encoder from whole training model.

        Returns
        -------
        flattened encoder model
        """

        assert self.encoder is not None

        inputs = self._get_encoder_input_layer()
        codes = self.encoder(inputs)

        code_length = int(len(codes) / 3)

        laterals = codes[:code_length]
        lat_means = codes[code_length:2 * code_length]
        lat_log_vars = codes[2 * code_length:]

        # concatenation of decoder outputs
        concat_model_samples = self._get_concatenation_model(model_name="concat_model_samples")
        z = concat_model_samples(laterals)

        concat_model_means = self._get_concatenation_model(model_name="concat_model_means")
        z_means = concat_model_means(lat_means)

        concat_model_log_vars = self._get_concatenation_model(model_name="concat_model_log_vars")
        z_log_vars = concat_model_log_vars(lat_log_vars)

        outputs = [z, z_means, z_log_vars]

        # ----------------- return -----------------

        return Model(inputs, outputs, name="encoder_flattened")

    # -------------------------------------------
    # ----------------- DECODER -----------------
    # -------------------------------------------

    def _get_decoder_inputs(self, model=None):

        """
        Get input layers for decoder from whole training model.

        Parameters
        ----------
        model

        Returns
        -------
        decoder inputs
        """

        if model is None:
            model = self.training_model

        latents = []
        conditionals = []

        for layer in model.get_layer("decoder").layers:
            if "decoder_input" in layer.name:
                if "conditional" not in layer.name:
                    latents.append(layer.name)
                else:
                    conditionals.append(layer.name)

        # TODO: check sorting of latents

        latents = list(sorted(latents))
        latents_layers = [model.get_layer("decoder").get_layer(name).input for name in latents]
        conditional_layers = [model.get_layer("decoder").get_layer(name).input for name in conditionals]

        # ----------------- return -----------------

        return self.extend_data(latents_layers, conditional_layers)

    def _get_decoder_model(self):

        """
        Get decoder from whole training model.

        Returns
        -------
        decoder model
        """

        inputs = []
        for i, shape in enumerate(self.latent_shapes):
            inputs.append(Input(shape[1:], name=f"{self.params.name}_decoder_input_{i}"))

        if self.params.conditional_input_shape is not None:
            inputs.append(Input(self.params.conditional_input_shape,
                                name=f"{self.params.name}_decoder_conditional_input"))

        outputs = self.training_model.get_layer("decoder")(inputs)

        # ----------------- return -----------------

        return Model(inputs, outputs, name="decoder")

    def _get_vae_model(self):

        """
        Defines VAE model, i.e., Input -> Encoder -> Decoder -> Output from training model
        -------

        """

        inputs = self._get_encoder_input_layer()
        conditionals = inputs[-1]

        codes = self.encoder(inputs)

        decoder_inputs = self.extend_data(codes, conditionals)
        outputs = self.decoder(decoder_inputs)

        return Model(inputs, outputs, name="vae")

    # ------------------------------------------------------------------
    # ----------------- METHOD TO OBTAIN ALL SUBMODELS -----------------
    # ------------------------------------------------------------------

    # this method will override method in base class and is used in load method

    def _get_models_from_model(self):

        """
        Get all relevant submodels from whole model.
        -------

        """

        self.encoder = self._get_encoder_model()
        self.encoder_flattened = self._get_encoder_flattened_model()

        self.decoder = self._get_decoder_model()
        self.decoder_flattened = self._build_decoder_flattened()

        self.conditional_preprocessor = self._get_conditional_preprocessor_model()
