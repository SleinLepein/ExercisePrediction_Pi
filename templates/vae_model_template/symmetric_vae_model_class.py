from pprint import pprint

from dotdict import DotDict
import copy

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Add
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from templates.vae_model_template.basic_vae_model_class import VariationalAutoencoderModel
from templates.keras_model_template.nn.CombinatorFunction import combinator
from templates.keras_model_template.nn.ForwardNet import ForwardNet
from templates.vae_model_template.defaults.symm_vae_model_defaults import standard_params_vae

DEFAULT_MODELS_PATH = "../"


class SymmetricVAE(VariationalAutoencoderModel):

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

        # build feature extractor
        self.feature_extractor.forward(data_inputs, conditional_inputs_pre)

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

        laterals = inputs
        if conditional_inputs is not None:
            conditional_inputs = conditional_inputs[0]

        # ----------------- conditional preprocessing -----------------

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
        list_reg_losses = list(reg_losses.values())
        if len(list_reg_losses) > 1:
            sum_reg_loss = Add()(list_reg_losses)
            self.training_model.add_metric(sum_reg_loss, "unweighted_reg_loss")


if __name__ == '__main__':
    example_VAE = SymmetricVAE()
