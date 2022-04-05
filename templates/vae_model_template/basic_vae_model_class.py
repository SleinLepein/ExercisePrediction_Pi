"""
This file implements a Variational Autoencoder class with regularization losses but without any architecture. In this
sense it can be used as a template for own VAE implementations like symmetric architectures.
"""

from dotdict import DotDict
from collections.abc import Iterable
import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, concatenate, Multiply, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from templates.keras_model_template.keras_model_class import KerasModel
from templates.keras_model_template.data_generators.annealing_data_generator import AnnealingDataGenerator
from templates.vae_model_template.defaults.basic_vae_model_defaults import standard_params_basic_vae
from templates.keras_model_template.nn.layers.recursive_flatten import RecursiveFlatten
from templates.keras_model_template.nn.layers.recursive_reshape import RecursiveReshape
from templates.keras_model_template.nn.layers.losses import KLLoss, TCLoss, MMDLoss

DEFAULT_MODELS_PATH = "../"


class VariationalAutoencoderModel(KerasModel):

    def __init__(self, **params_dict):

        super().__init__(**params_dict)

        # update standard parameters with given parameters
        self.update_params(self.input_params, standard_params_basic_vae)

        # compilation must be done in derived class

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

    def _initialize(self):

        """
        Initialize class members.

        Returns
        -------
        None
        """

        # possible submodels
        self.encoder = None
        self.decoder = None

        # model inputs and model outputs
        self.encoder_inputs = []
        self.latent_dims = []
        self.encoder_outputs = []
        self.latent_dimension = None

        self.encoder_means = []
        self.encoder_log_vars = []

        self.decoder_inputs = []

        self.training_model_inputs = []
        self.training_model_outputs = []

        # specialization for child classes
        self._specialize_init()

    @staticmethod
    def _prepare_params(**params) -> DotDict:
        """
        Implement in child class.

        Returns
        -------

        """
        return DotDict(params)

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
        # define data generator
        annealing_params = []

        if self.params.use_kl_loss & self.params.schedule_kl_loss:
            annealing_params.append(self.params.kl_annealing_params)

        if self.params.use_tc_loss & self.params.schedule_tc_loss:
            annealing_params.append(self.params.tc_annealing_params)

        if self.params.use_mmd_loss & self.params.schedule_mmd_loss:
            annealing_params.append(self.params.mmd_annealing_params)

        if len(annealing_params) == 0:
            if isinstance(input_data, tf.keras.utils.Sequence):
                return input_data
            elif (input_data is not None) & (output_data is not None):
                return input_data, output_data
        else:
            if isinstance(input_data, tf.keras.utils.Sequence) | ((input_data is not None) & (output_data is not None)):
                gen_params = DotDict(batch_size=self.params.batch_size,
                                     annealing_params=annealing_params)

                data = AnnealingDataGenerator(model_inputs=input_data,
                                              model_outputs=output_data,
                                              **gen_params)

                # append annealing callbacks to list of callbacks
                for callback in data.callbacks:
                    self.callbacks.append(callback)

                return data
            else:
                return None

    # ------------------------------------------------------------------------------------------------------------------
    # --------------- build methods for models -------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _build_encoder_inputs(input_shapes: Iterable,
                              name: str = "encoder") -> list:

        """
        Build encoder inputs.

        Parameters
        ----------
        input_shapes : Iterable

        Returns
        -------
        Iterable of Keras Input Layers / None
            for input and conditional_input
        """

        inputs = []
        for i, input_shape in enumerate(input_shapes):
            if i == 0:
                # data input
                inputs.append(Input(shape=input_shape, name=f"{name}_input"))
            else:
                # conditional input
                inputs.append(Input(shape=input_shape, name=f"{name}_conditional_input_{i-1}"))

        return inputs

    def _compute_latents(self,
                         inputs: Iterable,
                         **params) -> (list, list, list):

        """
        Must be implemented in child class.
        Essentially, this is the encoder model.

        Parameters
        ----------
        inputs : list
            list of input tensors
        params : dict

        Returns
        -------
        (list, list, list)
            list of latent codes tensors, list of latent means tensors, list of latent log variances tensors
        """

        # this is just a pseudo implementation
        print(f"Do fancy calculation with inputs {inputs} and {params}")
        self._prepare_params(**params)

        latents, latents_means, latents_log_vars = [], [], []
        return latents, latents_means, latents_log_vars

    def _compute_latent_dimension(self) -> int:
        """
        Compute latent dimension by summing up latent dimensions for each hierarchy.

        Returns
        -------
        int
        """
        # encoder must already be build, because then output dimension can be determined
        assert isinstance(self.encoder, Model)

        return self.latent_dimension

    def _build_encoder(self, **params) -> None:

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
        None
        """

        # ----------------- parameter preparation -----------------

        params = self._prepare_params(**params)

        # ----------------- define encoder input -----------------

        self.encoder_inputs = self._build_encoder_inputs(input_shapes=params.input_shapes)

        latents, lat_means, lat_log_vars = self._compute_latents(self.encoder_inputs, **params)

        for latent in latents:
            # remember shapes for possible decoder definition
            self.latent_dims.append(K.int_shape(latent)[1:])

        # now recursively concatenate / flatten each list
        output_names = ["latents", "means", "log_vars"]
        for i, output in enumerate([latents, lat_means, lat_log_vars]):
            self.encoder_outputs.append(RecursiveFlatten(name=output_names[i])(output))

        self.latent_dimension = K.int_shape(self.encoder_outputs[0])[1]

        # ----------------- define encoder, show summary and plot model -----------------

        self.encoder = Model(self.encoder_inputs, self.encoder_outputs, name="encoder")

        if params.model_summary:
            self.save_summary_to_file(self.encoder, "encoder")

        if params.plot_model:
            self.save_model_plot_to_file(self.encoder, "encoder", expand_nested=False)
            self.save_model_plot_to_file(self.encoder, "encoder_expanded", expand_nested=True)

    def _build_decoder_inputs(self, input_shapes: Iterable) -> list:

        """
        Build encoder inputs.

        Parameters
        ----------
        input_shapes : Iterable

        Returns
        -------
        list
            list of Keras input layers
        """

        assert self.latent_dimension is not None

        inputs = []
        for i, input_shape in enumerate(input_shapes):
            if i == 0:
                # latent data input
                inputs.append(Input(shape=(self.latent_dimension,), name=f"decoder_input"))
            else:
                # conditional input
                inputs.append(Input(shape=input_shape, name=f"decoder_conditional_input_{i-1}"))

        return inputs

    def _compute_reconstructions(self,
                                 inputs: list,
                                 conditional_inputs: list,
                                 **params) -> list:

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
            list of tensors needed for reconstruction loss
        """

        # this is just a pseudo implementation
        print(f"Do fancy calculation with inputs {inputs} and {params}")
        self._prepare_params(**params)

        reconstructions = []
        return reconstructions

    def _build_decoder(self, **params):

        """
        Build decoder neural network based on parameters.

        Parameters
        ----------
        params

        Returns
        -------
        decoder output (reconstruction)
        """

        # ----------------- parameter preparation -----------------

        params = self._prepare_params(**params)

        # ----------------- define encoder input -----------------

        self.decoder_inputs = self._build_decoder_inputs(input_shapes=params.input_shapes)
        inputs = list(self.decoder_inputs)

        latent_inputs = inputs[0]

        if len(inputs) > 1:
            conditional_inputs = inputs[1:]
        else:
            conditional_inputs = None

        laterals = RecursiveReshape(output_shapes=self.latent_dims, name="laterals")(latent_inputs)

        self.decoder_outputs = self._compute_reconstructions(laterals, conditional_inputs, **params)

        # ----------------- define decoder, show summary and plot model -----------------

        self.decoder = Model(self.decoder_inputs, self.decoder_outputs, name="decoder")

        if params.model_summary:
            self.save_summary_to_file(self.decoder, "decoder")

        if self.params.plot_model:
            self.save_model_plot_to_file(self.decoder, "decoder", expand_nested=False)
            self.save_model_plot_to_file(self.decoder, "decoder_expanded", expand_nested=True)

    # ------------------------------------------------------------------------------------------------------------------
    # --------------- build ----------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _get_regularization_weights(**params) -> DotDict:
        params = DotDict(params)
        loss_weights = {}

        if params.use_kl_loss:
            if params.schedule_kl_loss:
                loss_weights["kl_loss"] = Input((1,), name=f"{params.name}_kl_weight")
            else:
                loss_weights["kl_loss"] = params.kl_loss_weight

        if params.use_tc_loss:
            if params.schedule_tc_loss:
                loss_weights["tc_loss"] = Input((1,), name=f"{params.name}_tc_weight")
            else:
                loss_weights["tc_loss"] = params.tc_loss_weight

        if params.use_mmd_loss:
            if params.schedule_mmd_loss:
                loss_weights["mmd_loss"] = Input((1,), name=f"{params.name}_mmd_weight")
            else:
                loss_weights["mmd_loss"] = params.mmd_loss_weight

        return DotDict(loss_weights)

    @staticmethod
    def _get_dynamic_weights(regularization_weights) -> DotDict:
        """
        Extract dynamic weights from all regularization weights.

        Parameters
        ----------
        regularization_weights : dict

        Returns
        -------
        DotDict
        """
        weights = {}
        for weight in regularization_weights:
            if tf.is_tensor(regularization_weights[weight]):
                weights[weight] = regularization_weights[weight]
        return DotDict(weights)

    def _add_regularization_losses(self,
                                   z,
                                   z_means,
                                   z_log_vars,
                                   loss_weights: dict = None) -> DotDict:

        # ----------------- parameter preparation -----------------
        if loss_weights is None:
            loss_weights = {}

        losses = {}
        for str_loss in loss_weights:
            if str_loss == "kl_loss":
                loss = KLLoss()([z_means, z_log_vars])
            elif str_loss == "tc_loss":
                loss = TCLoss()([z, z_means, z_log_vars])
            elif str_loss == "mmd_loss":
                loss = MMDLoss()(z)
            else:
                loss = None
            losses[str_loss] = loss

            if tf.is_tensor(loss_weights[str_loss]):
                weight = Lambda(lambda x: tf.reduce_mean(x))(loss_weights[str_loss])
                self.training_model.add_metric(weight, f"{str_loss}_weight")
                loss_scaled = Multiply()([weight, loss])
            else:
                loss_scaled = Lambda(lambda x: loss_weights[str_loss] * x)(loss)

            self.training_model.add_loss(loss_scaled)
            self.training_model.add_metric(loss, name=str_loss)

        return DotDict(losses)

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
        pass

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
        self._build_encoder(**params)

        model_inputs = self._build_encoder_inputs(input_shapes=params.input_shapes, name="vae")
        latents, lat_means, lat_log_vars = self.encoder(model_inputs)

        self.logger.info("Building decoder.")
        self._build_decoder(**params)

        decoder_inputs = [latents] + [model_inputs[i] for i in range(1, len(model_inputs))]
        reconstructions = self.decoder(decoder_inputs)

        # ----------------- show summary and plot model without further loss terms -----------------

        self.logger.info("Building core VAE model.")
        self.model = Model(model_inputs, reconstructions, name=params.name)

        if self.params.plot_model:
            self.save_model_plot_to_file(self.model, f"{params.name}_no_regularizers", expand_nested=False)
            self.save_model_plot_to_file(self.model, f"{params.name}_no_regularizers_expanded", expand_nested=True)

        # ----------------- definition of training model -----------------

        self.logger.info("Building training model.")

        regularization_weights = self._get_regularization_weights(**params)
        dynamic_weights = self._get_dynamic_weights(regularization_weights)

        training_inputs = self.extend_data(model_inputs, [dynamic_weights[item] for item in dynamic_weights])

        self.training_model = Model(training_inputs, reconstructions, name=params.name)

        # ----------------- definition of weighted regularization losses -----------------

        # reconstruction loss
        rec_loss = self.get_loss(self.params.loss, reduction="auto")

        self.append_loss(rec_loss, loss_weight=self.params.loss_weight, loss_name="rec_loss")

        reg_losses = self._add_regularization_losses(latents, lat_means, lat_log_vars, regularization_weights)

        self._add_further_metrics(latents,
                                  lat_means,
                                  lat_log_vars,
                                  rec_loss,
                                  regularization_weights,
                                  reg_losses,
                                  **params)

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

            print("encoder", layer.name)

            if "latent" == layer.name:
                latents.append(layer.name)
            if "means" == layer.name:
                means.append(layer.name)
            if "log_vars" == layer.name:
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

    def _get_encoder_model(self) -> Model:

        """
        Get encoder from whole training model.

        Returns
        -------
        encoder model
        """

        inputs = self._get_encoder_input_layer()

        print(inputs)

        outputs = self._get_encoder_output_layers()

        # # now flatten each list
        # concat_models = {}
        # model_names = ["concat_model_latents", "concat_model_means", "concat_model_log_vars"]
        #
        # flattened_outputs = []
        # for i, output in enumerate(outputs):
        #     concat_models[i] = self._get_concatenation_model(model_names[i])
        #     flattened_outputs.append(concat_models[i](output))
        #
        # print(flattened_outputs)

        # ----------------- return -----------------

        return Model(inputs, outputs, name="encoder")

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

            print("decoder", layer.name)

            if "decoder_input" == layer.name:
                latents.append(layer.name)
            if "decoder_conditional_input" in layer.name:
                conditionals.append(layer.name)

        latents_layers = [model.get_layer("decoder").get_layer(name).input for name in latents]
        conditional_layers = [model.get_layer("decoder").get_layer(name).input for name in conditionals]

        # ----------------- return -----------------

        return latents_layers + conditional_layers

    def _get_decoder_model(self) -> Model:

        """
        Get decoder from whole training model.

        Returns
        -------
        decoder model
        """

        # inputs = []
        # for i, shape in enumerate(self.latent_dims):
        #     inputs.append(Input(shape[1:], name=f"{self.params.name}_decoder_input_{i}"))
        #
        # if self.params.conditional_input_shape is not None:
        #     inputs.append(Input(self.params.conditional_input_shape,
        #                         name=f"{self.params.name}_decoder_conditional_input"))

        inputs = self._get_decoder_inputs()
        outputs = self.training_model.get_layer("decoder")(inputs)

        # ----------------- return -----------------

        return Model(inputs, outputs, name="decoder")

    def _get_vae_model(self, sample=False) -> Model:

        """
        Defines VAE model, i.e., Input -> Encoder -> Decoder -> Output from training model
        -------

        """

        assert isinstance(self.encoder, Model)
        assert isinstance(self.decoder, Model)

        # get model inputs
        inputs = self._get_encoder_input_layer()
        codes = self.encoder(inputs)

        # TODO: adapt to more conditions
        if sample:
            outputs = self.decoder(self.extend_data(codes[0], inputs[-1]))
        else:
            outputs = self.decoder(self.extend_data(codes[1], inputs[-1]))

        return Model(inputs, outputs, name="vae")

    def _get_models_from_model(self):

        """
        Get all relevant submodels from whole model.
        -------

        """

        self.encoder = self._get_encoder_model()
        self.decoder = self._get_decoder_model()
        # self.conditional_preprocessor = self._get_conditional_preprocessor_model()
        self.sample_model = self._get_vae_model(sample=True)
        self.mean_model = self._get_vae_model(sample=False)

        self.compile(self.sample_model)
        self.compile(self.mean_model)

        self.model = self.mean_model
