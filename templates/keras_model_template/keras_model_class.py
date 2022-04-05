"""
This file implements a Wrapper Class for Keras Models based on the Model Template wrapper class. In particular,
various callbacks for training are provided.
"""

import os
import io
import datetime as dt
import gc

gc.collect()

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

from templates.model_template.model_class import ModelClass
from templates.keras_model_template.defaults.keras_model_defaults import standard_params_keras
from templates.keras_model_template.helpers.data_handling import extend_data
from templates.keras_model_template.nn.LossFunctions import get_loss
from templates.keras_model_template.callbacks.learning_rate_callback import LearningRateCallback


# # allow growth of GPU memory
# config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
# sess = tf.compat.v1.Session(config=config)
#
# # do not use eager execution
# tf.compat.v1.disable_v2_behavior()

class KerasModel(ModelClass):
    """
    This class is derived from the base model class and provides functionality that every Keras model should have.

    Particularly, save and load methods are defined as well as methods to compile the model, to use callbacks and
    how to train the model with given training and test data.
    """

    def __init__(self, **params_dict):

        """
        Initializes an instance of a Keras model class according to specified keyword parameters.

        Parameters
        ----------
        params_dict

        ----------------------------------------------------------------------------------------------------------------
        ------------------------------------- PARAMETER DETAILS --------------------------------------------------------
        ----------------------------------------------------------------------------------------------------------------

        -------------------------- meta --------------------------
        save_folder: folder path for meta data and model checkpoints
        name: name for the model to be used for instance during generation of file names
        plot_model: whether to save graphviz / pydot model visualizations to file, default: True
        model_summary: whether to save model summary to file, default: True

        -------------------------- optimizer --------------------------
        optimizer: which optimizer to use, default: "adam"
        lr: start value of learning rate, default: 0.75 * 1e-2
        epochs: number of epochs for training, default: 1
        batch_size: training batch size, default: 32

        -------------------------- callbacks --------------------------
        --- checkpoint ---
        checkpoint_monitor: which metric to use for checkpoint callback, default: val_loss
        checkpoint_save_best_only: default: True
        save_weights_only: default: True
        save_latest: default: True

        --- early stopping ---
        early_stopping_monitor: which metric to use for early stopping checkpoint, default: None
        early_stopping_patience: default: 10

        --- tensorboard ---
        tensorboard: whether to use TensorBoard, default: True
        tensorboard_update_freq: default: "epoch"
        tensorboard_write_graph: default: False
        tensorboard_write_images: default: False
        tensorboard_histogram_freq: default: 0 (i.e. None),

        --- learning rate scheduler ---
        lr_scheduler: dictionary defining learning rate scheduler, default: None
                      type: string for which kind of scheduler to use (currently only "exponential_decay")
                      decay_steps: integer
                      decay_rate: float between 0 and 1

        --- learning rate reduce on plateau (if not scheduling) ---
        lr_reducer_monitor:
        lr_reducer_factor: , default: 0.95
        lr_reducer_cooldown: , default: 0
        lr_reducer_patience: how long to wait for, default: 1
        lr_reducer_verbose: verbosity of reducer callback, default: 1
        lr_reducer_min_lr: minimum learning rate ; default: 0.5e-6

        ----------------------------------------------------------------------------------------------------------------
        ------------------------------------- IMPORTANT MEMBERS --------------------------------------------------------
        ----------------------------------------------------------------------------------------------------------------

        -------------------------- models --------------------------
        training_model: should be used in child classes for Keras model that will be used during training

        -------------------------- losses and metrics --------------------------
        losses: list for loss functions to use
        losses_weights: list for corresponding loss weights
        metrics: list for metrics to be used for evaluation

        last_loss_history: list of results of last training
        loss_histories: results of all trainings (currently not really implemented)

        -------------------------- callbacks --------------------------
        callbacks: list for callbacks to be used during training

        -------------------------- paths --------------------------
        model_summary_path: sub folder "model_summary" of save folder
        training_results_path: sub folder "training_results" of save folder

        ----------------------------------------------------------------------------------------------------------------
        ------------------------------------- NOTES ON USAGE -----------------------------------------------------------
        ----------------------------------------------------------------------------------------------------------------
        Implement a new and specific class for the intended Keras Model family by implementing the class member
        training_model, e.g.:


            from dotdict import DotDict
            from tensorflow.keras import Input, Model
            from templates.keras_model_template.keras_model_class import KerasModel

            from src.YourStuff import your_input_shape, your_build_method


            class YourKerasModel(KerasModel):

                YourStandardParams = DotDict(your_param_1=1,
                                             your_param_2=True,
                                             loss="mse")

                def __init__(self, **params_dict):
                    super().__init__(**params_dict)

                    # update internal params based on given input params and given standard params
                    self.update_params(self.input_params, self.YourStandardParams)

                    # build model according to specification in params, build method is implemented below
                    self._build()

                    # compile model according to specification in params,
                    # compile method is already implemented in keras model
                    self.compile()

                def _build(self):
                    # input definition
                    inputs = Input(your_input_shape)

                    # build up network from inputs to get outputs
                    outputs = your_build_method(inputs)

                    # define training model
                    self.training_model = Model(inputs, outputs)

                    # define loss
                    rec_loss = self.get_loss(self.params.loss, reduction="auto")
                    self.append_loss(rec_loss, loss_weight=1, loss_name="loss", add_metric=False)

        The model can be saved with the help of the save method, e.g.,

            model_instance = YourKerasModel(**params_dict)
            model_instance.save(folder_path, model_name)

        The model will be either saved to .h5 or to .ckpt data as specified by the parameter save_weights_only.
        """

        super().__init__(**params_dict)

        # update standard parameters with given parameters
        self.update_params(self.input_params, standard_params_keras)

        # initialize class members
        self.losses = []
        self.losses_weights = []
        self.metrics = []

        self.callbacks = []
        self.checkpoint_monitor = self.params.checkpoint_monitor
        self.early_stopping_monitor = self.params.early_stopping_monitor
        self.lr_reducer_monitor = self.params.lr_reducer_monitor

        self.lr = None
        self.lr_callback = None

        self.last_loss_history = None
        self.loss_histories = []
        self.lr_histories = []

        self.save_folder = self.params.save_folder
        self.model_summary_path = os.path.join(self.save_folder, "model_summary")
        self.training_results_path = os.path.join(self.save_folder, "training_results")

        now_date = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_log_path = os.path.join(self.training_results_path, "logs", now_date)
        self.file_writer = None

        self.training_model = None

        # make paths
        if not os.path.exists(self.model_summary_path):
            os.mkdir(self.model_summary_path)

        if not os.path.exists(self.training_results_path):
            os.mkdir(self.training_results_path)

    # ------------------------------------------------------------------------------------------------------------------
    # --------------- convenience --------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def extend_data(first_data, second_data) -> list:
        return extend_data(first_data, second_data)

    # ------------------------------------------------------------------------------------------------------------------
    # --------------- save and load functionality ----------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def save_summary_to_file(self,
                             model: tf.keras.models.Model,
                             model_name: str) -> None:

        """
        Saves summary for model to a selected save folder using given model name for creation of file name.

        Parameters
        ----------
        model
        model_name

        Returns
        -------
        None
        """
        folder_path = self.model_summary_path
        file_name = model_name + "_summary.txt"

        self.print_to_file(model.summary, folder_path, file_name)

    def save_model_plot_to_file(self,
                                model: tf.keras.models.Model,
                                model_name: str,
                                show_shapes: bool = True,
                                expand_nested: bool = False) -> None:

        """
        Saves a graphviz / pydot visualization of the given model to a png file.

        Parameters
        ----------
        model : : tf.keras.models.Model
        model_name : str
        show_shapes : bool
        expand_nested : bool

        Returns
        -------
        None
        """
        
        save_path = os.path.join(self.model_summary_path, model_name + ".png")
        plot_model(model, to_file=save_path, show_shapes=show_shapes, expand_nested=expand_nested)

    def save(self,
             folder_path: str,
             model_name: str) -> None:

        """
        Save internal training model to file.

        Parameters
        ----------
        folder_path : str
        model_name : str

        Returns
        -------
        None
        """

        # TODO: change behaviour
        assert isinstance(self.training_model, tf.keras.models.Model)

        if self.params.save_weights_only:
            save_path = os.path.join(folder_path, model_name + ".ckpt")
            self.training_model.save_weights(save_path)
        else:
            save_path = os.path.join(folder_path, model_name + ".ckpt")
            self.training_model.save_weights(save_path)

            save_path = os.path.join(folder_path, model_name + ".h5")
            self.training_model.save(save_path, save_format="tf")

    def _get_models_from_model(self) -> None:

        """
        Needs implementation in child class how to define possibly relevant submodels from training model.
        This could be needed after loading training model from file if only weights have been saved.

        -------

        """

        pass

    @staticmethod
    def _guess_epoch_from_model_name(model_name: str) -> int:
        """
        Guess epoch from model name to set as initial_epoch for further training

        Parameters
        ----------
        model_name : str

        Returns
        -------
        int
            initial epoch
        """

        try:
            return int(model_name.split("_")[0])
        except ValueError:
            return 0

    def load(self,
             folder_path: str,
             model_name: str) -> None:

        """
        Load whole model from file and define submodels.

        Parameters
        ----------
        folder_path : str
        model_name : str

        Returns
        -------
        None
        """
        # TODO: change behaviour as in save

        save_path = os.path.join(folder_path, model_name)
        if model_name.endswith("ckpt") | model_name.endswith(".h5"):
            self.training_model.load_weights(save_path)
        else:
            self.training_model = load_model(save_path)

        assert isinstance(self.training_model, tf.keras.models.Model)

        # get submodels
        self._get_models_from_model()

        # get initial epoch
        self.params.initial_epoch = self._guess_epoch_from_model_name(model_name)

        # get training results
        self._load_training_results(self.training_results_path)

    # ------------------------------------------------------------------------------------------------------------------
    # --------------- loss functions and metrics -----------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def get_loss(loss_as_string: str,
                 reduction: str = "auto"):
        """
        Get Keras Loss function for a string identifier.

        Parameters
        ----------
        loss_as_string : str
        reduction : str

        Returns
        -------
        loss function
        """

        return get_loss(loss_as_string, reduction)

    def append_metric(self,
                      metric_function,
                      metric_name: str = "metric") -> None:

        """
        Add metric function to model metrics and use given metric name.

        Parameters
        ----------
        metric_function
        metric_name : str
            name that should be used for given metric function
        Returns
        -------
        None
        """

        metric_function.__name__ = metric_name
        self.metrics.append(metric_function)

    def append_loss(self,
                    loss,
                    loss_weight: float = 1.0,
                    loss_name: str = "custom",
                    add_metric: bool = True) -> None:

        """
        Append a loss function with weight to internal list of loss functions and add loss to model metrics.

        Parameters
        ----------
        loss
            string identifier of loss function or loss function
        loss_weight
        loss_name
        add_metric

        Returns
        -------
        None
        """

        # TODO: improve error handling

        # convert loss given as string identifier to corresponding loss function
        if isinstance(loss, str):
            loss_as_string = loss
            loss = self.get_loss(loss, reduction="auto")
        else:
            loss_as_string = loss_name

        # TODO: do some checks before appending
        # TODO: check loss weights (do not use loss weights in compile, it is already part of loss definition)

        # append loss weight
        self.losses_weights.append(loss_weight)

        # define and append loss
        def loss_function(true, pred):
            return loss_weight * loss(true, pred)

        def loss_metric(true, pred):
            return loss(true, pred)

        # loss_function.__name__ = loss_name
        self.losses.append(loss_function)

        # add metric
        if add_metric:
            self.append_metric(loss_metric, f'{loss_as_string}')

    # ------------------------------------------------------------------------------------------------------------------
    # --------------- scheduling and callbacks -------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _plot_to_tf_image(figure,
                          png_save_function,
                          convert_to_png_method=None):

        """
        Converts a figure / plot to a tensorflow image to be able to display it in TensorBoard. Corresponding to figure,
        e.g. matplotlib figure or bokeh plot, a png_save_function to save plot as png must be provided in child class.
        In the case of Bokeh with Holoviews a suitable choice is: holoviews.save(figure, path, fmt="png").

        Parameters
        ----------
        figure
        png_save_function

        Returns
        -------
        tensorflow image
        """

        buf = None
        try:
            # Save the plot to a PNG in memory.
            buf = io.BytesIO()

            if convert_to_png_method is None:
                png_save_function(figure, buf)
            else:
                pli_img = convert_to_png_method(figure)
                pli_img.save(buf, format="png")

            buf.seek(0)
            img = buf.getvalue()

            # Convert PNG buffer to TF image
            image = tf.image.decode_png(img, channels=4)

            # Add the batch dimension
            image = tf.expand_dims(image, 0)

            return image
        finally:
            if buf is not None:
                buf.close()
            del buf
            gc.collect()

    def _add_lr_info_callback(self) -> None:

        """
        Add callback that prints current learning rate to console during training.

        Returns
        -------
        None
        """

        if self.params.lr_info is not None:
            if self.params.lr_info:
                self.lr_callback = LearningRateCallback()
                self.callbacks.append(self.lr_callback)

    def _add_latest_checkpoint_callback(self) -> None:

        """
        Add callback to always save model of latest epoch to file during training.

        Returns
        -------
        None
        """

        if self.params.save_latest:
            file_name_latest = "model_latest"
            if self.params.save_weights_only:
                save_path_latest = os.path.join(self.params.save_folder, file_name_latest + ".ckpt")
            else:
                save_path_latest = os.path.join(self.params.save_folder, file_name_latest + ".h5")

            checkpoint_latest = ModelCheckpoint(filepath=save_path_latest,
                                                verbose=0,
                                                save_weights_only=self.params.save_weights_only,
                                                save_best_only=False)
            self.callbacks.append(checkpoint_latest)

    def _add_checkpoint_callback(self) -> None:

        """
        Add callback that saves models to file based on their validation loss / checkpoint monitor metric
        during training.

        Returns
        -------
        None
        """

        if self.checkpoint_monitor is not None:
            if self.checkpoint_monitor == 'val_loss':
                checkpoint_string = "{val_loss:.4f}"
            else:
                checkpoint_string = "val_{val_loss:.2f}_mon_{" + str(self.checkpoint_monitor) + ":.2f}"

            file_name = "{epoch:02d}_" + checkpoint_string

            if self.params.save_weights_only:
                save_path = os.path.join(self.params.save_folder, file_name + ".ckpt")
            else:
                save_path = os.path.join(self.params.save_folder, file_name + ".h5")

            checkpoint = ModelCheckpoint(filepath=save_path,
                                         monitor=self.checkpoint_monitor,
                                         verbose=1,
                                         save_weights_only=self.params.save_weights_only,
                                         save_best_only=self.params.checkpoint_save_best_only)
            self.callbacks.append(checkpoint)

    def _add_lr_reducer_callback(self) -> None:

        """
        Add callback that reduces learning rate if validation loss / lr reducer monitor metric does not decrease.

        Returns
        -------
        None
        """

        if self.lr_reducer_monitor is not None:
            lr_reducer = ReduceLROnPlateau(monitor=self.lr_reducer_monitor,
                                           factor=self.params.lr_reducer_factor,
                                           cooldown=self.params.lr_reducer_cooldown,
                                           patience=self.params.lr_reducer_patience,
                                           verbose=self.params.lr_reducer_verbose,
                                           min_lr=self.params.lr_reducer_min_lr)
            self.callbacks.append(lr_reducer)

    def _add_early_stopping_callback(self) -> None:

        """
        Add callback that stops training if validation loss / early stopping monitor metric does not decrease.

        Returns
        -------
        None
        """

        if self.early_stopping_monitor is not None:
            early_stopping = EarlyStopping(monitor=self.early_stopping_monitor,
                                           patience=self.params.early_stopping_patience)
            self.callbacks.append(early_stopping)

    def _update_tensorboard_logging_path(self) -> None:
        """
        Update TensorBoard logging path to current datetime.

        Returns
        -------
        None
        """

        now_date = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_log_path = os.path.join(self.training_results_path, "logs", now_date)

    def _add_tensorboard_callback(self) -> None:

        """
        Add callback for tensorboard to track training metrics.

        Returns
        -------
        None
        """

        log_dir = os.path.join(self.training_results_path, "logs")
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        if self.params.tensorboard:
            self._update_tensorboard_logging_path()
            tensorboard = TensorBoard(log_dir=self.tensorboard_log_path,
                                      update_freq=self.params.tensorboard_update_freq,
                                      write_graph=self.params.tensorboard_write_graph,
                                      write_images=self.params.tensorboard_write_images,
                                      histogram_freq=self.params.tensorboard_histogram_freq,
                                      # fix for calling fit multiple times
                                      profile_batch=100000000)
            self.callbacks.append(tensorboard)

    def _add_custom_file_writer(self) -> None:

        """
        Add a custom file writer for TensorBoard.

        Returns
        -------
        None
        """

        if self.params.custom_file_writer:
            if not os.path.exists(self.tensorboard_log_path):
                os.mkdir(self.tensorboard_log_path)

            self.file_writer = tf.summary.create_file_writer(os.path.join(self.tensorboard_log_path, "custom"))

    def _add_epoch_end_lambda_callback(self, function) -> None:

        """
        Add a callback for provided function to be executed at epoch end. Can be used in child classes, for instance, to
        regularly generate plots with current training model in TensorBoard.

        Parameters
        ----------
        function

        Returns
        -------
        None
        """

        if function is not None:
            callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=function)
            self.callbacks.append(callback)

    def _determine_lr_scheduler(self) -> None:

        """
        Determine learning rate scheduler based on given internal class parameters.

        Returns
        -------
        None
        """

        # TODO: Add other scheduling types

        if self.params.lr_scheduler is not None:
            if self.params.lr_scheduler.type == "exponential_decay":
                self.lr = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=self.params.lr,
                    decay_steps=self.params.lr_scheduler.decay_steps,
                    decay_rate=self.params.lr_scheduler.decay_rate,
                    staircase=False,
                    name=None
                )
        else:
            self.lr = self.params.lr

    def _define_standard_callbacks(self) -> None:

        """
        Define callbacks like early stopping, learning rate reducer, checkpoint according to specified internal
        parameters.

        Returns
        -------
        None
        """

        # stop if validation loss / early stopping monitor metric does not decrease over some time
        self._add_early_stopping_callback()

        # reduce learning rate if validation loss / lr reducer monitor metric does not improve over some time
        self._add_lr_reducer_callback()

        # checkpoint callback to save individual models per epoch
        self._add_checkpoint_callback()

        # checkpoint callback to always save latest model to file
        self._add_latest_checkpoint_callback()

        # learning rate callback for traceback
        self._add_lr_info_callback()

        # tensorboard callback
        self._add_tensorboard_callback()

    def _define_data_dependent_callbacks(self,
                                         train_input_data,
                                         train_output_data,
                                         test_input_data,
                                         test_output_data):

        """
        Must be overridden in child class to add a callback which depends on training and / or test data.

        Parameters
        ----------
        train_input_data
        train_output_data
        test_input_data
        test_output_data

        Returns
        -------
        None
        """

        pass

    # ------------------------------------------------------------------------------------------------------------------
    # --------------- compile --------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def compile(self, model=None) -> None:

        """
        Compile model according to specified internal parameters. Particularly, which loss function and which
        optimizer to use.

        Returns
        -------
        None
        """

        if model is None:
            model = self.training_model

        assert model is not None

        self._determine_lr_scheduler()

        if self.params.optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        else:
            optimizer = self.params.optimizer

        if len(self.losses) == 1:
            model.compile(loss=self.losses[0],
                          metrics=self.metrics,
                          optimizer=optimizer,
                          experimental_run_tf_function=self.params.run_tf_function
                          )
        else:
            model.compile(loss=self.losses,
                          metrics=self.metrics,
                          # loss weights already defined in loss
                          # loss_weights=None,
                          optimizer=optimizer,
                          experimental_run_tf_function=self.params.run_tf_function
                          )

    # ------------------------------------------------------------------------------------------------------------------
    # --------------- training and validation --------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def _prepare_data(self, input_data=None, output_data=None):

        """
        Returns data tuple for input data and output_data.
        Can be overridden in child class to yield a data generator, for example.

        Parameters
        ----------
        input_data
        output_data

        Returns
        -------
        tuple of input and output data or Keras data generator
        """

        if isinstance(input_data, tf.keras.utils.Sequence):
            return input_data
        else:
            return input_data, output_data

    def train(self,
              train_input_data,
              train_output_data,
              test_input_data,
              test_output_data,
              epochs: int = None,
              initial_epoch: int = None) -> None:

        """
        Training method for a general keras model. Prepare training and test data, define callbacks, train and
        save training results when finished.

        Parameters
        ----------
        train_input_data
            numpy array or list of numpy arrays with equal length along first / batch dimension
            or Keras data generator for training model input data
        train_output_data
            numpy array or list of numpy arrays with equal length along first / batch dimension
            for training model output data
        test_input_data
            numpy array or list of numpy arrays with equal length along first / batch dimension
            or Keras data generator for validation model input data
        test_output_data
            numpy array or list of numpy arrays with equal length along first / batch dimension
            for validation model output data
        epochs : int
            number of epochs for training
        initial_epoch : int
            initial epoch number, to resume training

        Returns
        -------
        None
        """

        assert isinstance(self.training_model, tf.keras.models.Model)

        if epochs is not None:
            self.params.epochs = epochs

        if initial_epoch is not None:
            self.params.initial_epoch = initial_epoch

        # ----------------- data and callback preparation -----------------

        # reset callbacks
        self.callbacks = []

        # prepare data
        train_data = self._prepare_data(train_input_data, train_output_data)
        test_data = self._prepare_data(test_input_data, test_output_data)

        # define callbacks for training
        # (annealing callbacks could have possibly been added in child implementation of _prepare_data)
        self._define_standard_callbacks()

        # add data dependent callback (must be implemented in child class)
        self._define_data_dependent_callbacks(train_input_data, train_output_data, test_input_data, test_output_data)

        # ----------------- evaluation of untrained model on test data -----------------

        # if test_data is not None:
        #     # evaluate untrained model on test data (just for record)
        #     loss = self.training_model.evaluate(test_input_data, test_output_data)
        #     loss = np.around(loss, 4)
        #
        #     # save untrained model (just for record)
        #     self.save(folder_path=self.params.save_folder, model_name=f"model_00_{loss}")

        # ----------------- training -----------------

        if isinstance(train_data, tf.keras.utils.Sequence):
            x = train_data
            y = None
            batch_size = None
            train_steps_per_epoch = train_data.__len__()
        else:
            x = train_data[0]
            y = train_data[1]
            batch_size = self.params.batch_size
            train_steps_per_epoch = None

        if isinstance(test_data, tf.keras.utils.Sequence):
            test_steps_per_epoch = test_data.__len__()
        else:
            test_steps_per_epoch = None

        history = self.training_model.fit(x=x,
                                          y=y,
                                          validation_data=test_data,
                                          epochs=self.params.epochs,
                                          batch_size=batch_size,
                                          callbacks=self.callbacks,
                                          initial_epoch=self.params.initial_epoch,
                                          steps_per_epoch=train_steps_per_epoch,
                                          validation_steps=test_steps_per_epoch
                                          )

        # ----------------- training results -----------------

        self._save_training_results(history, self.training_results_path)

    def _load_training_results(self,
                               save_folder: str) -> None:
        """
        Load training results (losses and metrics) from file.

        Parameters
        ----------
        save_folder : str
            path to folder with corresponding files "loss_histories.pkl" and "lr_histories.pkl"

        Returns
        -------
        None
        """

        if os.path.exists(os.path.join(save_folder, "loss_histories.pkl")):
            self.loss_histories = self.pickle_from_file(save_folder, "loss_histories.pkl")
        if os.path.exists(os.path.join(save_folder, "lr_histories.pkl")):
            self.lr_histories = self.pickle_from_file(save_folder, "lr_histories.pkl")

    def _save_training_results(self,
                               history,
                               save_folder: str) -> None:

        """
        Extract training information from training history and save to files.

        Parameters
        ----------
        history : keras training history
        save_folder : str
            path to folder where training results should be saved

        Returns
        -------
        None
        """

        self.last_loss_history = history.history
        self.loss_histories.append(self.last_loss_history)
        self.pickle_to_file(self.loss_histories, save_folder, "loss_histories.pkl")

        def print_losses():
            print(self.loss_histories)

        self.print_to_file(print_losses, save_folder, "loss_histories.txt")

        if self.params.lr_info:
            self.lr_histories.append(self.lr_callback.lr_history)
            self.pickle_to_file(self.lr_histories, save_folder, "lr_histories.pkl")

        def print_lr():
            print(self.lr_histories)

        self.print_to_file(print_lr, save_folder, "lr_histories.txt")

    # TODO: implement

    def validate(self,
                 val_input_data,
                 val_output_data):
        """
        Validate internal training model on given input and output data.

        Parameters
        ----------
        val_input_data
        val_output_data

        Returns
        -------
        evaluation metrics
        """

        metrics = None

        # ----------------- return -----------------

        return metrics

    @staticmethod
    def _custom_metric(aggregated_layer):

        """
        Create custom metric from some aggregated layer output.

        Parameters
        ----------
        aggregated_layer

        Returns
        -------
        metric function
        """

        # Pseudo condition is only used to
        # get rid of pyCharm warnings stating that function does not depend on inputs which is intended here.
        pseudo_condition = False

        def metric(true, pred):
            if pseudo_condition:
                print(true - pred)

            return aggregated_layer

        return metric
