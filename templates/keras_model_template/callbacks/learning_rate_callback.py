import tensorflow as tf
from dotdict import DotDict

from templates.parameter_class_template.helpers.params_handling import get_updated_params


class LearningRateCallback(tf.keras.callbacks.Callback):

    """
    Defines a Keras Callback to collect and display information about learning rate during training.
    """

    standard_params = DotDict(name="lr_callback",
                              verbose=1,
                              batch_log_every=100)

    def __init__(self,
                 **params_dict):

        """
        Initialize Learning Rate Callback.

        Parameters
        ----------
        params_dict
        """

        super().__init__()

        # ----------------- parameter loading and preparation -----------------

        # initialize params_dict if not given
        if params_dict is None:
            params_dict = {}

        self.parameters = DotDict(params_dict)
        self.input_params = self.parameters

        # update default params with new given params
        self.parameters = self._get_updated_params(self.parameters, self.standard_params)

        # ----------------- initialization -----------------

        self.lr_type = None
        self.lr_history = []
        self.lr_batch_history = []

    # ------------------------------------------------------------------------------------------------------------------
    # --------------- parameter handling -------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _get_updated_params(params, standard_params):
        return get_updated_params(params, standard_params)

    def _update_params(self, input_params, standard_params):

        """
        Update internal parameters based on standard parameters and new input parameters.

        Parameters
        ----------
        input_params : dict
        standard_params : dict

        Returns
        -------
        None
        """

        self.parameters = self._get_updated_params(standard_params, self.parameters)
        self.parameters = self._get_updated_params(input_params, self.parameters)

    # ------------------------------------------------------------------------------------------------------------------
    # --------------- functionality ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def _determine_lr_type(self):

        """
        Determine how to get current learning rate from optimizer,

        Returns
        -------
        None
        """

        try:
            self.model.optimizer.lr.__call__(0)
            self.lr_type = "scheduler"
        except AttributeError:
            self.lr_type = "variable"

    def _get_lr(self):

        """
        Get current learning rate.

        Returns
        -------
        float
            current learning rate
        """

        if self.lr_type is None:
            self._determine_lr_type()

        if self.lr_type == "scheduler":
            lr = self.model.optimizer.lr.__call__(self.model.optimizer.iterations)
        elif self.lr_type == "variable":
            lr = self.model.optimizer.lr
        else:
            raise NotImplementedError("Type of learning rate is not implemented.")

        return tf.keras.backend.get_value(lr)

    def on_epoch_begin(self, epoch, logs=None):

        """
        Retrieve current learning rate when a new training epoch starts. Print it to console and remember it.

        Parameters
        ----------
        epoch : int
        logs

        Returns
        -------
        None
        """

        lr = self._get_lr()
        self.lr_history.append(lr)

        if self.parameters.verbose:
            # print message to console
            print("\n")
            print(f"Current learning rate: {lr}")

    def on_batch_begin(self, batch, logs=None):

        """
        Remember learning rate after every x batch.

        Parameters
        ----------
        batch : int
        logs

        Returns
        -------
        None
        """

        if batch % self.parameters.batch_log_every == 0:
            lr = self._get_lr()
            self.lr_batch_history.append(lr)
