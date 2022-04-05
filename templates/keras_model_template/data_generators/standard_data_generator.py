"""
This file implements a standard Keras data generator for given input and output data as well as a specified batch size.
Such a data generator can be used for x with y=None and batch_size=None in a Keras model's train method instead of
providing input data for x, output data for y and the desired batch size parameter for batch_size. From this point
of view, the example provided here is rather educational and can serve as a starting point for more advanced
data generators that have real benefits, e.g., when total amount of data does not fit into memory.
"""

import numpy as np
import tensorflow as tf
from dotdict import DotDict

from templates.keras_model_template.helpers.data_handling import get_batch_data
from templates.parameter_class_template.helpers.params_handling import get_updated_params


class StandardDataGenerator(tf.keras.utils.Sequence):

    standard_params = DotDict(epoch_size=None)

    def __init__(self, inputs, outputs, batch_size, **params_dict):
        super().__init__()

        # ----------------- parameter loading and preparation -----------------
        # initialize params_dict if not given
        if params_dict is None:
            params_dict = {}

        self.params = DotDict(params_dict)

        # update default params with new given params
        self.params = get_updated_params(self.params, self.standard_params)

        # ----------------- init -----------------

        self.inputs = inputs
        self.outputs = outputs
        self.batch_size = batch_size
        self.indexes = None
        self.length = None
        self._determine_input_length()
        self.epoch_size = self.params.epoch_size

        # ----------------- preparations for next epoch -----------------
        self.on_epoch_end()

    def _determine_input_length(self):
        """
        Determine the count of total training examples and set corresponding class member length.

        Returns
        -------
        None
        """

        if isinstance(self.inputs, list):
            self.length = len(self.inputs[0])
        else:
            self.length = len(self.inputs)

    def __len__(self):
        """
        Compute count of batches per epoch.

        Returns
        -------
        int
            Count of batches per epoch
        """

        if self.params.epoch_size is None:
            length = self.length
        else:
            length = self.params.epoch_size

        return int(np.floor(length / self.batch_size))

    def __getitem__(self, index):
        """
        Get batch with specific index out of every batches.

        Parameters
        ----------
        index : int

        Returns
        -------
        tuple
            batch input data and corresponding batch output data
        """

        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Get batch data
        input_data = get_batch_data(self.inputs, indexes)
        output_data = get_batch_data(self.outputs, indexes)

        return input_data, output_data

    def on_epoch_end(self):
        """
        Randomize ordering of data indexes for random batches at the end of every epoch.

        Returns
        -------
        None
        """

        self.indexes = np.arange(self.length)
        np.random.shuffle(self.indexes)
