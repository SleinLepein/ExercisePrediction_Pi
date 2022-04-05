from dotdict import DotDict

import numpy as np

from templates.keras_model_template.keras_data_generator_class import KerasDataGenerator
from templates.keras_model_template.helpers.data_handling import get_batch_data


class StandardDataGeneratorClass(KerasDataGenerator):
    """
    This class defines a basic data generator.
    """

    standard_example_params = DotDict()

    def __init__(self, inputs, outputs, **params_dict):
        """
        Initialize base model class.

        Parameters
        ----------
        params_dict
        """

        # ----------------- parameter loading and preparation -----------------
        super().__init__(**params_dict)

        # update default parameters with given parameters
        self.update_params(self.input_params, self.standard_example_params)

        # ----------------- initialize -----------------
        self.inputs = inputs
        self.outputs = outputs
        self.batch_size = self.params.batch_size
        self.indexes = None
        self.length = None

        self._determine_input_length()

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
