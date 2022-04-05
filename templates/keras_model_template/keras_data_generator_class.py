from dotdict import DotDict

import tensorflow as tf
from templates.parameter_class_template.parameter_class import ParameterClass


class KerasDataGenerator(ParameterClass, tf.keras.utils.Sequence):
    """
    Basic data generator class.
    """

    standard_generator_params = DotDict(name="data_generator",
                                        batch_size=32)

    def __init__(self, **params_dict):
        """
        Initialize base data generator class.

        Parameters
        ----------
        params_dict
        """

        # ----------------- parameter loading and preparation -----------------
        super().__init__(**params_dict)

        # update default parameters with given parameters
        self.update_params(self.input_params, self.standard_generator_params)

    # ------------------------------------------------------------------------------------------------------------------
    # --------------- layout for functionality of child data generator classes  ----------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __len__(self):
        """
        Compute count of batches per epoch.

        Returns
        -------
        int
            Count of batches per epoch
        """

        pass

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

        pass
