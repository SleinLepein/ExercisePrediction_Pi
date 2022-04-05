import numpy as np
import tensorflow as tf
from dotdict import DotDict

from templates.keras_model_template.callbacks.annealing_callback import AnnealingCallback
from templates.keras_model_template.helpers.data_handling import extend_data, get_batch_data
from templates.keras_model_template.keras_data_generator_class import KerasDataGenerator


class AnnealingDataGenerator(KerasDataGenerator):
    standard_params = DotDict(batch_size=32,
                              annealing_params=[{}])

    def __init__(self, model_inputs, model_outputs, **params_dict):
        super().__init__(**params_dict)

        # update internal params based on given input params and given standard params
        self.update_params(self.input_params, self.standard_params)

        # ----------------- init -----------------

        self.inputs = model_inputs
        self.outputs = model_outputs
        self.batch_size = self.params.batch_size

        if isinstance(self.inputs, tf.keras.utils.Sequence):
            # TODO: check this with general keras data generators / find alternative

            self.inputs.batch_size = self.batch_size
            self.inputs.on_epoch_end()

        self.indexes = None
        self.data_count = None
        self._determine_input_length()

        self.callbacks = []
        self._get_annealing_callbacks()

        self.on_epoch_end()

    def _get_annealing_callbacks(self):
        for params in self.params.annealing_params:
            self.callbacks.append(AnnealingCallback(**params))

    def _determine_input_length(self):
        if isinstance(self.inputs, tf.keras.utils.Sequence):
            # self.length = self.inputs.length
            self.data_count = self.inputs.__len__() * self.batch_size
        elif isinstance(self.inputs, list):
            self.data_count = len(self.inputs[0])
        else:
            self.data_count = len(self.inputs)

    def __len__(self):
        """

        Returns the number of batches per epoch.
        -------

        """

        return int(np.floor(self.data_count / self.batch_size))

    def __getitem__(self, index):
        """

        Parameters
        ----------
        index

        Returns batch of data with given index.
        -------

        """

        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Get batch data
        if isinstance(self.inputs, tf.keras.utils.Sequence):
            input_data, output_data = self.inputs.__getitem__(index)
        else:
            input_data = self.get_batch_data(self.inputs, indexes)
            output_data = self.get_batch_data(self.outputs, indexes)

        callback_data = self._get_callback_batch_data(self.batch_size)
        input_data = extend_data(input_data, callback_data)

        return input_data, output_data

    def _get_callback_batch_data(self, batch_size):
        """

        Returns a batch with weights given by callbacks.
        -------

        """

        assert len(self.callbacks) > 0

        batch_data = []

        for callback in self.callbacks:
            temp_data = tf.keras.backend.get_value(callback.weight) * np.ones(batch_size).reshape((batch_size, 1))
            batch_data.append(temp_data)

        return batch_data

    def on_epoch_end(self):
        """

        Update indexes at end of epoch.
        -------

        """

        self.indexes = np.arange(self.data_count)
        np.random.shuffle(self.indexes)

    def get_batch_data(self, data, indexes):
        """

        Parameters
        ----------
        data
        indexes

        Returns subset of data with given indexes
        -------

        """

        if isinstance(data, list):
            batch_data = []
            for sub_data in data:

                if isinstance(sub_data, list):
                    # TODO implement
                    raise NotImplementedError("sub_data is of type list which is not implemented, yet")
                else:
                    batch_data.append(sub_data[indexes])

        else:
            batch_data = data[indexes]

        return batch_data


# class AnnealingDataGenerator(tf.keras.utils.Sequence):
#
#     standard_params = DotDict(batch_size=32,
#                               annealing_params=[{}])
#
#     def __init__(self, model_inputs, model_outputs, **params_dict):
#         super().__init__()
#
#         # ----------------- parameter loading and preparation -----------------
#
#         # initialize params_dict if not given
#         if params_dict is None:
#             params_dict = {}
#
#         self.parameters = DotDict(params_dict)
#         self.input_params = self.parameters
#
#         # update default params with new given params
#         self.parameters = get_updated_params(self.parameters, self.standard_params)
#         self.parameters = DotDict(self.parameters)
#
#         # ----------------- init -----------------
#
#         self.inputs = model_inputs
#         self.outputs = model_outputs
#         self.batch_size = self.parameters.batch_size
#
#         if isinstance(self.inputs, tf.keras.utils.Sequence):
#             self.inputs.batch_size = self.batch_size
#
#         self.indexes = None
#         self.length = None
#         self._determine_input_length()
#
#         self.callbacks = []
#         self._get_annealing_callbacks()
#
#         self.on_epoch_end()
#
#     def _get_annealing_callbacks(self):
#         for params in self.parameters.annealing_params:
#             self.callbacks.append(AnnealingCallback(**params))
#
#     def _determine_input_length(self):
#         if isinstance(self.inputs, tf.keras.utils.Sequence):
#             # self.length = self.inputs.length
#             self.length = self.inputs.__len__()
#         elif isinstance(self.inputs, list):
#             self.length = len(self.inputs[0])
#         else:
#             self.length = len(self.inputs)
#
#     def __len__(self):
#         """
#
#         Returns the number of batches per epoch.
#         -------
#
#         """
#
#         return int(np.floor(self.length / self.batch_size))
#
#     def __getitem__(self, index):
#         """
#
#         Parameters
#         ----------
#         index
#
#         Returns batch of data with given index.
#         -------
#
#         """
#
#         # Generate indexes of the batch
#         indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
#
#         # Get batch data
#         if isinstance(self.inputs, tf.keras.utils.Sequence):
#             input_data, output_data = self.inputs.__getitem__(index)
#         else:
#             input_data = get_batch_data(self.inputs, indexes)
#             output_data = get_batch_data(self.outputs, indexes)
#
#         callback_data = self._get_callback_batch_data(self.batch_size)
#         input_data = extend_data(input_data, callback_data)
#
#         return input_data, output_data
#
#     def _get_callback_batch_data(self, batch_size):
#         """
#
#         Returns a batch with weights given by callbacks.
#         -------
#
#         """
#
#         assert len(self.callbacks) > 0
#
#         batch_data = []
#
#         for callback in self.callbacks:
#             temp_data = np.ones(batch_size).reshape((batch_size, 1))
#
#             print(callback.weight)
#             print(tf.keras.backend.get_value(callback.weight))
#
#             temp_data = tf.keras.backend.get_value(callback.weight) * temp_data
#
#             # batch_data.append(np.array(temp_data))
#             batch_data.append(temp_data)
#
#         return batch_data
#
#     def on_epoch_end(self):
#         """
#
#         Update indexes at end of epoch.
#         -------
#
#         """
#
#         self.indexes = np.arange(self.length)
#         np.random.shuffle(self.indexes)
