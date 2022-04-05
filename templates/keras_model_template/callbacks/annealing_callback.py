from copy import copy

import tensorflow as tf
from dotdict import DotDict

from templates.parameter_class_template.parameter_class import ParameterClass


class AnnealingCallback(ParameterClass, tf.keras.callbacks.Callback):
    standard_params = DotDict(start_epoch=0,
                              annealing_epochs=5,
                              off_value=None,
                              start_value=0.0,
                              end_value=1.0,
                              method="linear",
                              name="annealing_callback",
                              verbose=0)

    def __init__(self, **params_dict):
        super().__init__(**params_dict)

        # update internal params based on given input params and given standard params
        self.update_params(self.input_params, self.standard_params)

        # init
        self._prepare_params()

        self.start_value = self.params.start_value
        self.end_value = self.params.end_value
        self.difference = self.end_value - self.start_value
        self.start_epoch = self.params.start_epoch
        self.annealing_epochs = self.params.annealing_epochs
        self.step = self.difference / self.annealing_epochs

        self.started = False
        self.weight = tf.keras.backend.variable(self.params.off_value)
        self.weight._trainable = False

        self.parameters = copy(self.params)

        self.on_epoch_begin(-1, None)

    def _prepare_params(self):
        if self.params.off_value is None:
            self.params.off_value = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        if epoch >= self.start_epoch:
            epoch_diff = epoch - self.start_epoch

            if not self.started:
                self.started = True
                self.weight.assign(self.start_value + epoch_diff * self.step)
                # tf.keras.backend.set_value(self.weight, self.start_value + epoch_diff * self.step)

            # compute new weight
            if self.difference >= 0:
                new_weight = min(self.start_value + epoch_diff * self.step,
                                 self.parameters.end_value)
            else:
                new_weight = max(self.start_value + epoch_diff * self.step,
                                 self.parameters.end_value)

            # set new weight
            # tf.keras.backend.set_value(self.weight, new_weight)
            self.weight.assign(new_weight)


# class AnnealingCallback(tf.keras.callbacks.Callback):
#     standard_params = DotDict(start_epoch=0,
#                               annealing_epochs=5,
#                               off_value=None,
#                               start_value=0.0,
#                               end_value=1.0,
#                               method="linear",
#                               name="annealing_callback",
#                               verbose=0)
#
#     def __init__(self,
#                  **params_dict):
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
#         self.parameters = self._get_updated_params(self.parameters, self.standard_params)
#         self.parameters = DotDict(self.parameters)
#
#         if self.parameters.off_value is None:
#             self.parameters.off_value = self.parameters.start_value
#
#         # ----------------- initialization -----------------
#
#         self.name = self.parameters['name']
#
#         self.start_value = float(self.parameters.start_value)
#         self.difference = self.parameters.end_value - self.parameters.start_value
#         self.start_epoch = self.parameters.start_epoch
#         self.annealing_epochs = self.parameters.annealing_epochs
#         self.step = self.difference / self.annealing_epochs
#
#         self.started = False
#         self.weight = tf.keras.backend.variable(self.parameters.off_value)
#         self.weight._trainable = False
#
#         self.on_epoch_end(-1)
#
#     # ----------------------------------------------------------------------------------------------------------------
#     # --------------- parameter handling -----------------------------------------------------------------------------
#     # ----------------------------------------------------------------------------------------------------------------
#
#     @staticmethod
#     def _get_updated_params(params, standard_params):
#         return get_updated_params(params, standard_params)
#
#     def _update_params(self, input_params, standard_params):
#
#         """
#         Update internal class parameters for given default parameters and new parameters.
#
#         :param input_params: new parameters
#         :param standard_params: default parameters
#         :return: None
#         """
#
#         self.parameters = self._get_updated_params(standard_params, self.parameters)
#         self.parameters = self._get_updated_params(input_params, self.parameters)
#
#     # ----------------------------------------------------------------------------------------------------------------
#     # --------------- functionality ----------------------------------------------------------------------------------
#     # ----------------------------------------------------------------------------------------------------------------
#
#     def on_epoch_begin(self, epoch, logs=None):
#
#         # TODO: add other annealing methods
#
#         if epoch >= self.start_epoch:
#             epoch_diff = epoch - self.start_epoch
#
#             if not self.started:
#                 self.started = True
#                 self.weight.assign(self.start_value + epoch_diff * self.step)
#                 # tf.keras.backend.set_value(self.weight, self.start_value + epoch_diff * self.step)
#
#             # compute new weight
#             if self.parameters["method"] == "linear":
#
#                 if self.difference >= 0:
#                     new_weight = min(self.start_value + epoch_diff * self.step,
#                                      self.parameters.end_value)
#                 else:
#                     new_weight = max(self.start_value + epoch_diff * self.step,
#                                      self.parameters.end_value)
#
#             else:
#                 raise NotImplementedError(f"Choice of annealing method {self.parameters['method']} not implemented.")
#
#             # set new weight
#             # tf.keras.backend.set_value(self.weight, new_weight)
#             self.weight.assign(new_weight)
#
#         if self.parameters.verbose:
#             # print message to console
#             print("\n")
#             print(f"Current weight in {self.name}: {tf.keras.backend.get_value(self.weight)}")
