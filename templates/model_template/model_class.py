"""
This file implements a basic layout for model like objects.
"""

from dotdict import DotDict
from collections.abc import Iterable

from templates.parameter_class_template.parameter_class import ParameterClass

# TODO: change to abstract class


class ModelClass(ParameterClass):

    """
    Basic model class that can be used for several kinds of models, e.g., neural networks, scaling transformations, ...

    This class defines a basic layout and provides basic functionality for parameter handling.
    """

    # default parameters for all model like objects, e.g., name, maybe more to follow
    standard_model_params = DotDict(name="model")

    def __init__(self, **params_dict):

        """
        Initialize base model class.

        Parameters
        ----------
        params_dict
        """

        # ----------------- parameter loading and preparation -----------------
        super().__init__(**params_dict)

        # update internal parameters self.params with input parameters and with default parameters
        self.update_params(self.input_params, self.standard_model_params)

        # ----------------- common members for child classes -----------------

        # member for model used during training
        self.training_model = None

        # member for model used during inference
        self.model = None

    # ------------------------------------------------------------------------------------------------------------------
    # --------------- layout for functionality of child model classes  -------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def _build(self) -> None:
        """
        Needs implementation in child class how to actually build the model. Use internal class parameters to
        control how to build the model.

        Returns
        -------
        None
        """

        pass

    def save(self,
             folder_path: str,
             model_name: str) -> None:
        """
        Needs implementation in child class how to actually save the model with given model name in given folder path.

        Parameters
        ----------
        folder_path : str
        model_name : str

        Returns
        -------
        None
        """

        pass

    def load(self,
             folder_path: str,
             model_name: str) -> object:

        """
        Needs implementation in child class how to actually load a model with given model name from given folder path.

        Parameters
        ----------
        folder_path : str
        model_name : str

        Returns
        -------
        object
            load model from file
        """

        pass

    def train(self,
              train_input_data,
              train_output_data,
              test_input_data,
              test_output_data) -> None:

        """
        Needs implementation in child class how to actually fit the model to training input and output data and
        how to use test input and output data for validation during training.

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

    def validate(self,
                 val_input_data: Iterable,
                 val_output_data: Iterable) -> None:

        """
        Needs implementation in child class how to actually validate the model for given test input and output data.

        Parameters
        ----------
        val_input_data : Iterable
        val_output_data : Iterable

        Returns
        -------

        """

        pass
