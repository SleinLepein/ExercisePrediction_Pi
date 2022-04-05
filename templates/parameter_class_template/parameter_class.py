"""
This file implements a class for parameter handling.
"""

import os
from pprint import pprint
import logging
import copy
from dotdict import DotDict

from templates.parameter_class_template.helpers.params_handling import get_updated_params
from templates.parameter_class_template.helpers.input_output import print_to_file, pickle_to_file, pickle_from_file


class ParameterClass:

    """
    Basic parameter class that can be used for several kinds of objects that need to handle parameters.
    """

    # standard parameters for all objects with parameter handling, e.g., save folder path, etc.
    standard_params = DotDict(params_file_path=None,
                              save_folder="./",
                              logger_name="OWN LOGGER")

    def __init__(self, **params_dict):

        """
        Initialize parameter class.

        Parameters
        ----------
        params_dict : dict
            Dictionary with parameter keys and corresponding parameter values
        """

        # ----------------- parameter loading and preparation -----------------

        # initialize params_dict if not given
        if params_dict is None:
            params_dict = {}

        # use dotted dict functionality
        self.params = DotDict(params_dict)

        # remember given input parameters for chaining updates
        self.input_params = copy.deepcopy(self.params)

        # load params from file if specified by "params_file_path" keyword
        if self.params.params_file_path is not None:
            folder = os.path.dirname(self.params.params_file_path)
            file_name = os.path.basename(self.params.params_file_path)
            self.input_params = self.load_params(folder, file_name)

        # update default params (for this parameter class) with new given params
        self.params = self._get_updated_params(self.input_params, self.standard_params)

        # ----------------- logger -----------------

        # create logger
        self.logger = logging.getLogger(self.params.logger_name)
        self.logger.setLevel(logging.DEBUG)

        # create console handler and set level to debug
        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter('%(name)s: %(asctime)s - %(levelname)s - %(message)s')

        # add formatter to ch
        self.ch.setFormatter(formatter)

        # add ch to logger
        self.logger.addHandler(self.ch)

    # ------------------------------------------------------------------------------------------------------------------
    # --------------- parameter handling -------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _get_updated_params(params: dict,
                            default_params: dict) -> DotDict:
        """
        Wrapper method to update dictionary with default parameters by another dictionary with given parameters.

        Parameters
        ----------
        params : dict
        default_params : dict

        Returns
        -------
        DotDict
            dotted dictionary with updated parameter keys and values, i.e., as in default_params if not in params
        """

        return DotDict(get_updated_params(params, default_params))

    def update_params(self,
                      input_params: dict,
                      standard_params: dict) -> None:
        """
        Update INTERNAL class parameters (self.params) for given default parameters and new input parameters.
        Provides functionality for child classes to chain updates of default parameters at each heritage level.

        Parameters
        ----------
        input_params : dict
        standard_params : dict

        Returns
        -------
        None
        """

        # first update standard params with given input params
        new_params = self._get_updated_params(input_params, standard_params)

        # then update internal class params with these updated params
        self.params = self._get_updated_params(new_params, self.params)

    # ------------------------------------------------------------------------------------------------------------------
    # --------------- saving and loading of meta data ------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def print_to_file(print_function,
                      folder_path: str,
                      file_name: str) -> None:

        """
        Wrapper method to be able to save console output of a print function to a file.

        Parameters
        ----------
        print_function : function producing console output to be written to file
        folder_path : str
            path to save folder
        file_name : str
            file name of save file

        Returns
        -------
        None
        """

        print_to_file(print_function, folder_path, file_name)

    def print_params_to_file(self,
                             folder_path: str,
                             file_name: str) -> None:

        """
        Saves internal class parameters to a human readable file.

        Parameters
        ----------
        folder_path : str
            path to save folder
        file_name : str
            file name of save file

        Returns
        -------
        None
        """

        def print_function():
            # pretty print parameter dictionary
            pprint(self.params)

        print_to_file(print_function, folder_path, file_name + "_params.txt")

    @staticmethod
    def pickle_to_file(object_to_dump: object,
                       folder_path: str,
                       file_name: str) -> None:
        """
        Wrapper method to serialize an object via pickle.

        Parameters
        ----------
        object_to_dump : object
        folder_path : str
            path to save folder
        file_name : str
            file name of save file

        Returns
        -------
        None
        """

        pickle_to_file(object_to_dump, folder_path, file_name)

    def save_params(self,
                    folder_path: str,
                    name: str) -> None:

        """
        Save internal class parameters to file.

        Parameters
        ----------
        folder_path : str
            path to save folder
        name : str
            name for parameter dictionary

        Returns
        -------
        None
        """

        self.pickle_to_file(self.params, folder_path, name + "_params.pkl")

    @staticmethod
    def pickle_from_file(folder_path: str,
                         file_name: str) -> object:
        """
        Wrapper method to load serialized object via pickle.

        Parameters
        ----------
        folder_path : str
            path to save folder
        file_name : str
            file name of save file

        Returns
        -------
        object
            loaded object
        """
        return pickle_from_file(folder_path, file_name)

    def load_params(self,
                    folder_path: str,
                    file_name: str) -> DotDict:

        """
        Load parameters from file as dotted dictionary.

        Parameters
        ----------
        folder_path : str
            path to save folder
        file_name : str
            file name of save file

        Returns
        -------
        DotDict
            loaded params
        """

        # ----------------- return -----------------

        return DotDict(self.pickle_from_file(folder_path, file_name))
