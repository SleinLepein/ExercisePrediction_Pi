import os
from contextlib import redirect_stdout

import pickle


def print_to_file(print_function, folder_path, file_name):
    """
    Write output of a print function to file. Can be used for instance to save the output of a model summary.

    Parameters
    ----------
    print_function : function
        console print function whose output should be saved to file
    folder_path : str
    file_name : str

    Returns
    -------
    None
    """

    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'w') as f:
        with redirect_stdout(f):
            print_function()


def pickle_to_file(object_to_dump, save_folder, file_name):

    """
    Pickle object to file in save folder with specific file name.

    Parameters
    ----------
    object_to_dump : object
        serializable object to be pickled to file
    save_folder : str
    file_name : str

    Returns
    -------
    None
    """

    file_path = os.path.join(save_folder, file_name)
    with open(file_path, "wb") as f:
        pickle.dump(object_to_dump, f)


def pickle_from_file(folder_path, file_name):

    """
    Load data from file via Pickle.

    Parameters
    ----------
    folder_path : str
    file_name : str

    Returns
    -------
    loaded data
    """

    file_path = os.path.join(folder_path, file_name)
    with open(file_path, "rb") as f:
        result = pickle.load(f)

    # ----------------- return -----------------

    return result
