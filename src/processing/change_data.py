import os

PATH = "./old_data"


def move_data(path_from):
    """
    moves csv files into another folder
    ----------
    path_from : String
        path to the directory containing the files that should be moved
    """
    if not os.path.isdir(PATH):
        os.mkdir(PATH)

    files_raw_data = [path_from + "/" + x for x in os.listdir(path_from)]
    files_old_data = [x.replace("raw", "old") for x in files_raw_data]
    for i in range(len(files_raw_data)):
        try:
            os.replace(files_raw_data[i], files_old_data[i])
        except PermissionError:
            pass


def delete_data(path_from):
    """
    delete files in a given folder
    ----------
    path_from : String
        path to the directory containing the files that should be removed
    """
    if not os.path.isdir(path_from):
        return None

    files_raw_data = [path_from + "/" + x for x in os.listdir(path_from)]
    for i in range(len(files_raw_data)):
        try:
            os.remove(files_raw_data[i])
        except PermissionError:
            pass
