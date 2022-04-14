import os

PATH_TO = "./old_data"

def move_data(path_from):
    if os.path.isdir(PATH_TO) == False:
        os.mkdir(PATH_TO)

    from_files_in_folder = [path_from + "/" + x for x in os.listdir(path_from)]
    to_files_in_folder = [x.replace("raw", "old") for x in from_files_in_folder]
    for i in range(len(from_files_in_folder)):
        os.replace(from_files_in_folder[i], to_files_in_folder[i])
