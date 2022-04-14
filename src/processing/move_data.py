import os

PATH_TO = "./old_data"

def move_data(path_from):
    if os.path.isdir(PATH_TO) == False:
        os.mkdir(PATH_TO)

    files_raw_data = [path_from + "/" + x for x in os.listdir(path_from)]
    files_old_data = [x.replace("raw", "old") for x in files_raw_data]
    for i in range(len(files_raw_data)):
        os.replace(files_raw_data[i], files_old_data[i])
