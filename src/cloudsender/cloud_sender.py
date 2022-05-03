import os
import datetime
from azure.storage.blob import ContainerClient
from azure.core import exceptions as azure_exception
from src.predict.prediction import construct_message

CONNECTION_STR = "DefaultEndpointsProtocol=https;AccountName=inputdatapreprocessing;AccountKey=xsZo9J4xXVGTTQCrSBT9Bv4wMvSDWnzhbdYAxOJRTSi0ZGtRrsKtl3O+iFtNsycOcR7VIvwzfQUI7F77RMj64g==;EndpointSuffix=core.windows.net"
CONTAINER_NAME = "csvbackup"


def upload_to_azure_cloud(path):
    """
    creates prediction.txt file and sends both txt and csv file into the azure cloud blob storage
    Parameters
    ----------
    path : String
        path to the folder containing the csv files
    """
    container_client = ContainerClient.from_connection_string(CONNECTION_STR, CONTAINER_NAME)
    files_raw_data = [path + "/" + x for x in os.listdir(path)]
    now = datetime.datetime.now()
    for i in range(len(files_raw_data)):
        filename = files_raw_data[i][files_raw_data[i].rfind("/") + 1:files_raw_data[i].rfind(".")]
        try:
            files_raw_data[i] = files_raw_data[i].replace("\\", "/")[2:]
            prediction, batch, success, _ = construct_message(files_raw_data[i])
            if success:
                prediction_filename = f"raw_data/{filename}.txt"
                with open(prediction_filename, 'w') as data:
                    data.write(f"{filename}.csv\n")
                    data.write(batch)
                    data.write(prediction)

                blob_client = container_client.get_blob_client(
                    f"{now.year}/{now.month}/{now.day}/{filename}/{filename}.csv")
                with open(files_raw_data[i], "rb") as data:
                    blob_client.upload_blob(data)
                    print(f"File: {files_raw_data[i]} uploaded to azure")

                blob_client = container_client.get_blob_client(
                    f"{now.year}/{now.month}/{now.day}/{filename}/{filename}.txt")
                with open(prediction_filename, "rb") as data:
                    blob_client.upload_blob(data)
                    print(f"File: {prediction_filename} uploaded to azure")
        except azure_exception.ResourceExistsError:
            print(f"Skipping file {filename} as there is already a file with that name in the container")


def upload_set_to_azure_cloud(prediction, batch, file_path):
    """
    creates prediction.txt file and sends both txt and csv file into the azure cloud blob storage
    will be called when a set is finished to create separate csv files for each set/exercise
    Parameters
    ----------
    prediction : Str
        full prediction String to save in txt
    batch : Str
        String with all windows and the predictions to save in txt
    file_path : Str
        path to the file to upload
    """
    container_client = ContainerClient.from_connection_string(CONNECTION_STR, CONTAINER_NAME)
    now = datetime.datetime.now()
    try:
        filename = file_path[file_path.rfind("/")+1:file_path.rfind(".csv")]
        prediction_filename = file_path.replace(".csv", ".txt")
        with open(prediction_filename, 'w') as data:
            data.write(f"{filename}.csv\n")
            data.write(batch)
            data.write(prediction)
        blob_client = container_client.get_blob_client(
            f"{now.year}/{now.month}/{now.day}/{filename}_{now.hour}{now.minute}{now.second}/{filename}_{now.hour}:{now.minute}:{now.second}.csv")
        with open(file_path, "rb") as data:
            blob_client.upload_blob(data)
            print(f"File: {file_path} uploaded to azure")

        blob_client = container_client.get_blob_client(
            f"{now.year}/{now.month}/{now.day}/{filename}_{now.hour}{now.minute}{now.second}/{filename}_{now.hour}:{now.minute}:{now.second}.txt")
        with open(prediction_filename, "rb") as data:
            blob_client.upload_blob(data)
            print(f"File: {prediction_filename} uploaded to azure")
    except azure_exception.ResourceExistsError:
        pass
