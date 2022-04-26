from azure.storage.blob import ContainerClient
import os
from azure.core import exceptions as ae

CONNECTION_STR = ""
CONTAINER_NAME = "csvbackup"

def upload_to_azure_cloud(path):
    container_client = ContainerClient.from_connection_string(CONNECTION_STR, CONTAINER_NAME)
    files_raw_data = [path + "/" + x for x in os.listdir(path)]
    for i in range(len(files_raw_data)):
        try:
            files_raw_data[i] = files_raw_data[i].replace("\\", "/")[2:]
            blob_client = container_client.get_blob_client(files_raw_data[i])
            with open(files_raw_data[i], "rb") as data:
                blob_client.upload_blob(data)
                print(f"File: {files_raw_data[i]} uploaded to azure")
        except ae.ResourceExistsError:
            print(f"skipping file {files_raw_data[i]} as there is already a file with that name in the container")
