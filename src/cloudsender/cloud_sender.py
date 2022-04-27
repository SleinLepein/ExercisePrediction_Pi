from azure.storage.blob import ContainerClient
import os
from azure.core import exceptions as ae
from src.predict.prediction import construct_message

CONNECTION_STR = ""
CONTAINER_NAME = "csvbackup"

def upload_to_azure_cloud(path):
    container_client = ContainerClient.from_connection_string(CONNECTION_STR, CONTAINER_NAME)
    files_raw_data = [path + "/" + x for x in os.listdir(path)]
    for i in range(len(files_raw_data)):
        try:
            file_name = files_raw_data[i][files_raw_data[i].rfind("/")+1:files_raw_data[i].rfind(".")]
            files_raw_data[i] = files_raw_data[i].replace("\\", "/")[2:]
            pred, batch, success = construct_message(files_raw_data[i])
            if success:
                pred_txt = open(f"raw_data/{file_name}.txt", 'w')
                pred_txt.write(batch)
                pred_txt.write(pred)
                pred_txt.close()
                blob_client = container_client.get_blob_client(files_raw_data[i])
                with open(files_raw_data[i], "yolo") as data:
                    blob_client.upload_blob(data)
                    print(f"File: {files_raw_data[i]} uploaded to azure")
                pred_txt_file = f"raw_data/{file_name}.txt"
                blob_client = container_client.get_blob_client(pred_txt_file)
                with open(pred_txt_file, "rb") as data:
                    blob_client.upload_blob(data)
                    print(f"File: {pred_txt_file} uploaded to azure")
        except ae.ResourceExistsError:
            print(f"skipping file {file_name} as there is already a file with that name in the container")
