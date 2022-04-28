import time
import os
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler
from src.predict.prediction import construct_message
from src.sender.sender import send_to_app
from src.processing.change_data import delete_data
from src.cloudsender.cloud_sender import upload_to_azure_cloud

PATH = "./raw_data"


def start_observer():
    observer = PollingObserver()
    file_event_handler = FileHandler()
    if not os.path.isdir(PATH):
        os.mkdir(PATH)

    observer.schedule(file_event_handler, path=PATH)
    observer.start()
    print("Observer started ...\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

    # Try sending data to cloud and if it was successful delete it locally
    if input("\nMove files to the cloud and delete locally? [Y/N]\n") in ["y", "Y", "ye", "Ye", "yes", "Yes"]:
        try:
            upload_to_azure_cloud(PATH)
        except Exception as ex:
            print(f"An error occurred when uploading data to the cloud\n{ex}")
        else:
            delete_data(PATH)


class FileHandler(FileSystemEventHandler):
    def __init__(self):
        self.last_run_time = time.time()

    def on_modified(self, event):
        # Prevent the function from running multiple times in a row by checking previous runtime
        if time.time() - self.last_run_time < 5.0:
            return None
        else:
            self.last_run_time = time.time()
        # Filepath can be corrupted and not show to the file but only the folder
        # if that's the case we look at the recently edited file
        if event.is_directory:
            try:
                files_in_folder = [event.src_path + "/" + x for x in os.listdir(event.src_path)]
                file_path = max(files_in_folder, key=os.path.getctime)
            except ValueError:
                return None
        else:
            file_path = event.src_path
        # Prevent the prediction to run on any non-csv-file
        if file_path.endswith(".csv"):
            print(f"Event Type:\t{event.event_type}\nPath:\t\t{file_path}\nTime:\t\t{time.asctime()}\n")
            prediction, _, success = construct_message(file_path)
            if success:
                print("Sending ...")
                send_to_app(prediction)
            else:
                print(f"Error:\n{prediction}")


if __name__ == "__main__":
    start_observer()
