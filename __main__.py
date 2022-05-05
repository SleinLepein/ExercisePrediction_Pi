import time
import os
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler
from src.predict.prediction import construct_message
from src.sender.sender import send_to_app

PATH = "./raw_data"


def start_observer():
    observer = PollingObserver()
    file_event_handler = FileHandler()
    if not os.path.isdir(PATH):
        os.mkdir(PATH)

    observer.schedule(file_event_handler, path=PATH)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


class FileHandler(FileSystemEventHandler):
    def __init__(self):
        self.last_run_time = time.time()

    def on_modified(self, event):
        # Prevent the function from running multiple times in a row by checking previous runtime
        if time.time() - self.last_run_time < 10.0:
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
            prediction = construct_message(file_path)
            send_to_app(prediction)


if __name__ == "__main__":
    start_observer()
