import time
import os
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler
from src.predict.prediction import construct_message
from src.sender.sender import send_to_app
from src.processing.move_data import move_data

PATH = "./raw_data"

def start_Observer():
    observer = PollingObserver()
    file_event_handler = File_Handler()
    if os.path.isdir(PATH) == False:
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
    if input("\nMove files from /raw_data to /old_data? [Y/N]\n") in ["y", "Y", "yes", "Yes", "ye", "Ye"]:
        move_data(PATH)
        print("Data moved!")

class File_Handler(FileSystemEventHandler):
    def __init__(self):
        self.recent_run_time = time.time()

    def on_modified(self, event):
        if time.time() - self.recent_run_time < 5.0:
            return None
        else:
            self.recent_run_time = time.time()
        if event.is_directory:
            try:
                files_in_folder = [event.src_path + "/" + x for x in os.listdir(event.src_path)]
                mod_file_path = max(files_in_folder, key=os.path.getctime)
            except ValueError:
                return None
        else:
            mod_file_path = event.src_path
        if mod_file_path.endswith(".csv"):
            print(f"Event Type:\t{event.event_type}\nPath:\t\t{mod_file_path}\nTime:\t\t{time.asctime()}\n")
            pred, success = construct_message(mod_file_path)
            if success:
                print("Sending ...")
                send_to_app(pred)
            else:
                print(f"Error:\n{pred}")

if __name__ == "__main__":
    start_Observer()