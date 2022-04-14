import time
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler
import src.predict.prediction as prediction
import src.sender.sender as sender
import src.processing.move_data as mv
import os

PATH = "./raw_data"

def start_Observer():
    observer = PollingObserver()
    file_event_handler = File_Handler()
    if os.path.isdir(PATH) == False:
        os.mkdir(PATH)

    observer.schedule(file_event_handler, path=PATH)
    observer.start()
    print("Observer started ...")

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    print("\nMove files from /raw_data to /old_data? [Y/N]")
    move_files = input()
    if move_files in ["y", "Y", "yes", "Yes", "ye", "Ye"]:
        mv.move_data(PATH)
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
                mod_file_path = max(files_in_folder, key=os.path.getmtime)
            except ValueError:
                return None
        else:
            mod_file_path = event.src_path
        if mod_file_path.endswith(".csv"):
            print(f"Event Triggered:\n\tEvent Type:\t{event.event_type}\n\tPath:\t\t{mod_file_path}\n\tTime:\t\t{time.asctime()}\n")
            pred, success = prediction.main(mod_file_path)
            if success:
                print("Sending ...")
                sender.send_to_pi(pred)
            else:
                print(f"Error:\n{pred}")

if __name__ == "__main__":
    start_Observer()