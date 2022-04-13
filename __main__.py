import time
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler
import shared_code.prediction as prediction
import os

def start_Observer():
    observer = PollingObserver()
    file_event_handler = File_Handler()
    observer.schedule(file_event_handler, path='./raw_data')
    observer.start()
    print("Observer started ...")

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

class File_Handler(FileSystemEventHandler):
    def __init__(self):
        self.recent_run_time = time.time()

    def on_modified(self, event):
        if time.time() - self.recent_run_time < 5.0:
            return
        else:
            self.recent_run_time = time.time()
        if event.is_directory:
            try:
                files_in_folder = [event.src_path + "/" + f for f in os.listdir(event.src_path)]
                mod_file_path = max(files_in_folder, key=os.path.getmtime)
            except ValueError:
                return
        else:
            mod_file_path = event.src_path
        print(f"Event Triggered:\n\tEvent Type:\t{event.event_type}\n\tPath:\t\t{mod_file_path}\n\tTime:\t\t{time.asctime()}\n")
        pred = prediction.main(mod_file_path)
        print(pred)

if __name__ == "__main__":
    start_Observer()