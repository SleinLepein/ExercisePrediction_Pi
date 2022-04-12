import time 
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import shared_code.prediction as prediction

def startObserver():
    observer = Observer()
    file_event_handler = FileHandler()
    observer.schedule(file_event_handler, path='./raw_data')
    observer.start()
    print("Observer started ...")

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()

class FileHandler(FileSystemEventHandler):
    def __init__(self):
        self.recent_run_time = time.time()
    
    def on_modified(self, event):
        if time.time() - self.recent_run_time < 5.0:
            return
        else:
            self.recent_run_time = time.time()
        print(f"File {event.src_path} got modified")
        pred = prediction.main(event.src_path)
        print(pred)
        print(type(pred))

if __name__ == "__main__":
    startObserver()