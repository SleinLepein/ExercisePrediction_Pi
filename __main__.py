from datetime import datetime, timedelta
import time 
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import shared_code.prediction as prediction

def startObserver():
    observer = Observer()
    event_handler = FileHandler() # create event handler
    # set observer to use created handler in directory
    observer.schedule(event_handler, path='./raw_data')
    observer.start()
    print("Observer started ...")

    # sleep until keyboard interrupt, then stop + rejoin the observer
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()

class FileHandler(FileSystemEventHandler):
    def __init__(self):
        self.last_modified = datetime.now()
    
    def on_modified(self, event):
        if datetime.now() - self.last_modified < timedelta(seconds=5):
            return
        else:
            self.last_modified = datetime.now()
        # do something, eg. call your function to process the image
        print(f"File {event.src_path} got modified")
        prediction.main(event.src_path)

if __name__ == "__main__":
    startObserver()