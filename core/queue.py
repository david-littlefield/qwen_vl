import queue
import threading
import time
import uuid

class RequestQueue:    
    def __init__(self, max_size = 100):
        self.queue = queue.Queue(maxsize=max_size)
        self.results = {}
        self.result_lock = threading.Lock()
        
    def submit(self, request_id, request_data):
        try:
            self.queue.put((request_id, request_data), block=False)
            return True
        except queue.Full:
            return False

    def get_next(self, timeout = None):
        return self.queue.get(timeout=timeout)

    def store_result(self, request_id, result):
        with self.result_lock:
            self.results[request_id] = result

    def get_result(self, request_id, timeout = 60):
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.result_lock:
                if request_id in self.results:
                    return self.results.pop(request_id)
            time.sleep(0.1)
        raise TimeoutError(f"Request {request_id} timed out")
    
    @property
    def size(self):
        return self.queue.qsize()