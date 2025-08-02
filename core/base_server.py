import threading
import uuid
import queue
from abc import ABC, abstractmethod
from .pool import ModelPool
from .queue import RequestQueue

class ModelServer(ABC):    
    def __init__(
        self, 
        model_class,
        model_arguments,
        pool_size = 1,
        queue_size = 50,
        number_of_workers = 1
    ):
        self.model_pool = ModelPool(model_class, model_arguments, pool_size)
        self.request_queue = RequestQueue(queue_size)
        self.workers = []
        self.running = True
        self._start_workers(number_of_workers)

    def _start_workers(self, number_of_workers):
        for worker_index in range(number_of_workers):
            arguments = tuple([worker_index])
            worker = threading.Thread(
                target=self._worker_loop,
                args=arguments,
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        print(f"Started {number_of_workers} worker threads")

    def _worker_loop(self, worker_id):
        while self.running:
            try:
                request_id, request_data = self.request_queue.get_next(timeout=1)
                self._handle_request(worker_id, request_id, request_data)
            except queue.Empty:
                continue
    
    def _handle_request(self, worker_id, request_id, request_data):
        model = None
        try:
            model = self.model_pool.acquire()
            print(f"Worker {worker_id} processing {request_id}")
            result = self._process_request(model, request_data)
            self.request_queue.store_result(request_id, {
                "success": True,
                "result": result
            })
        except Exception as error:
            print(f"Worker {worker_id} error: {str(error)}")
            self.request_queue.store_result(request_id, {
                "success": False,
                "error": str(error)
            })
        finally:
            if model:
                self.model_pool.release(model)
    
    @abstractmethod
    def _process_request(self, model, request_data):
        pass
    
    def submit_request(self, request_data):
        request_id = str(uuid.uuid4())        
        if not self.request_queue.submit(request_id, request_data):
            raise Exception("Request queue is full")
        return self.request_queue.get_result(request_id)

    def get_status(self):
        return {
            "queue_size": self.request_queue.size,
            "models_available": self.model_pool.available_count,
            "pending_results": len(self.request_queue.results),
            "workers": len(self.workers)
        }
    
    def shutdown(self):
        print("Shutting down model server...")
        self.running = False
        for worker in self.workers:
            worker.join(timeout=5)