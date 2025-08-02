import queue

class ModelPool:    
    def __init__(self, model_class, model_arguments, pool_size=2):
        self.pool = queue.Queue()
        self.model_class = model_class
        self.model_arguments = model_arguments
        self.pool_size = pool_size
        self._initialize_pool()
    
    def _initialize_pool(self):
        for pool_index in range(self.pool_size):
            print(f"Loading model instance {pool_index+1}/{self.pool_size}...")
            model = self.model_class(**self.model_arguments)
            self.pool.put(model)
        if self.pool_size == 1:
            print("Model pool initialized with 1 instance")
        else:
            print(f"Model pool initialized with {self.pool_size} instances")
    
    def acquire(self, timeout=30):
        try:
            return self.pool.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError("No model available in pool")
    
    def release(self, model):
        self.pool.put(model)
    
    @property
    def available_count(self):
        return self.pool.qsize()