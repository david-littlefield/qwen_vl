import requests
import io
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from core.base_server import ModelServer
from models.qwen_vl import QwenVLModel

class QwenVLServer(ModelServer):    
    def __init__(self, pool_size = 1, queue_size = 50, number_of_workers = 1):
        super().__init__(
            model_class=QwenVLModel,
            model_arguments={},
            pool_size=pool_size,
            queue_size=queue_size,
            number_of_workers=number_of_workers
        )
        self.download_pool = ThreadPoolExecutor(max_workers=5)

    def _download_image(self, url):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            image_buffer = io.BytesIO(response.content)
            image = Image.open(image_buffer)
            return image.convert("RGB")
        except Exception as error:
            print(f"Failed to download image from {url}: {str(error)}")
            raise

    def _load_images(self, image_urls):
        futures = []
        if image_urls:
            for url in image_urls:
                future = self.download_pool.submit(self._download_image, url)
                futures.append(future)        
        images = []
        for future_index, future in enumerate(futures):
            try:
                image = future.result()
                images.append(image)
            except Exception as error:
                print(f"Failed to load image {future_index}: {str(error)}")
                raise
        return images
    
    def _process_request(self, model, request_data):
        task = request_data["task"]
        image_urls = request_data.get('image_urls', [])
        images = self._load_images(image_urls)
        text = request_data.get("text", "")
        return model.process(images, text, task)