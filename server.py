import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

import json
from flask import Flask, request, jsonify
from services.qwen_vl_service import QwenVLServer

QUEUE_SIZE = 50

application = Flask(__name__)

print("Initializing Qwen-VL server...")
qwen_vl_server = QwenVLServer(
    pool_size=1,
    queue_size=QUEUE_SIZE,
    number_of_workers=1
)

@application.route("/process", methods=["POST"])
def process():
    try:
        task = request.form.get("task")
        if not task:
            return jsonify({"error": "Task parameter required"}), 400
        image_urls = request.form.get("image_urls", "[]")
        image_urls = json.loads(image_urls)
        text = request.form.get("text", "")
        result = qwen_vl_server.submit_request({
            "task": task,
            "image_urls": image_urls,
            "text": text
        })
        if result["success"]:
            return jsonify(result["result"])
        else:
            return jsonify({"error": result["error"]}), 500  
    except Exception as error:
        print(f"Request processing error: {str(error)}")
        return jsonify({"error": str(error)}), 500

@application.route("/health", methods=["GET"])
def health():
    status = qwen_vl_server.get_status()
    is_healthy = status["queue_size"] < QUEUE_SIZE * 0.8
    if is_healthy:
        return jsonify({
            "status": "healthy",
            "details": status
        }), 200
    else:
        return jsonify({
            "status": "degraded",
            "details": status
        }), 503

from PIL import Image
@application.route("/test", methods=["POST"])
def test():
    try:
        images = []
        image_count = 0
        while f'image_{image_count}' in request.files:
            image_file = request.files[f'image_{image_count}']
            image_buffer = image_file.stream
            image = Image.open(image_buffer)
            image = image.convert('RGB')
            images.append(image)
            image_count += 1
        prompt = request.form.get("prompt")
        if not prompt:
            return jsonify({"error": "Prompt parameter required"}), 400        
        max_tokens = int(request.form.get("max_tokens", 512))
        result = qwen_vl_server.submit_request({
            "task": "test",
            "images": images,
            "prompt": prompt,
            "max_tokens": max_tokens
        })
        if result["success"]:
            return jsonify({"output": result["result"]})
        else:
            return jsonify({"error": result["error"]}), 500  
    except Exception as error:
        print(f"Test request error: {str(error)}")
        return jsonify({"error": str(error)}), 500

if __name__ == "__main__":
    application.run(
        host="0.0.0.0",
        port=9003,
        threaded=True
    )