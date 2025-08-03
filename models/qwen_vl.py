import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from PIL import Image

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
QUANTIZATION_CONFIGURATION = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_compute_dtype = torch.float16
)
MINIMUM_PIXELS = 256 * 28 * 28
MAXIMUM_PIXELS = 1280 * 28 * 28

class QwenVLModel:
    def __init__(self):
        print(f"Loading Qwen-VL model: {MODEL_ID}")
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype = torch.float16,
            quantization_config = QUANTIZATION_CONFIGURATION,
            attn_implementation = "flash_attention_2",
            device_map = "auto",
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            min_pixels = MINIMUM_PIXELS,
            max_pixels = MAXIMUM_PIXELS
        )
        print("Model loaded successfully")

    def _prepare_inputs_for_model(self, inputs, model):
        prepared = {}
        parameters = model.parameters()
        model_dtype = next(parameters).dtype    
        for key, tensor in inputs.items():
            if torch.is_floating_point(tensor):
                prepared[key] = tensor.to(
                    device=model.device, 
                    dtype=model_dtype
                )
            else:
                prepared[key] = tensor.to(model.device)
        return prepared

    def _extract_images_from_messages(self, messages):
        images = []
        for message in messages:
            if message["role"] == "user":
                content = message["content"]
                if isinstance(content, list):
                    for item in content:
                        if item["type"] == "image":
                            image = item["image"]
                            images.append(image)
        if not images:
            return None
        else:
            return images

    def _prepare_image_messages(self, images, text_prompt):
        content = []
        for image in images:
            content.append({
                "type": "image",
                "image": image,
            })
        content.append({
            "type": "text",
            "text": text_prompt
        })
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        return messages

    def _prepare_text_messages(self, text_prompt):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": text_prompt
                    }
                ]
            }
        ]
        return messages

    def _generate(self, messages, max_new_tokens = 512):
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        images = self._extract_images_from_messages(messages)
        inputs = self.processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens = max_new_tokens,
            )
        generated_ids_trimmed = []
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids):
            input_tokens_length = len(input_ids)
            new_tokens = output_ids[input_tokens_length:]
            generated_ids_trimmed.append(new_tokens)
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        return output_text[0]

    def distill(self, text = ""):
        return "distilled text"

    def extract_features(self, images, text = ""):
        return "extracted features"

    def generate_alternatives(self, images, text = ""):
        return "alternative texts"

    def process(self, images, text, task):
        if task == "distill":
            return self.distill(text)
        elif task == "extract_features":
            return self.extract_features(images, text)
        elif task == "generate_alternatives":
            return self.generate_alternatives(images, text)
        else:
            raise ValueError(f"Unknown task: {task}")