import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
import time
import torch
from PIL import Image

text = "Okay so this is my black Chanel bag... it's the classic flap, um, I think it's called the medium or maybe the large? I always forget. It's the quilted one with the diamond pattern... caviar leather I think is what they call it. Got it for my 30th birthday from my husband, we were in Paris actually. It has the gold hardware, the chain... you can wear it crossbody or doubled up on the shoulder. Oh and it has the CC turnlock thing on the front. I usually use this for special occasions, like dinner dates or events. It's in pretty good condition, there's like a tiny scratch on the back but you can barely see it. Oh wait, I should mention it has the burgundy lining inside... maroon? Whatever that dark red color is called. There's a little pocket inside with a zipper and then an open pocket too. Still have the authenticity card somewhere."
# text = "Okay these are my Louboutins... the black ones. Super high heels, like crazy high. Um, they have that platform part at the front which makes them actually possible to walk in, sort of. They're leather, really nice smooth leather. I think these are called... Daffodils? Daffodile? Something like that, I can never remember the exact name. But they're the ones with the really thick platform. Got them for my anniversary... wait no, it was my promotion actually. I remember because I wore them to the celebration dinner and my feet were killing me by the end of the night. They have the red bottom obviously, that's like the signature thing. The heel is probably... I don't know, 5 inches? 6 inches? They're definitely over 5. I've only worn them maybe three or four times total, they're more like special occasion shoes, you know? For when I need to feel really dressed up. They're in great condition still, I always put them back in the dust bag. Oh and they're a size 38.5, which is like an 8 in US sizes I think. Classic black pumps but like, the fancy version. They go with everything but honestly they're not the most comfortable things in the world. Still love them though."

def prepare_inputs_for_model(inputs, model):
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

def extract_images_from_messages(messages):
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

def distill_item_details(text=""):
    prompt = f"""
        Convert this transcription into a bulleted list of facts about the item. 

        Transcription: {text}
    """
    return prompt

def extract_visual_features(text=""):
    prompt = f"""
        The user describes this item as "{text}"

        Convert this item description into a bulleted list of observable facts about the item.
    """
    return prompt

QUANTIZATION_CONFIGURATION = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_compute_dtype = torch.float16
)

model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", 
    quantization_config=QUANTIZATION_CONFIGURATION,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", 
    min_pixels=min_pixels, 
    max_pixels=max_pixels
)

image = Image.open("samples/heel.png")

messages = [
    {
        "role": "user",
        "content": [
            # {
            #     "type": "image",
            #     "image": "samples/output_1.png",
            # },
            # {
            #     "type": "image",
            #     "image": "samples/output_2.png",
            # },
            # {
            #     "type": "image",
            #     "image": "samples/output_3.png",
            # },
            # {
            #     "type": "text", 
            #     "text": extract_all_context(text),
            # },

            # {
            #     "type": "text",
            #     "text": f"""
            #         Write a concise description with item type, brand, color, material, and key visual features.

                    # - The item is a pair of black Louboutin high-heeled shoes.
                    # - They are super high heels, approximately 5-6 inches tall.
                    # - The shoes feature a platform at the front, making them wearable despite their height.    
                    # - The material is leather, described as smooth and of high quality.
                    # - The exact name of the shoe model is uncertain, possibly "Daffodils" or something similar.
                    # - The shoes were purchased for a promotion celebration dinner.
                    # - The red soles are a signature feature of Louboutin shoes.
                    # - The heel height is likely around 5-6 inches, confirmed to be over 5 inches.
                    # - The wearer has worn them only three or four times in total.
                    # - They are considered special occasion shoes due to their discomfort.
                    # - Despite being uncomfortable, the wearer still loves them.
                    # - The shoes are in excellent condition and stored in a dust bag.
                    # - The size is 38.5, equivalent to an 8 in US sizes.
                    # - They are classic black pumps but considered a fancy version.
                    # - They are versatile and can match various outfits.
            #     """
            # }
            {
                "type": "image",
                "image": image,
            },
            {
                "type": "text",
                "text": f"""
                    The user describes this item as "My red bottoms".

                    Convert this item description into a bulleted list of observable facts about the item.
                """
            }
        ],
    }
]

text = processor.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)

image_inputs = extract_images_from_messages(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = prepare_inputs_for_model(inputs, model)

start_time = time.time()

generated_ids = model.generate(**inputs, max_new_tokens=512)

inference_time = time.time() - start_time

new_tokens = []
for input_tokens, full_output in zip(inputs["input_ids"], generated_ids):
    input_length = len(input_tokens)
    new_output = full_output[input_length:]
    new_tokens.append(new_output)

output_text = processor.batch_decode(
    new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print(output_text[0])
print(f"\nInference time: {inference_time:.2f} seconds")

def format_to_paragraph(bulleted_text):
    lines = bulleted_text.strip().split("\n")
    cleaned_lines = []
    for line in lines:
        cleaned = line.lstrip("- ").strip()
        cleaned = cleaned.replace("•", "").strip()
        if cleaned:
            cleaned_lines.append(cleaned)
    paragraph = " ".join(cleaned_lines)
    while "  " in paragraph:
        paragraph = paragraph.replace("  ", " ")    
    paragraph = paragraph.replace('"', "'")
    return paragraph

paragraph_output = format_to_paragraph(output_text[0])

print("\nFormatted Paragraph:")
print(paragraph_output)