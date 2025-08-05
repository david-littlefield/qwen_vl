import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
import time
import torch
from PIL import Image

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

def extract_stated_facts(transcription=""):
    prompt = f"""
        TRANSCRIPTION:
        {transcription}

        TASK:
        List EVERYTHING that was explicitly stated.

        RULES:
        Do NOT summarize, interpret, or omit anything.
        Maintain the original context and meaning.

        FORMAT:
        Output the details as a single paragraph starting with "The item is...".

        OUTPUT:
    """
    messages = [
        {
            "role": "system",
            "content": "You are a detail-obsessed stenographer."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
            ],
        }
    ]
    return messages

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MINIMUM_PIXELS = 256 * 28 * 28
MAXIMUM_PIXELS = 1280 * 28 * 28
QUANTIZATION_CONFIGURATION = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_compute_dtype = torch.float16
)
TRANSCRIPTIONS = [
    "My Chanel bag",
    "My Louboutins",
    "My Apple Watch",
    # "Okay so this is my black Chanel bag... it's the classic flap, um, I think it's called the medium or maybe the large? I always forget. It's the quilted one with the diamond pattern... caviar leather I think is what they call it. Got it for my 30th birthday from my husband, we were in Paris actually. It has the gold hardware, the chain... you can wear it crossbody or doubled up on the shoulder. Oh and it has the CC turnlock thing on the front. I usually use this for special occasions, like dinner dates or events. It's in pretty good condition, there's like a tiny scratch on the back but you can barely see it. Oh wait, I should mention it has the burgundy lining inside... maroon? Whatever that dark red color is called. There's a little pocket inside with a zipper and then an open pocket too. Still have the authenticity card somewhere.",
    # "Okay these are my Louboutins... the black ones. Super high heels, like crazy high. Um, they have that platform part at the front which makes them actually possible to walk in, sort of. They're leather, really nice smooth leather. I think these are called... Daffodils? Daffodile? Something like that, I can never remember the exact name. But they're the ones with the really thick platform. Got them for my anniversary... wait no, it was my promotion actually. I remember because I wore them to the celebration dinner and my feet were killing me by the end of the night. They have the red bottom obviously, that's like the signature thing. The heel is probably... I don't know, 5 inches? 6 inches? They're definitely over 5. I've only worn them maybe three or four times total, they're more like special occasion shoes, you know? For when I need to feel really dressed up. They're in great condition still, I always put them back in the dust bag. Oh and they're a size 38.5, which is like an 8 in US sizes I think. Classic black pumps but like, the fancy version. They go with everything but honestly they're not the most comfortable things in the world. Still love them though.",
    # "This is my everyday watch... the Apple Watch. It's the Series 8 I think? Or maybe 7, I can't remember. The 45mm one because I like the bigger screen. I got the aluminum case in midnight - basically black but Apple doesn't call it black for some reason. Has the sport band, also in black... or midnight, whatever. I use it mainly for fitness tracking, you know, closing my rings and all that. Also great for notifications so I don't have to pull out my phone constantly. Battery lasts about a day and a half if I'm not using it too much. Oh and I have a few other bands I switch out sometimes - a leather one for when I want to look a bit nicer, and a braided solo loop that's super comfortable. The screen has always-on display which is nice. No major scratches or anything, I've been pretty careful with it."
]
IMAGE_PATHS = [
    "samples/purse.png", 
    "samples/heel.png", 
    "samples/watch.png",
]

images = []

for path in IMAGE_PATHS:
    image = Image.open(path)
    images.append(image)

prompts = [
    extract_stated_facts(TRANSCRIPTIONS[0]),
    extract_stated_facts(TRANSCRIPTIONS[1]),
    extract_stated_facts(TRANSCRIPTIONS[2]),
]

print("=" * 70)
print("QWEN VL: INDIVIDUAL vs BATCH PROCESSING COMPARISON")
print("=" * 70)

print("\n🔧 Initializing model...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, 
    quantization_config=QUANTIZATION_CONFIGURATION,
    torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(
    MODEL_ID, 
    min_pixels=MINIMUM_PIXELS, 
    max_pixels=MAXIMUM_PIXELS,
    use_fast=True,
    padding_side="left",
)

print("✓ Model loaded successfully")

# print("\n🔥 Running warm-up...")
# warm_up_start_time = time.time()
# warm_up_image = images[0]
# warm_up_prompt = "Describe this item briefly."
# warm_up_messages = [
#     {
#         "role": "user",
#         "content": [
#             # {
#             #     "type": "image", 
#             #     "image": warm_up_image
#             # },
#             {
#                 "type": "text", 
#                 "text": warm_up_prompt
#             }
#         ],
#     }
# ]
# warm_up_text = processor.apply_chat_template(
#     warm_up_messages, 
#     tokenize=False, 
#     add_generation_prompt=True
# )
# warm_up_inputs = processor(
#     text=[warm_up_text],
#     # images=[warm_up_image],
#     images=None,
#     padding=True,
#     return_tensors="pt",
# )
# warm_up_inputs = prepare_inputs_for_model(warm_up_inputs, model)
# _ = model.generate(**warm_up_inputs, max_new_tokens=512)
# warm_up_inference_time = time.time() - warm_up_start_time
# print(f"✓ Warm-up completed in {warm_up_inference_time:.2f}s")

# # ============================================
# # APPROACH 1: INDIVIDUAL PROCESSING
# # ============================================
# print("\n" + "=" * 70)
# print("🔵 INDIVIDUAL PROCESSING")
# print("=" * 70)
# individual_results = []
# individual_start_time = time.time()
# for index, (image, prompt) in enumerate(zip(images, prompts)):
#     print(f"\n  Processing request {index + 1}/{len(images)}...")    
#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 # {
#                 #     "type": "image",
#                 #     "image": image,
#                 # },
#                 {
#                     "type": "text",
#                     "text": prompt
#                 },
#             ],
#         }
#     ]    
#     start_time = time.time()
#     text = processor.apply_chat_template(
#         messages, 
#         tokenize=False, 
#         add_generation_prompt=True
#     )
#     image_inputs = extract_images_from_messages(messages)
#     inputs = processor(
#         text=[text],
#         images=image_inputs,
#         padding=True,
#         return_tensors="pt",
#     )
#     inputs = prepare_inputs_for_model(
#         inputs, 
#         model
#     )
#     generated_ids = model.generate(
#         **inputs, 
#         max_new_tokens=512
#     )
#     new_tokens = []
#     for input_tokens, full_output in zip(inputs["input_ids"], generated_ids):
#         input_length = len(input_tokens)
#         new_output = full_output[input_length:]
#         new_tokens.append(new_output)
#     output_texts = processor.batch_decode(
#         new_tokens, 
#         skip_special_tokens=True, 
#         clean_up_tokenization_spaces=False
#     )
#     inference_time = time.time() - start_time
#     individual_results.append({
#         'output': output_texts[0],
#         'time': inference_time
#     })
#     print(f"  ✓ Completed in {inference_time:.2f}s")
#     print(f"  Result: {output_texts[0]}")
# individual_inference_time = time.time() - individual_start_time
# print(f"\n📊 Individual Processing Summary:")
# print(f"  - Total time: {individual_inference_time:.2f}s")
# print(f"  - Average time per request: {individual_inference_time/len(images):.2f}s")
# print(f"  - Throughput: {len(images)/individual_inference_time:.2f} requests/second")

# ============================================
# APPROACH 2: BATCH PROCESSING
# ============================================
print("\n" + "=" * 70)
print("🟢 BATCH PROCESSING")
print("=" * 70)
batch_start_time = time.time()
all_texts = []
all_messages = []
all_images = []
for transcription in TRANSCRIPTIONS:
    messages = extract_stated_facts(transcription)
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    all_texts.append(text)
    all_messages.append(messages)
for messages in all_messages:
    extracted_images = extract_images_from_messages(messages)
    if extracted_images:
        all_images.extend(extracted_images)
if not all_images:
    all_images = None
print(f"\n  Processing batch of {len(all_texts)} requests...")
inputs = processor(
    text=all_texts,
    images=all_images,
    padding=True,
    return_tensors="pt",
)
inputs = prepare_inputs_for_model(inputs, model)
generated_ids = model.generate(**inputs, max_new_tokens=1024)
new_tokens = []
for input_tokens, full_output in zip(inputs["input_ids"], generated_ids):
    input_length = len(input_tokens)
    new_output = full_output[input_length:]
    new_tokens.append(new_output)
output_texts = processor.batch_decode(
    new_tokens, 
    skip_special_tokens=True, 
    clean_up_tokenization_spaces=False
)
batch_inference_time = time.time() - batch_start_time
print(f"  ✓ Batch completed in {batch_inference_time:.2f}s")
print(f"\n  Batch Results:")
for index, text in enumerate(output_texts):
    print(f"  Result {index + 1}: {text}")
print(f"\n📊 Batch Processing Summary:")
print(f"  - Total time: {batch_inference_time:.2f}s")
print(f"  - Average time per request: {batch_inference_time/len(images):.2f}s")
print(f"  - Throughput: {len(images)/batch_inference_time:.2f} requests/second")

# # ============================================
# # PERFORMANCE COMPARISON
# # ============================================
# print("\n" + "=" * 70)
# print("📈 PERFORMANCE COMPARISON")
# print("=" * 70)
# time_saved = individual_inference_time - batch_inference_time
# efficiency_gain = ((individual_inference_time - batch_inference_time) / individual_inference_time) * 100
# print(f"\n🏁 Results:")
# print(f"  Individual Processing: {individual_inference_time:.2f}s total")
# print(f"  Batch Processing:      {batch_inference_time:.2f}s total")
# print(f"  ⏰ Time saved:       {time_saved:.2f}s ({efficiency_gain:.1f}%)")


