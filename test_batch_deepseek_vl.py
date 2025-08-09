import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from deepseek_vl.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl.utils.io import load_pil_images

# Specify the path to the model
model_path = "deepseek-ai/deepseek-vl2-small"
vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

# Load model with quantization - no device_map needed
print("Loading quantized model...")
vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    low_cpu_mem_usage=True, 
    torch_dtype=torch.bfloat16, 
    device_map="cuda"
)
vl_gpt.eval()

# DO NOT use .to("cuda") or .eval() with quantized models
print("Model loaded successfully!")

# Single image conversation example with technical description prompt
conversation = [
    {
        "role": "<|User|>",
        "content": """<image>
Provide a technical product description of the item in the image. 

Follow these rules:
- State only visible, objective features
- Use neutral, descriptive language without subjective adjectives
- Include: material, color, finish, shape, hardware details, visible components
- Format: "The item is [type] with [primary features]. It features [secondary details]. [Additional observable characteristics]."
- Do not include: opinions, marketing language, emotional descriptors, quality judgments, or assumed functionality
- Describe only what is physically visible in the image""",
        "images": ["./samples/heels-Photoroom.png"],  # Change to your image path
    },
    {"role": "<|Assistant|>", "content": ""},
]

# Alternative: Reference format for object localization (returns bounding boxes)
# conversation = [
#     {
#         "role": "<|User|>",
#         "content": "<image>\n<|ref|>The purse in the center.<|/ref|>.",
#         "images": ["./samples/purse-Photoroom.png"],
#     },
#     {"role": "<|Assistant|>", "content": ""},
# ]

# Load images and prepare for inputs
print("Processing image...")
pil_images = load_pil_images(conversation)

# Prepare inputs
prepare_inputs = vl_chat_processor(
    conversations=conversation,
    images=pil_images,
    force_batchify=True,
    system_prompt=""
)

# Get the device where the quantized model is located
model_device = next(vl_gpt.parameters()).device
prepare_inputs = prepare_inputs.to(model_device)

# Run image encoder to get the image embeddings
print("Generating response...")
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

# IMPORTANT: Use vl_gpt.language.generate() NOT vl_gpt.generate()
# This is from the official DeepSeek-VL2 documentation
outputs = vl_gpt.language.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=512,
    do_sample=False,  # Greedy decoding for consistent results
    use_cache=True
)

# Decode the output
answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False)

# Print the conversation and response
print("=" * 70)
print("CONVERSATION:")
print(prepare_inputs['sft_format'][0])
print("\nRESPONSE:")
print(answer)
print("=" * 70)

















# Example batch processing
# IMAGE_PATHS = [
#     "./samples/purse-Photoroom.png",
#     "./samples/heels-Photoroom.png",
#     "./samples/watch-Photoroom.png"
# ]
# ITEM_DESCRIPTIONS = [
#     "a Chanel handbag",
#     "Christian Louboutin high-heeled shoes",
#     "an Apple Watch"
# ]
# 
# batch_results = process_batch(IMAGE_PATHS, ITEM_DESCRIPTIONS)

# import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" 


# import torch
# from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# from deepseek_vl.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
# from deepseek_vl.utils.io import load_pil_images

# QUANTIZATION_CONFIGURATION = BitsAndBytesConfig(
#     load_in_4bit = True,
#     bnb_4bit_compute_dtype = torch.float16
# )

# # specify the path to the model
# model_path = "deepseek-ai/deepseek-vl2-small"
# vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
# tokenizer = vl_chat_processor.tokenizer

# vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     quantization_config=QUANTIZATION_CONFIGURATION,
#     trust_remote_code=True,
#     torch_dtype=torch.bfloat16,
#     device_map="cuda",
# )
# # vl_gpt = vl_gpt.to("cuda")
# # vl_gpt.eval()

# ## single image conversation example
# # conversation = [
# #     {
# #         "role": "<|User|>",
# #         "content": "<image>\n<|ref|>The purse in the center.<|/ref|>.",
# #         "images": ["./samples/purse-Photoroom.png"],
# #     },
# #     {"role": "<|Assistant|>", "content": ""},
# # ]

# conversation = [
#     {
#         "role": "<|User|>",
#         "content": """<image>
# Provide a technical product description of the item in the image. 

# Follow these rules:
# - State only visible, objective features
# - Use neutral, descriptive language without subjective adjectives
# - Include: material, color, finish, shape, hardware details, visible components
# - Format: "The item is [type] with [primary features]. It features [secondary details]. [Additional observable characteristics]."
# - Do not include: opinions, marketing language, emotional descriptors, quality judgments, or assumed functionality
# - Describe only what is physically visible in the image""",
#         "images": ["./samples/heels-Photoroom.png"],
#     },
#     {"role": "<|Assistant|>", "content": ""},
# ]

# ## multiple images (or in-context learning) conversation example
# # conversation = [
# #     {
# #         "role": "User",
# #         "content": "<image_placeholder>A dog wearing nothing in the foreground, "
# #                    "<image_placeholder>a dog wearing a santa hat, "
# #                    "<image_placeholder>a dog wearing a wizard outfit, and "
# #                    "<image_placeholder>what's the dog wearing?",
# #         "images": [
# #             "images/dog_a.png",
# #             "images/dog_b.png",
# #             "images/dog_c.png",
# #             "images/dog_d.png",
# #         ],
# #     },
# #     {"role": "Assistant", "content": ""}
# # ]

# # load images and prepare for inputs
# pil_images = load_pil_images(conversation)
# prepare_inputs = vl_chat_processor(
#     conversations=conversation,
#     images=pil_images,
#     force_batchify=True,
#     system_prompt=""
# )
# # .to(vl_gpt.device)
# prepare_inputs = prepare_inputs.to("cuda")

# # run image encoder to get the image embeddings
# inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

# # run the model to get the response
# outputs = vl_gpt.generate(
#     inputs_embeds=inputs_embeds,
#     attention_mask=prepare_inputs.attention_mask,
#     pad_token_id=tokenizer.eos_token_id,
#     bos_token_id=tokenizer.bos_token_id,
#     eos_token_id=tokenizer.eos_token_id,
#     max_new_tokens=512,
#     do_sample=False,
#     use_cache=True
# )

# answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
# print(f"{prepare_inputs['sft_format'][0]}", answer)




# from transformers import AutoModelForCausalLM, AutoModel, AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
# import time
# import torch
# from PIL import Image

# from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
# from deepseek_vl2.utils.io import load_pil_images

# def prepare_inputs_for_model(inputs, model):
#     prepared = {}
#     parameters = model.parameters()
#     model_dtype = next(parameters).dtype    
#     for key, tensor in inputs.items():
#         if torch.is_floating_point(tensor):
#             prepared[key] = tensor.to(
#                 device=model.device, 
#                 dtype=model_dtype
#             )
#         else:
#             prepared[key] = tensor.to(model.device)
#     return prepared

# def extract_images_from_messages(messages):
#     images = []
#     for message in messages:
#         if message["role"] == "user":
#             content = message["content"]
#             if isinstance(content, list):
#                 for item in content:
#                     if item["type"] == "image":
#                         image = item["image"]
#                         images.append(image)
#     if not images:
#         return None
#     else:
#         return images

# # 1. extract stated facts
# # 2. 

# def extract_stated_facts(transcription=""):
#     prompt = f"""
#         TRANSCRIPTION:
#         {transcription}

#         TASK:
#         List EVERYTHING that was explicitly stated in the transcription

#         RULES:
#         Do NOT summarize, interpret, or omit anything
#         Maintain the original context and meaning

#         FORMAT:
#         Output the facts as a single paragraph

#         OUTPUT:
#         The item is
#     """
#     messages = [
#         {
#             "role": "system",
#             "content": "You are a detail-obsessed stenographer"
#         },
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": prompt
#                 },
#             ],
#         }
#     ]
#     return messages

# def distill_item_description(image, stated_facts):
#     # prompt = f"""
#     #     STATED FACTS: 
#     #     {stated_facts}
        
#     #     TASK:
#     #     Extract the brand and product type from the stated facts

#     #     RULES:
#     #     First identify any brand mentions
#     #     Include the nicknames and slang
#     #     Normalize to the official brand name
#     #     Identify the PRODUCT TYPE
 
#     #     FORMAT:
#     #     NO markdown
#     #     NO special characters
 
#     #     OUTPUT:
#     #     The item is a [BRAND] [PRODUCT TYPE]
#     # """
#     # prompt = f"""
#     #     STATED FACTS: 
#     #     {stated_facts}
        
#     #     STEP 1 - BRAND IDENTIFICATION:
#     #     Identify all brand mentions in the stated facts
#     #     Look for official names, nicknames, and slang terms
#     #     Consider visual brand markers in the image
        
#     #     STEP 2 - NORMALIZATION:
#     #     Normalize to the official brand name and extract product type
        
#     #     RULES:
#     #     Use both stated facts and image for context
#     #     Include nicknames and slang
#     #     Normalize to official brand names
#     #     Identify the specific product type
#     #     Use proper grammar with articles (a/an)
 
#     #     FORMAT:
#     #     No markdown 
#     #     No special characters

#     #     Choose ONE format based on brand identification:
#     #     The item is [a/an] [brand] [product type]
#     #     The item is [a/an] [product type]

#     #     OUTPUT:
#     #     The item is
#     # """
#     prompt = f"""
#         Provide a technical product description of the item in the image. 
        
#         Follow these rules:
#         - State only visible, objective features
#         - Use neutral, descriptive language without subjective adjectives
#         - Include: material, color, finish, shape, hardware details, visible components
#         - Format: "The item is [type] with [primary features]. It features [secondary details]. [Additional observable characteristics]."
#         - Do not include: opinions, marketing language, emotional descriptors, quality judgments, or assumed functionality
#         - Describe only what is physically visible in the image
#     """
#     messages = [
#         {
#             "role": "system",
#             "content": "You are an authentication expert. Describe only verifiable visual elements using industry-standard terminology."
#         },
#         {
#             "role": "user",
#             "content": [
#                   {
#                     "type": "image",
#                     "image": image
#                 },
#                 {
#                     "type": "text",
#                     "text": prompt
#                 },
#             ],
#         }
#     ]
#     return messages

# def extract_visual_features(image, item_description=""):
#     # prompt = f"""
#     #     ITEM DESCRIPTION: 
#     #     {item_description}

#     #     INSTRUCTIONS:
#     #     1. Use the item description and image to identify the item in the image
#     #     2. Write an empirical visual description for the identified item
#     #     3. Include only distinguishing physical features that can be seen
#     #     4. Remove everything that is not a visual descriptor
#     #     5. Remove all markdown, special characters, and bullet points
#     #     6. Present the description in a paragraph

#     #     RULES:
#     #     - No assumptions 
#     #     - No interpretations
#     #     - No commentary
#     #     - No explanations
#     #     - No opinions
#     #     - No conclusions
#     #     - No summaries

#     #     OUTPUT:
#     #     The item is 
#     # """
#     # prompt = f"""
#     #     ITEM DESCRIPTION: 
#     #     {item_description}

#     #     STEP 1 - IDENTIFICATION:
#     #     Use the item description and image to identify what item this is
        
#     #     STEP 2 - FEATURE DESCRIPTION:
#     #     Describe every visible feature of the identified item
        
#     #     RULES:
#     #     Use the item description as context
#     #     Include only what you can directly see
#     #     Prioritize brand-identifying features
#     #     Use proper grammar with articles (a/an)
 
#     #     FORMAT:
#     #     No assumptions, interpretations, commentary, explanations, or opinions
#     #     No markdown, special characters, or bullet points
#     #     Start with "The item is a/an [identified item] that has"

#     #     OUTPUT:
#     #     The item is 
#     # """
#     prompt = f"""
#         ITEM DESCRIPTION: 
#         {item_description}

#         TASK:
#         Provide a technical product description of the item in the image. 

#         RULES:
#         - State only visible, objective features
#         - Use neutral, descriptive language without subjective adjectives
#         - Include: material, color, finish, shape, hardware details, visible components
#         - Format: "The item is [type] with [primary features]. It features [secondary details]. [Additional observable characteristics]."
#         - Do not include: opinions, marketing language, emotional descriptors, quality judgments, or assumed functionality
#         - Describe only what is physically visible in the image

#         EXAMPLE:
#         The item is a pair of high-heeled pumps with black leather or leather-like upper material and a glossy finish. They feature red lacquered soles, stiletto heels, and platform front soles. The shoes have a closed, rounded toe design and beige/cream colored interior lining. Visible text branding appears on the interior footbed showing "Christian Louboutin" and "Paris". The heel height appears to be approximately 5-6 inches with a platform sole of approximately 1-2 inches. Both shoes are shown at an angle displaying the profile and sole construction.
    
#         OUTPUT:
#         The item is
#     """
#     messages = [
#         {
#             "role": "system",
#             "content": "You are a product authentication specialist"
#         },
#         {
#             "role": "user",
#             "content": [
#                   {
#                     "type": "image",
#                     "image": image
#                 },
#                 {
#                     "type": "text",
#                     "text": prompt
#                 },
#             ],
#         }
#     ]
#     return messages

# # MODEL_ID = "Qwen/Qwen2.5-VL-32B-Instruct"
# # MODEL_ID = "OpenGVLab/InternVL3-14B"
# # MODEL_ID = "moonshotai/Kimi-VL-A3B-Instruct"
# MODEL_ID = "deepseek-ai/deepseek-vl2-small"
# MINIMUM_PIXELS = 256 * 28 * 28
# MAXIMUM_PIXELS = 1280 * 28 * 28
# QUANTIZATION_CONFIGURATION = BitsAndBytesConfig(
#     load_in_4bit = True,
#     bnb_4bit_compute_dtype = torch.float16
# )
# TRANSCRIPTIONS = [
#     # "My Chanel bag",
#     # "My Louboutins",
#     # "My Apple Watch",
#     "Okay so this is my black Chanel bag... it's the classic flap, um, I think it's called the medium or maybe the large? I always forget. It's the quilted one with the diamond pattern... caviar leather I think is what they call it. Got it for my 30th birthday from my husband, we were in Paris actually. It has the gold hardware, the chain... you can wear it crossbody or doubled up on the shoulder. Oh and it has the CC turnlock thing on the front. I usually use this for special occasions, like dinner dates or events. It's in pretty good condition, there's like a tiny scratch on the back but you can barely see it. Oh wait, I should mention it has the burgundy lining inside... maroon? Whatever that dark red color is called. There's a little pocket inside with a zipper and then an open pocket too. Still have the authenticity card somewhere.",
#     "Okay these are my Louboutins... the black ones. Super high heels, like crazy high. Um, they have that platform part at the front which makes them actually possible to walk in, sort of. They're leather, really nice smooth leather. I think these are called... Daffodils? Daffodile? Something like that, I can never remember the exact name. But they're the ones with the really thick platform. Got them for my anniversary... wait no, it was my promotion actually. I remember because I wore them to the celebration dinner and my feet were killing me by the end of the night. They have the red bottom obviously, that's like the signature thing. The heel is probably... I don't know, 5 inches? 6 inches? They're definitely over 5. I've only worn them maybe three or four times total, they're more like special occasion shoes, you know? For when I need to feel really dressed up. They're in great condition still, I always put them back in the dust bag. Oh and they're a size 38.5, which is like an 8 in US sizes I think. Classic black pumps but like, the fancy version. They go with everything but honestly they're not the most comfortable things in the world. Still love them though.",
#     "This is my everyday watch... the Apple Watch. It's the Series 8 I think? Or maybe 7, I can't remember. The 45mm one because I like the bigger screen. I got the aluminum case in midnight - basically black but Apple doesn't call it black for some reason. Has the sport band, also in black... or midnight, whatever. I use it mainly for fitness tracking, you know, closing my rings and all that. Also great for notifications so I don't have to pull out my phone constantly. Battery lasts about a day and a half if I'm not using it too much. Oh and I have a few other bands I switch out sometimes - a leather one for when I want to look a bit nicer, and a braided solo loop that's super comfortable. The screen has always-on display which is nice. No major scratches or anything, I've been pretty careful with it.",
# ]

# IMAGE_PATHS = [
#     # "samples/purse.png", 
#     # "samples/heels.png", 
#     # "samples/watch.png",
#     "samples/purse-Photoroom.png", 
#     "samples/heels-Photoroom.png", 
#     "samples/watch-Photoroom.png",
# ]

# STATED_FACTS = [
#     # "The item is a Chanel bag.",
#     # "The item is 'My Louboutins' which refers to a pair of red-soled shoes, specifically those designed by Christian Louboutin.",  
#     # "The item is an Apple Watch.",
#     "The item is a black Chanel bag, specifically the classic flap style, which might be the medium or large size. It features a quilted diamond pattern and is made of caviar leather. The bag was purchased for the person's 30th birthday by their husband while they were in Paris. It comes with gold hardware, including a chain strap that allows for crossbody or shoulder wear. The bag has a CC turnlock closure on the front. It is typically used for special occasions such as dinner dates or events. The bag is in good condition with only a tiny scratch visible on the back. Inside, it has burgundy lining, described as maroon or another dark red color. There is a small pocket with a zipper and an open pocket inside. The authenticity card is still present, though its exact location is not specified.",
#     "The item is a pair of black Louboutin high-heeled shoes with a platform design, made of smooth leather, featuring a red sole. They are described as being approximately 5 to 6 inches tall, with a size 38.5 (US size 8). The wearer acquired them during their promotion celebration, where they experienced discomfort by the end of the evening. These shoes are considered special occasion footwear due to their high heel and platform, and they are rarely worn, having been used only three or four times. Despite their discomfort, the wearer still loves them and keeps them in excellent condition by storing them in a dust bag.",
#     "The item is an Apple Watch, specifically the Series 8 or possibly the Series 7, with a 45mm case size due to the preference for a larger screen. The watch has an aluminum case in midnight black, paired with a sport band also in midnight black. The user primarily uses it for fitness tracking, including completing daily activity goals ('rings'), and it serves as a notification device to avoid constantly checking their phone. The battery life is approximately a day and a half without heavy usage. The owner mentions having additional bands, including a leather one for a more formal appearance and a braided solo loop that is very comfortable. The screen features an always-on display, which the user appreciates, and there are no significant scratches on the watch despite careful handling.",
# ]

# ITEM_DESCRIPTIONS = [
#     "The item is a Chanel handbag.",
#     "The item is a Christian Louboutin high-heeled shoe.",
#     "The item is an Apple Watch."
# ]

# VISUAL_FEATURES = [
#     "The item has a quilted black leather exterior with a diamond pattern. It features a gold-tone chain strap and hardware, including a CC logo clasp on the front. The bag appears to be structured with a rectangular shape and a flap closure.",
#     "The item has a black leather upper with a rounded toe, a thick platform sole, and a high stiletto heel. The interior lining appears to be beige or light-colored fabric. The heel is slightly angled inward at the base.",
#     "The item has a sleek, rectangular watch face with a black bezel. It features a digital clock display with white hour, minute, and second hands on a dark gray background. The watch has a black band with a sport loop design, and there is a black crown on the right side of the watch face for adjusting settings. The overall design is modern and minimalist."
# ]

# images = []
# for path in IMAGE_PATHS:
#     image = Image.open(path)
#     images.append(image)

# print("=" * 70)
# print("QWEN VL: INDIVIDUAL vs BATCH PROCESSING COMPARISON")
# print("=" * 70)

# print("\n🔧 Initializing model...")
# # model = AutoModel.from_pretrained(
# #     MODEL_ID, 
# #     quantization_config=QUANTIZATION_CONFIGURATION,
# #     torch_dtype=torch.bfloat16,
# #     # attn_implementation="flash_attention_2",
# #     device_map="auto",
# #     trust_remote_code=True,
# # )
# # processor = AutoProcessor.from_pretrained(
# #     MODEL_ID, 
# #     min_pixels=MINIMUM_PIXELS, 
# #     max_pixels=MAXIMUM_PIXELS,
# #     use_fast=True,
# #     padding_side="left",
# #     trust_remote_code=True,
# # )

# vl_chat_processor = DeepseekVLV2Processor.from_pretrained(MODEL_ID)
# tokenizer = vl_chat_processor.tokenizer
# vl_gpt = AutoModelForCausalLM.from_pretrained(
#     MODEL_ID,
#     quantization_config=QUANTIZATION_CONFIGURATION,
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True,
#     device_map="auto"
# )

# print("✓ Model loaded successfully")

# # print("\n🔥 Running warm-up...")
# # warm_up_start_time = time.time()
# # warm_up_image = images[0]
# # warm_up_prompt = "Describe this item briefly."
# # warm_up_messages = [
# #     {
# #         "role": "user",
# #         "content": [
# #             # {
# #             #     "type": "image", 
# #             #     "image": warm_up_image
# #             # },
# #             {
# #                 "type": "text", 
# #                 "text": warm_up_prompt
# #             }
# #         ],
# #     }
# # ]
# # warm_up_text = processor.apply_chat_template(
# #     warm_up_messages, 
# #     tokenize=False, 
# #     add_generation_prompt=True
# # )
# # warm_up_inputs = processor(
# #     text=[warm_up_text],
# #     # images=[warm_up_image],
# #     images=None,
# #     padding=True,
# #     return_tensors="pt",
# # )
# # warm_up_inputs = prepare_inputs_for_model(warm_up_inputs, model)
# # _ = model.generate(**warm_up_inputs, max_new_tokens=512)
# # warm_up_inference_time = time.time() - warm_up_start_time
# # print(f"✓ Warm-up completed in {warm_up_inference_time:.2f}s")

# # # ============================================
# # # APPROACH 1: INDIVIDUAL PROCESSING
# # # ============================================
# # print("\n" + "=" * 70)
# # print("🔵 INDIVIDUAL PROCESSING")
# # print("=" * 70)
# # individual_results = []
# # individual_start_time = time.time()
# # for index, (image, prompt) in enumerate(zip(images, prompts)):
# #     print(f"\n  Processing request {index + 1}/{len(images)}...")    
# #     messages = [
# #         {
# #             "role": "user",
# #             "content": [
# #                 # {
# #                 #     "type": "image",
# #                 #     "image": image,
# #                 # },
# #                 {
# #                     "type": "text",
# #                     "text": prompt
# #                 },
# #             ],
# #         }
# #     ]    
# #     start_time = time.time()
# #     text = processor.apply_chat_template(
# #         messages, 
# #         tokenize=False, 
# #         add_generation_prompt=True
# #     )
# #     image_inputs = extract_images_from_messages(messages)
# #     inputs = processor(
# #         text=[text],
# #         images=image_inputs,
# #         padding=True,
# #         return_tensors="pt",
# #     )
# #     inputs = prepare_inputs_for_model(
# #         inputs, 
# #         model
# #     )
# #     generated_ids = model.generate(
# #         **inputs, 
# #         max_new_tokens=512
# #     )
# #     new_tokens = []
# #     for input_tokens, full_output in zip(inputs["input_ids"], generated_ids):
# #         input_length = len(input_tokens)
# #         new_output = full_output[input_length:]
# #         new_tokens.append(new_output)
# #     output_texts = processor.batch_decode(
# #         new_tokens, 
# #         skip_special_tokens=True, 
# #         clean_up_tokenization_spaces=False
# #     )
# #     inference_time = time.time() - start_time
# #     individual_results.append({
# #         'output': output_texts[0],
# #         'time': inference_time
# #     })
# #     print(f"  ✓ Completed in {inference_time:.2f}s")
# #     print(f"  Result: {output_texts[0]}")
# # individual_inference_time = time.time() - individual_start_time
# # print(f"\n📊 Individual Processing Summary:")
# # print(f"  - Total time: {individual_inference_time:.2f}s")
# # print(f"  - Average time per request: {individual_inference_time/len(images):.2f}s")
# # print(f"  - Throughput: {len(images)/individual_inference_time:.2f} requests/second")

# # ============================================
# # APPROACH 2: BATCH PROCESSING
# # ============================================
# print("\n" + "=" * 70)
# print("🟢 BATCH PROCESSING")
# print("=" * 70)
# batch_start_time = time.time()
# all_texts = []
# all_messages = []
# all_images = []
# # for image, transcription in zip(images, TRANSCRIPTIONS):
# # for image, facts in zip(images, STATED_FACTS):
# for image, item_description in zip(images, ITEM_DESCRIPTIONS):
#     # messages = extract_stated_facts(transcription)
#     # messages = distill_item_description(image, facts)
#     messages = extract_visual_features(image, item_description)
#     text = processor.apply_chat_template(
#         messages, 
#         tokenize=False, 
#         add_generation_prompt=True
#     )
#     all_texts.append(text)
#     all_messages.append(messages)
# for messages in all_messages:
#     extracted_images = extract_images_from_messages(messages)
#     if extracted_images:
#         all_images.extend(extracted_images)
# if not all_images:
#     all_images = None
# print(f"\n  Processing batch of {len(all_texts)} requests...")
# inputs = processor(
#     text=[all_texts[0]],
#     images=[all_images[0]],
#     padding=True,
#     return_tensors="pt",
# )
# inputs = prepare_inputs_for_model(inputs, model)
# generated_ids = model.generate(**inputs, max_new_tokens=2048)
# new_tokens = []
# for input_tokens, full_output in zip(inputs["input_ids"], generated_ids):
#     input_length = len(input_tokens)
#     new_output = full_output[input_length:]
#     new_tokens.append(new_output)
# output_texts = processor.batch_decode(
#     new_tokens, 
#     skip_special_tokens=True, 
#     clean_up_tokenization_spaces=False
# )
# batch_inference_time = time.time() - batch_start_time
# print(f"  ✓ Batch completed in {batch_inference_time:.2f}s")
# print(f"\n  Batch Results:")
# for index, text in enumerate(output_texts):
#     print(f"  Result {index + 1}: {text}")
# print(f"\n📊 Batch Processing Summary:")
# print(f"  - Total time: {batch_inference_time:.2f}s")
# print(f"  - Average time per request: {batch_inference_time/len(images):.2f}s")
# print(f"  - Throughput: {len(images)/batch_inference_time:.2f} requests/second")

# # # ============================================
# # # PERFORMANCE COMPARISON
# # # ============================================
# # print("\n" + "=" * 70)
# # print("📈 PERFORMANCE COMPARISON")
# # print("=" * 70)
# # time_saved = individual_inference_time - batch_inference_time
# # efficiency_gain = ((individual_inference_time - batch_inference_time) / individual_inference_time) * 100
# # print(f"\n🏁 Results:")
# # print(f"  Individual Processing: {individual_inference_time:.2f}s total")
# # print(f"  Batch Processing:      {batch_inference_time:.2f}s total")
# # print(f"  ⏰ Time saved:       {time_saved:.2f}s ({efficiency_gain:.1f}%)")


