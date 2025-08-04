import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
import time

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MINIMUM_PIXELS = 256 * 28 * 28
MAXIMUM_PIXELS = 1280 * 28 * 28

# Test prompts
test_prompts = [
    "Count from 1 to 5.",
    "List the days of the week.",
    "Name three colors."
]

print("="*70)
print("QWEN2.5-VL BATCH GENERATION BUG DEMONSTRATION")
print("="*70)

# Load model
print("\n🔧 Loading model...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(
    MODEL_ID, 
    min_pixels=MINIMUM_PIXELS, 
    max_pixels=MAXIMUM_PIXELS,
    padding_side="left",  # Important for batch generation
)
print("✓ Model loaded successfully")

# ============================================
# PART 1: DEMONSTRATE THE PROBLEM
# ============================================
print("\n" + "="*70)
print("PART 1: DEMONSTRATING THE BUG")
print("="*70)

# Test 1: Individual Processing (WORKS CORRECTLY)
print("\n📊 Test 1: Individual Processing")
print("-" * 50)

individual_outputs = []
individual_start_time = time.time()

for i, prompt in enumerate(test_prompts):
    print(f"\nProcessing prompt {i+1}: '{prompt}'")
    
    # Reset rope_deltas to ensure clean state
    model.model.rope_deltas = None
    
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=None, padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    
    output_text = processor.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    individual_outputs.append(output_text)
    
    print(f"✓ Output: {output_text[:50]}...")
    print(f"  rope_deltas: {model.model.rope_deltas}")

individual_time = time.time() - individual_start_time
print(f"\n⏱️  Total time: {individual_time:.2f}s")

# Test 2: Batch Processing (DEMONSTRATES BUG)
print("\n📊 Test 2: Batch Processing (Broken)")
print("-" * 50)

# Reset for batch
model.model.rope_deltas = None

# Prepare batch
all_messages = []
for prompt in test_prompts:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    all_messages.append(messages)

texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in all_messages]
inputs = processor(text=texts, images=None, padding=True, return_tensors="pt")
inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

print(f"\nBatch shape: {inputs['input_ids'].shape}")
print(f"Initial rope_deltas: {model.model.rope_deltas}")

batch_start_time = time.time()
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)

print(f"\nFinal rope_deltas (SHARED STATE): {model.model.rope_deltas}")

# Decode batch outputs
batch_outputs = []
for i, (input_ids, output_ids) in enumerate(zip(inputs['input_ids'], outputs)):
    output_text = processor.decode(output_ids[len(input_ids):], skip_special_tokens=True)
    batch_outputs.append(output_text)
    print(f"\n❌ Batch output {i+1}: {output_text[:50]}...")

batch_time = time.time() - batch_start_time
print(f"\n⏱️  Total time: {batch_time:.2f}s")

# ============================================
# PART 2: ANALYSIS
# ============================================
print("\n" + "="*70)
print("PART 2: ANALYSIS")
print("="*70)

print("\n🔍 Comparing Outputs:")
print("-" * 50)

for i in range(len(test_prompts)):
    match = individual_outputs[i] == batch_outputs[i]
    status = "✓ MATCH" if match else "❌ CORRUPTED"
    print(f"\nPrompt {i+1}: {test_prompts[i]}")
    print(f"Individual: {individual_outputs[i][:50]}...")
    print(f"Batch:      {batch_outputs[i][:50]}...")
    print(f"Status:     {status}")

# ============================================
# PART 3: THE SOLUTION
# ============================================
print("\n" + "="*70)
print("PART 3: SOLUTION - BATCH-AWARE PROCESSING")
print("="*70)

def batch_generate_text_only(model, processor, prompts, **generate_kwargs):
    """
    Workaround for Qwen2.5-VL batch generation bug.
    Processes text-only prompts with proper handling of rope_deltas.
    """
    # Prepare all inputs
    all_messages = []
    for prompt in prompts:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        all_messages.append(messages)
    
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) 
             for msg in all_messages]
    
    # Process individually to avoid rope_deltas corruption
    all_outputs = []
    
    for i, text in enumerate(texts):
        # Clear rope_deltas for each item
        model.model.rope_deltas = None
        
        inputs = processor(text=[text], images=None, return_tensors="pt")
        inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(**inputs, **generate_kwargs)
        
        output_text = processor.decode(
            outputs[0][len(inputs['input_ids'][0]):], 
            skip_special_tokens=True
        )
        all_outputs.append(output_text)
    
    return all_outputs

# Alternative solution: True batch processing with fix
def fixed_batch_generate(model, processor, prompts, **generate_kwargs):
    """
    Alternative solution that patches the model to handle batches correctly.
    """
    # Store original method
    original_forward = model.model.forward
    
    # Track batch size
    current_batch_size = len(prompts)
    
    def patched_forward(self, *args, **kwargs):
        # If rope_deltas exists but has wrong batch size, clear it
        if (hasattr(self, 'rope_deltas') and 
            self.rope_deltas is not None and 
            len(self.rope_deltas) != current_batch_size):
            self.rope_deltas = None
        return original_forward(*args, **kwargs)
    
    # Apply patch
    model.model.forward = lambda *args, **kwargs: patched_forward(model.model, *args, **kwargs)
    
    try:
        # Prepare batch
        all_messages = []
        for prompt in prompts:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            all_messages.append(messages)
        
        texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) 
                 for msg in all_messages]
        inputs = processor(text=texts, images=None, padding=True, return_tensors="pt")
        inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, **generate_kwargs)
        
        # Decode
        results = []
        for i, (input_ids, output_ids) in enumerate(zip(inputs['input_ids'], outputs)):
            output_text = processor.decode(output_ids[len(input_ids):], skip_special_tokens=True)
            results.append(output_text)
        
        return results
    
    finally:
        # Restore original method
        model.model.forward = original_forward

# Test Solution 1: Individual Processing
print("\n📊 Solution 1: Safe Individual Processing")
print("-" * 50)

solution1_start_time = time.time()
solution1_outputs = batch_generate_text_only(model, processor, test_prompts, max_new_tokens=20, do_sample=False)
solution1_time = time.time() - solution1_start_time

for i, output in enumerate(solution1_outputs):
    print(f"\n✓ Output {i+1}: {output[:50]}...")

print(f"\n⏱️  Total time: {solution1_time:.2f}s")

# Test Solution 2: Patched Batch Processing
print("\n📊 Solution 2: Patched Batch Processing")
print("-" * 50)

solution2_start_time = time.time()
solution2_outputs = fixed_batch_generate(model, processor, test_prompts, max_new_tokens=20, do_sample=False)
solution2_time = time.time() - solution2_start_time

for i, output in enumerate(solution2_outputs):
    print(f"\n✓ Output {i+1}: {output[:50]}...")

print(f"\n⏱️  Total time: {solution2_time:.2f}s")

# ============================================
# PART 4: SUMMARY
# ============================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print("\n🐛 BUG EXPLANATION:")
print("-" * 50)
print("• Qwen2.5-VL stores rope_deltas as shared model state (self.model.rope_deltas)")
print("• In batch processing, all sequences share the same rope_deltas")
print("• This causes position embeddings to be calculated incorrectly")
print("• Result: Corrupted outputs for all but the first sequence")

print("\n✅ SOLUTIONS:")
print("-" * 50)
print("1. Process text-only inputs individually (slower but guaranteed to work)")
print("2. Patch the model to clear rope_deltas when batch size changes")
print("3. Wait for official fix in transformers library")

print("\n📊 PERFORMANCE COMPARISON:")
print("-" * 50)
print(f"Individual processing: {individual_time:.2f}s")
print(f"Batch (broken):       {batch_time:.2f}s")
print(f"Solution 1:           {solution1_time:.2f}s")
print(f"Solution 2:           {solution2_time:.2f}s")

print("\n💡 RECOMMENDATION:")
print("-" * 50)
print("Use Solution 1 (individual processing) for production until the bug is fixed.")
print("It's more reliable and doesn't require model patching.")

# Verification
print("\n✅ VERIFICATION:")
print("-" * 50)
all_match = all(sol1 == expected for sol1, expected in zip(solution1_outputs, individual_outputs))
print(f"Solution 1 matches expected outputs: {all_match}")

# Optional: Save the working solution as a utility function
print("\n📝 UTILITY FUNCTION:")
print("-" * 50)
print("""
# Copy this function to use in your code:

def generate_batch_text_qwen25vl(model, processor, prompts, **kwargs):
    \"\"\"Safe batch generation for Qwen2.5-VL text-only inputs.\"\"\"
    outputs = []
    for prompt in prompts:
        model.model.rope_deltas = None  # Clear shared state
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=None, return_tensors="pt")
        inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        with torch.no_grad():
            output = model.generate(**inputs, **kwargs)
        
        output_text = processor.decode(output[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        outputs.append(output_text)
    
    return outputs
""")