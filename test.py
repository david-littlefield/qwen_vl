#!/usr/bin/env python3
import requests
import io

MODEL_SERVER = "http://localhost:9003"
IMAGE_PATH = "samples/purse.png"
SHORT_TRANSCRIPTION = "My Chanel bag"
LONG_TRANSCRIPTION = "Okay so this is my black Chanel bag... it's the classic flap, um, I think it's called the medium or maybe the large? I always forget. It's the quilted one with the diamond pattern... caviar leather I think is what they call it. Got it for my 30th birthday from my husband, we were in Paris actually. It has the gold hardware, the chain... you can wear it crossbody or doubled up on the shoulder. Oh and it has the CC turnlock thing on the front. I usually use this for special occasions, like dinner dates or events. It's in pretty good condition, there's like a tiny scratch on the back but you can barely see it. Oh wait, I should mention it has the burgundy lining inside... maroon? Whatever that dark red color is called. There's a little pocket inside with a zipper and then an open pocket too. Still have the authenticity card somewhere.",

def extract_stated_facts(transcription=""):
    prompt = f"""
        Transcription: {transcription}

        List ONLY what the user explicitly stated about the item.
        Do not interpret, assume, or guess what the item might be.

        Format: This is a fact, fact, fact, etc.

        Output: Paragraph
    """
    return prompt


def distill_item_description(stated_facts):
    prompt = f"""
        Stated facts: {stated_facts}
        
        Reduce to 2-4 words ONLY. Pick ONE:
        STOP after: [color] [brand] [item]
        
        Pick format based on what's most identifying:
        - A [item].
        - A [brand] [item]. 
        - A [color] [item].
        - A [color] [brand] [item].

        Output: Sentence
    """
    return prompt

def extract_visual_features(item_description=""):
    prompt = f"""
        Item description: "{item_description}"
        
        List ALL visual features you can see.
        
        Format: A [item description] with [feature], [feature], [feature], etc.

        Output: Paragraph
    """
    return prompt

def call_model(prompt, image_path=None):
    try:
        if image_path:
            with open(image_path, "rb") as file:
                files = {
                    "image_0": (image_path, file, "image/png")
                }
                data = {
                    "prompt": prompt,
                }
                response = requests.post(
                    f"{MODEL_SERVER}/test",
                    files=files,
                    data=data
                )
        else:
            response = requests.post(
                f"{MODEL_SERVER}/test",
                data={
                    "prompt": prompt,
                }
            )
        result = response.json()
        if "output" in result:
            return result["output"].strip()
        else:
            raise Exception(f"Model error: {result.get('error', 'Unknown error')}")
    except Exception as error:
        print(f"Error calling model: {error}")

def run_full_pipeline(transcription, image_path):
    print("="*60)
    print("RUNNING FULL PIPELINE")
    print("="*60)
    
    results = {}
    
    print("\n1. Extracting stated facts from transcription...")
    stated_facts = call_model(extract_stated_facts(transcription))
    results['stated_facts'] = stated_facts
    print(f"Result: {stated_facts}")
    
    print("\n2. Distilling to item description...")
    item_description = call_model(distill_item_description(stated_facts))
    results['item_description'] = item_description
    print(f"Result: {item_description}")
    
    print("\n3. Extracting visual features from image...")
    visual_features = call_model(extract_visual_features(item_description), image_path)
    results['visual_features'] = visual_features
    print(f"Result: {visual_features}")
    
    return results

def main():
    print("\n### TESTING WITH SHORT TRANSCRIPTION ###")
    results_short = run_full_pipeline(SHORT_TRANSCRIPTION, IMAGE_PATH)

    print("\n\n### TESTING WITH LONG TRANSCRIPTION ###")
    results_long = run_full_pipeline(LONG_TRANSCRIPTION, IMAGE_PATH)

    print("\n\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    print("\nShort transcription results:")
    for key, value in results_short.items():
        print(f"\n{key}:\n{value}")
    
    print("\n\nLong transcription results:")
    for key, value in results_long.items():
        print(f"\n{key}:\n{value}")

if __name__ == "__main__":
    main()
