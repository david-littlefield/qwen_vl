#!/usr/bin/env python3
import requests
import io
import re

MODEL_SERVER = "http://localhost:9003"
IMAGE_PATH = "samples/purse.png"
SHORT_TRANSCRIPTION = "My Chanel bag"
LONG_TRANSCRIPTION = "Okay so this is my black Chanel bag... it's the classic flap, um, I think it's called the medium or maybe the large? I always forget. It's the quilted one with the diamond pattern... caviar leather I think is what they call it. Got it for my 30th birthday from my husband, we were in Paris actually. It has the gold hardware, the chain... you can wear it crossbody or doubled up on the shoulder. Oh and it has the CC turnlock thing on the front. I usually use this for special occasions, like dinner dates or events. It's in pretty good condition, there's like a tiny scratch on the back but you can barely see it. Oh wait, I should mention it has the burgundy lining inside... maroon? Whatever that dark red color is called. There's a little pocket inside with a zipper and then an open pocket too. Still have the authenticity card somewhere."

# todo: 
# - save the current 32b model prompts
# - check if old 7b prompts were on githib
# - if not, redo prompts for 7b model
# - try different approaches to compensate for 7b model's limitations
# - try breaking up tasks into smaller steps
# - for example, iterating through each sentence in consolidated text
# - breaking down each sentence into atomic facts
# - look into batch processing capabilities of the model

def extract_stated_facts(transcription=""):
    prompt = f"""
        TRANSCRIPTION: 
        {transcription}

        TASK: 
        List what the user explicitly stated about the item.
        
        RULES:
        - Include ONLY information directly stated
        - Remove uncertainty and qualifiers
        - NO interpretations
        - NO clarifications
        - NO explanatory phrases
        - NO commentary

        FORMAT:
        - Fact one about the item
        - Fact two about the item
        - Continue for all stated facts

        OUTPUT:
        Bulleted list
    """
    return prompt

def process_stated_facts(stated_facts):
    processed = []
    texts = re.sub(r"\-", "", stated_facts)
    texts = re.sub(r"\.", "", texts)
    texts = texts.split("\n")
    for text in texts:
        text = text.strip()
        if text:
            processed.append(text)
    return processed

def distill_item_description(stated_facts):
    prompt = f"""
        STATED FACTS: 
        {stated_facts}
        
        TASK:
        Create the shortest possible item description (2-4 words maximum).

        RULES:
        - STOP after [brand] and [item]
        - NO markdown formatting
        - NO special characters
        
        FORMAT:
        - Pick the format that fits best
        - A [item].
        - A [brand] [item]. 

        OUTPUT:
        Sentence
    """
    return prompt

def extract_visual_features(item_description=""):
    prompt = f"""
        ITEM DESCRIPTION: 
        {item_description}
        
        TASK: 
        List ONLY what you can directly see

        RULES:
        - Include ALL visible elements
        - BE specific and detailed
        - NO markdown formatting
        - NO interpretations
        - NO clarifications
        - NO explanatory phrases
        - NO commentary
        
        FORMAT: 
        A [item description] with [feature], [feature], [feature], etc.

        OUTPUT:
        Sentence
    """
    return prompt

def consolidate_item_details(stated_facts, item_description, visual_features):
    prompt = f"""
        STATED FACTS: 
        {stated_facts}
        
        ITEM DESCRIPTION: 
        {item_description}

        VISUAL FEATURES: 
        {visual_features}

        TASK: 
        Combine all information into one natural paragraph.

        RULES:
        - ONLY use words that appear in the sources above
        - If a word isn't in the sources, DON'T use it
        - NO NEW WORDS AT ALL
        - Remove duplicate information
        - Factual tone only
        - NO markdown formatting

        PROCESS:
        1. Start with item description
        2. Add visual details
        3. Add stated facts
        4. Connect with simple words: is, has, was, with, and

        OUTPUT: 
        Paragraph
    """
    return prompt


def extract_atomic_facts(consolidated_text):
    prompt = f"""
        CONSOLIDATED TEXT: 
        {consolidated_text}

        TASK:
        Extract ALL atomic facts from the consolidated text.

        RULES:
        1. Each fact must be a complete sentence
        2. One attribute per fact
        3. Use exact words from the text
        4. NO interpretations
        5. NO preambles

        FORMAT: 
        - First fact
        - Second fact
        - Continue for all facts

        OUTPUT:
        Bulleted list
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
    

    print("\n1. Extracting stated facts...")
    stated_facts = call_model(
        extract_stated_facts(transcription)
    )
    results['stated_facts'] = stated_facts
    print(f"\n{stated_facts}")


    print("\n2. Processing stated facts...")
    processed_facts = process_stated_facts(stated_facts)
    for fact in processed_facts:
        print(f"• {fact}")

    # print("\n2. Distilling to item description...")
    # item_description = call_model(
    #     distill_item_description(stated_facts)
    # )
    # results['item_description'] = item_description
    # print(f"\n{item_description}")
    
    # print("\n3. Extracting visual features...")
    # visual_features = call_model(
    #     extract_visual_features(item_description), 
    #     image_path
    # )
    # results['visual_features'] = visual_features
    # print(f"\n{visual_features}")

    # print("\n4. Consolidating all information...")
    # consolidated = call_model(
    #     consolidate_item_details(
    #         results['stated_facts'],
    #         results['item_description'], 
    #         results['visual_features']
    #     )
    # )
    # results['consolidated_description'] = consolidated
    # print(f"\n{consolidated}")

    # print("\n5. Extracting atomic facts...")
    # atomic_facts = call_model(
    #     extract_atomic_facts(consolidated)
    # )
    # results['atomic_facts'] = atomic_facts
    # print(f"\n{atomic_facts}")

    return results

def main():
    # print("\n### TESTING WITH SHORT TRANSCRIPTION ###")
    # results_short = run_full_pipeline(SHORT_TRANSCRIPTION, IMAGE_PATH)

    print("\n\n### TESTING WITH LONG TRANSCRIPTION ###")
    results_long = run_full_pipeline(LONG_TRANSCRIPTION, IMAGE_PATH)

    # print("\n\n" + "="*60)
    # print("PIPELINE SUMMARY")
    # print("="*60)
    # print("\nShort transcription results:")
    # for key, value in results_short.items():
    #     print(f"\n{key}:\n{value}")
    
    # print("\n\nLong transcription results:")
    # for key, value in results_long.items():
    #     print(f"\n{key}:\n{value}")

if __name__ == "__main__":
    main()
