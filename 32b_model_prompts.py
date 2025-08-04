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
