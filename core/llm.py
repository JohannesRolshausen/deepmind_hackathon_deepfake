import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from core.schemas import AggregatedContext

from typing import List, Any, Optional

load_dotenv()

def query_llm(prompt: str, images: Optional[List[Any]] = None) -> str:
    """
    Generic function to query the Gemini API with a given prompt and optional images.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not found in environment variables."

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')

    try:
        content = [prompt]
        if images:
            content.extend(images)
            
        response = model.generate_content(content)
        return response.text
    except Exception as e:
        return f"Error calling Gemini API: {e}"

import PIL.Image

def call_llm(context: AggregatedContext) -> str:
    """
    Calls the Gemini API to analyze the context and return a deepfake probability and explanation.
    """
    context_json = context.model_dump_json(indent=2)
    
    # Try to load the image
    image_path = context.task_input.image_path
    img = None
    try:
        if image_path:
            img = PIL.Image.open(image_path)
    except Exception as e:
        print(f"Warning: Could not load image for final judgment: {e}")

    prompt = f"""
    You are a world-class Digital Forensics Expert and Deepfake Detection System.
    
    Your task is to analyze the provided image (if available) and the results from various forensic tools to determine the probability that the content is AI-generated (a deepfake).
    
    ### Input Data
    - **Visual Evidence**: The attached image.
    - **Forensic Tool Results**:
    {context_json}
    
    ### Instructions
    1. **Synthesize Evidence**: Combine your own visual analysis of the image with the findings from the tools. Look for inconsistencies, artifacts, metadata anomalies, and logical flaws.
    2. **Weigh the Tools**: 
       - `SynthIDDetection` and `VisualForensicsAgent` are high-signal tools.
       - `JudgeSystem` provides a debate summary.
       - `ReverseImageSearch` helps identify context.
    3. **Determine Probability**: Assign a score from 0 to 100 (0 = Definitely Real, 100 = Definitely Fake).
    4. **Explain**: Provide a clear, professional explanation citing specific evidence.
    
    ### Output Format
    Return a valid JSON object with EXACTLY this structure:
    {{
        "probability_score": <int between 0 and 100>,
        "explanation": "<string explanation>"
    }}
    Do not include markdown formatting like ```json.
    """

    response_text = query_llm(prompt, images=[img] if img else None)
    
    # Clean up potential markdown code blocks
    cleaned_response = response_text.replace("```json", "").replace("```", "").strip()
    return cleaned_response
