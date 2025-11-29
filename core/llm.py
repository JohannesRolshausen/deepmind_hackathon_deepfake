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

def call_llm(context: AggregatedContext) -> str:
    """
    Calls the Gemini API to analyze the context and return a deepfake probability and explanation.
    """
    context_json = context.model_dump_json(indent=2)
    
    prompt = f"""
    Analyze the following task input and tool results to determine if the content is a deepfake.
    
    Context:
    {context_json}
    
    Return a JSON object with the following structure:
    {{
        "probability_score": <int between 0 and 100>,
        "explanation": "<string explanation>"
    }}
    Do not include markdown formatting like ```json.
    """

    return query_llm(prompt)
