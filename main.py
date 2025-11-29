# main.py
import argparse
from typing import List
from core.schemas import AggregatedContext, TaskInput
from core.llm import call_llm
from steps.base import BaseStep
from steps.exampletool1 import ImageMetadataExtractor
from steps.exampletool2 import TextAnalyzer
from steps.visual_forensics import VisualForensicsAgent
from steps.judge_system import JudgeSystem

def main():
    # 1. User Input (Simuliert oder via CLI)
    parser = argparse.ArgumentParser(description="Hackathon Gemini Pipeline")
    parser.add_argument("--img", type=str, default="./example_data/anypic.png", help="Path to the image file")
    parser.add_argument("--text", type=str, default="instagram", help="instagram picture")
    
    args = parser.parse_args()
    
    image_path = args.img
    user_text = args.text
    
    print(f"Starting pipeline for img: {image_path} and text: '{user_text}'\n")
    
    # 2. Setup Context & Input
    task_input = TaskInput(image_path=image_path, text=user_text)
    context = AggregatedContext(task_input=task_input)
    
    # 3. Registriere Steps
    steps: List[BaseStep] = [
        ImageMetadataExtractor(),
        TextAnalyzer(),
        # VisualForensicsAgent(),
        JudgeSystem()
    ]
    
    # 4. Sequenzielle Ausführung
    for step in steps:
        try:
            result = step.run(task_input)
            context.results.append(result)
            print(f"✅ {result.source} done.")
        except Exception as e:
            print(f"❌ error in step {step.__class__.__name__}: {e}")

    # 5. Finaler LLM Call
    print("\n--- FINAL LLM CALL ---")
    final_answer = call_llm(context)
    
    try:
        import json
        data = json.loads(final_answer)
        print(f"\n🔍 Deepfake Probability: {data.get('probability_score')}%")
        print(f"📝 Explanation: {data.get('explanation')}")
    except json.JSONDecodeError:
        print(f"Raw Output: {final_answer}")

if __name__ == "__main__":
    main()