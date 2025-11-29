import os
import json
from core.llm import call_llm
from core.schemas import AggregatedContext, TaskInput, StepResult

def verify():
    # Setup mock data
    image_path = "./example_data/anypic.png"
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    task_input = TaskInput(image_path=image_path, text="instagram")
    context = AggregatedContext(task_input=task_input)
    
    # Add some mock step results
    context.results.append(StepResult(
        source="MockStep",
        content={"analysis": "Suspicious artifacts found."}
    ))

    print("Calling call_llm...")
    result = call_llm(context)
    print("\nResult:")
    print(result)

    try:
        data = json.loads(result)
        print(f"\nParsed Probability: {data.get('probability_score')}")
    except json.JSONDecodeError:
        print("\nFailed to parse JSON")

if __name__ == "__main__":
    verify()
