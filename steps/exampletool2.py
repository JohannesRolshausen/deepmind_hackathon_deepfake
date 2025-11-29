from steps.base import BaseStep
from core.schemas import TaskInput, StepResult

class TextAnalyzer(BaseStep):
    def run(self, input_data: TaskInput) -> StepResult:
        # PUT YOUR LOGIC IN HERE
        # EXAMPLE
        text = input_data.text or "NO TEXT"
        print(f"  [Tool2] Analyzing Text: '{text}'...")
        
        return StepResult(
            source="TextAnalyzer",
            content=f"Text length: {len(text)} characters. Sentiment: Neutral."
        )
