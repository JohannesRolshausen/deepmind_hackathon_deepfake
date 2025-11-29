# core/schemas.py
from pydantic import BaseModel
from typing import List, Optional, Any

# Input for all steps in pipeleine
class TaskInput(BaseModel):
    image_path: str
    text: Optional[str] = None

# return value of single step
class StepResult(BaseModel):
    source: str
    content: Any # STRING OR JSON

# collected context for final llm call
class AggregatedContext(BaseModel):
    task_input: TaskInput
    results: List[StepResult] = []