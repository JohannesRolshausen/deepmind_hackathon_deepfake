from abc import ABC, abstractmethod
from core.schemas import TaskInput, StepResult

class BaseStep(ABC):
    @abstractmethod
    def run(self, input_data: TaskInput) -> StepResult:
        pass
