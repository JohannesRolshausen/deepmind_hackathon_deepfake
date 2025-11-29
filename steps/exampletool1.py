from steps.base import BaseStep
from core.schemas import TaskInput, StepResult

class ImageMetadataExtractor(BaseStep):
    def run(self, input_data: TaskInput) -> StepResult:
        # PUT YOUR LOGIC IN HERE
        # EXAMPLE
        print(f"  [Tool1] Extracting metadata from  {input_data.image_path}...")
        
        return StepResult(
            source="ImageMetadataExtractor",
            content={
                "resolution": "1920x1080",
                "format": "png",
                "has_faces": True
            }
        )
