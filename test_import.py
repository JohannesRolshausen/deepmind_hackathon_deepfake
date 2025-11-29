try:
    import serpapi
    print("Successfully imported serpapi")
except ImportError as e:
    print(f"Failed to import serpapi: {e}")

try:
    import google.generativeai as genai_old
    print("Successfully imported google.generativeai")
except ImportError as e:
    print(f"Failed to import google.generativeai: {e}")

try:
    from core.schemas import AggregatedContext, TaskInput
    print("Successfully imported core.schemas")
except ImportError as e:
    print(f"Failed to import core.schemas: {e}")

try:
    from steps.base import BaseStep
    print("Successfully imported steps.base")
except ImportError as e:
    print(f"Failed to import steps.base: {e}")

try:
    from steps.reverse_image_search import ReverseImageSearch
    print("Successfully imported steps.reverse_image_search")
except ImportError as e:
    print(f"Failed to import steps.reverse_image_search: {e}")

try:
    from steps.synthid_detection import SynthIDDetection
    print("Successfully imported steps.synthid_detection")
except ImportError as e:
    print(f"Failed to import steps.synthid_detection: {e}")

try:
    from steps.visual_forensics import VisualForensicsAgent
    print("Successfully imported steps.visual_forensics")
except ImportError as e:
    print(f"Failed to import steps.visual_forensics: {e}")

try:
    from steps.judge_system import JudgeSystem
    print("Successfully imported steps.judge_system")
except ImportError as e:
    print(f"Failed to import steps.judge_system: {e}")

try:
    from steps.ai_metadata_analyzer import AIMetadataAnalyzer
    print("Successfully imported steps.ai_metadata_analyzer")
except ImportError as e:
    print(f"Failed to import steps.ai_metadata_analyzer: {e}")
