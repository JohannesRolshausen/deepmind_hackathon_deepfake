from steps.base import BaseStep
from core.schemas import TaskInput, StepResult
from typing import Optional, List, Dict, Any
import os

try:
    from sd_parsers import ParserManager, Eagerness
    from PIL import Image
    SD_PARSERS_AVAILABLE = True
except ImportError:
    SD_PARSERS_AVAILABLE = False


class AIMetadataAnalyzer(BaseStep):
    """
    Analyzes images for AI-generation metadata using sd-parsers library.
    Returns a raw metadata report for further analysis by an LLM.
    """

    def __init__(self):
        """Initialize the AI metadata analyzer with sd-parsers."""
        if not SD_PARSERS_AVAILABLE:
            print("  [AIMetadataAnalyzer] Warning: sd-parsers not installed. Run: pip install sd-parsers")
            self.parser_manager = None
        else:
            # Initialize ParserManager with DEFAULT mode (balanced performance/thoroughness)
            self.parser_manager = ParserManager(eagerness=Eagerness.DEFAULT)

    def run(self, input_data: TaskInput) -> StepResult:
        """
        Main entry point for AI metadata analysis.

        Args:
            input_data: TaskInput containing image_path

        Returns:
            StepResult with comprehensive raw metadata report
        """
        print(f"  [AIMetadataAnalyzer] Extracting metadata from {input_data.image_path}...")

        # Check if sd-parsers is available
        if not SD_PARSERS_AVAILABLE:
            return StepResult(
                source="AIMetadataAnalyzer",
                content={
                    'status': 'error',
                    'error_type': 'dependency_missing',
                    'message': 'sd-parsers library not installed. Run: pip install sd-parsers'
                }
            )

        # Validate file exists
        if not os.path.exists(input_data.image_path):
            return StepResult(
                source="AIMetadataAnalyzer",
                content={
                    'status': 'error',
                    'error_type': 'file_not_found',
                    'message': f'Image file not found: {input_data.image_path}'
                }
            )

        try:
            # 1. Extract SD/AI Metadata
            sd_metadata = self._extract_sd_metadata(input_data.image_path)

            # 2. Extract File System / Basic Image Metadata
            file_metadata = self._extract_file_metadata(input_data.image_path)

            # Combine into a raw report
            content = {
                'status': 'success',
                'analysis_type': 'metadata_extraction',
                'sd_parsers_report': sd_metadata,
                'file_system_report': file_metadata
            }

            return StepResult(source="AIMetadataAnalyzer", content=content)

        except Exception as e:
            # Ultimate fallback - never crash the pipeline
            print(f"  [AIMetadataAnalyzer] Unexpected error: {e}")
            return StepResult(
                source="AIMetadataAnalyzer",
                content={
                    'status': 'error',
                    'error_type': 'unexpected_error',
                    'message': str(e)
                }
            )

    def _extract_sd_metadata(self, image_path: str) -> Dict[str, Any]:
        """
        Extract detailed AI metadata using sd-parsers.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing all extracted SD metadata
        """
        try:
            prompt_info = self.parser_manager.parse(image_path)
            
            if not prompt_info:
                return {'metadata_found': False}

            # Extract generator information
            generator = str(prompt_info.generator) if hasattr(prompt_info, 'generator') and prompt_info.generator else None

            # Extract prompts
            positive_prompt = None
            negative_prompt = None

            if hasattr(prompt_info, 'prompts') and prompt_info.prompts:
                # Combine all positive prompts
                positive_prompt = ', '.join([p.value for p in prompt_info.prompts if hasattr(p, 'value')])

            if hasattr(prompt_info, 'negative_prompts') and prompt_info.negative_prompts:
                # Combine all negative prompts
                negative_prompt = ', '.join([p.value for p in prompt_info.negative_prompts if hasattr(p, 'value')])

            # Extract models
            models = []
            if hasattr(prompt_info, 'models') and prompt_info.models:
                models = [str(m) for m in prompt_info.models]

            # Extract detailed sampler information
            samplers = []
            if hasattr(prompt_info, 'samplers') and prompt_info.samplers:
                for sampler in prompt_info.samplers:
                    sampler_dict = {
                        'name': str(sampler.name) if hasattr(sampler, 'name') else None,
                        'cfg_scale': sampler.cfg_scale if hasattr(sampler, 'cfg_scale') else None,
                        'seed': sampler.seed if hasattr(sampler, 'seed') else None,
                        'steps': sampler.steps if hasattr(sampler, 'steps') else None,
                        'parameters': sampler.parameters if hasattr(sampler, 'parameters') else {}
                    }
                    samplers.append(sampler_dict)
            
            # Extract raw metadata if available (often contains the full generation string)
            raw_metadata = {}
            if hasattr(prompt_info, 'metadata'):
                raw_metadata = prompt_info.metadata

            return {
                'metadata_found': True,
                'generator': generator,
                'prompts': {
                    'positive': positive_prompt,
                    'negative': negative_prompt
                },
                'models': models,
                'samplers': samplers,
                'raw_metadata': raw_metadata
            }

        except Exception as e:
            print(f"  [AIMetadataAnalyzer] Warning: Failed to parse SD metadata: {e}")
            return {'metadata_found': False, 'error': str(e)}

    def _extract_file_metadata(self, image_path: str) -> Dict[str, Any]:
        """
        Extract basic file and image metadata using PIL, filtering for insightful tags.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing file system and image format metadata
        """
        try:
            from PIL import ExifTags
            
            file_stats = os.stat(image_path)
            file_size = file_stats.st_size
            
            with Image.open(image_path) as img:
                image_format = img.format
                image_mode = img.mode
                image_size = img.size
                
                # Insightful EXIF tags to look for
                # 271: Make, 272: Model, 305: Software, 306: DateTime
                # 36867: DateTimeOriginal, 36868: DateTimeDigitized
                # 42034: LensModel, 37510: UserComment
                insightful_tags = {
                    271: 'Make',
                    272: 'Model',
                    305: 'Software',
                    306: 'DateTime',
                    36867: 'DateTimeOriginal',
                    36868: 'DateTimeDigitized',
                    42034: 'LensModel',
                    37510: 'UserComment',
                    33432: 'Copyright',
                    315: 'Artist'
                }

                # Get raw EXIF data and filter
                exif_data = {}
                if hasattr(img, 'getexif'):
                    exif_raw = img.getexif()
                    if exif_raw:
                        for tag_id, value in exif_raw.items():
                            if tag_id in insightful_tags:
                                tag_name = insightful_tags[tag_id]
                                # Clean up value if needed (decode bytes)
                                if isinstance(value, bytes):
                                    try:
                                        value = value.decode()
                                    except:
                                        value = str(value)
                                exif_data[tag_name] = str(value)
                            
                            # Also check for ExifTags mapping if not in our explicit list but potentially interesting
                            elif tag_id in ExifTags.TAGS:
                                tag_name = ExifTags.TAGS[tag_id]
                                if tag_name in ['BodySerialNumber', 'CameraOwnerName', 'LensSpecification']:
                                     exif_data[tag_name] = str(value)
                
                # Get raw PNG info and filter
                # PNG info often contains 'parameters' (SD), 'Software', 'Comment'
                png_info = {}
                if hasattr(img, 'info'):
                    for k, v in img.info.items():
                        # Filter out large binary blobs like ICC profiles or thumbnails unless needed
                        if k in ['icc_profile', 'exif']: 
                            continue
                        
                        # Keep text-based info
                        if isinstance(v, (str, int, float)):
                            png_info[str(k)] = str(v)
                        elif isinstance(v, bytes):
                            # Try to decode short bytes, skip long ones
                            if len(v) < 1000:
                                try:
                                    png_info[str(k)] = v.decode()
                                except:
                                    pass

                return {
                    'has_exif': bool(exif_data),
                    'has_png_info': bool(png_info),
                    'image_format': str(image_format),
                    'image_mode': str(image_mode),
                    'image_size': list(image_size),
                    'file_size_bytes': file_size,
                    'exif_data': exif_data,
                    'png_info': png_info
                }
                
        except Exception as e:
            print(f"  [AIMetadataAnalyzer] Warning: Failed to extract file metadata: {e}")
            return {
                'error': str(e),
                'file_size_bytes': os.path.getsize(image_path) if os.path.exists(image_path) else 0
            }
