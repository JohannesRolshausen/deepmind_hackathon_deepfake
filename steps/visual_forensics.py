"""
Visual Forensics Agent - Advanced AI-Generated Image Detection
Author: Mark
Description: Uses Vision-Language Models to detect AI-generated images based on
             2025 state-of-the-art detection indicators targeting Diffusion Model weaknesses.
"""

import json
import os
import base64
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass

# --- Standalone Definitions (previously in steps.base and core.schemas) ---

@dataclass
class TaskInput:
    image_path: str
    text: Optional[str] = None

@dataclass
class StepResult:
    source: str
    content: Any

class BaseStep:
    def run(self, input_data: TaskInput) -> StepResult:
        raise NotImplementedError

# --- End Standalone Definitions ---


class VisualForensicsAgent(BaseStep):
    """
    Visual Forensics Agent that analyzes images for signs of AI generation.

    This agent uses a Vision-Language Model with specialized prompt engineering
    to detect artifacts and anomalies specific to modern diffusion models
    (Stable Diffusion, DALL-E 3, Midjourney, Flux, etc.).
    """

    # System prompt engineered to target specific weaknesses in Diffusion Models (2025 Standard)
    SYSTEM_PROMPT = """You are an expert Visual Forensics Analyst specializing in AI-generated image detection.
Your task is to analyze images for signs of AI generation using the latest 2025 detection methodology.

## Core Detection Framework: The "Top 5 Indicators" (2025 Standard)

You MUST systematically evaluate the image against these five critical indicators, which specifically target weaknesses in modern diffusion models (Stable Diffusion XL, DALL-E 3, Midjourney v6, Flux):

### Indicator 1: Physics & Spatial Logic (The "Flux" Weakness)
**Why this matters**: Diffusion models lack true 3D understanding and physics engines. They learn visual patterns but not physical laws.

**What to look for**:
- **Clipping**: Objects passing through each other (hands through clothing, hair through shoulders)
- **Gravity violations**: Clothing that defies gravity, hair floating unnaturally
- **Object permanence failures**: Objects that appear/disappear inconsistently across similar angles
- **Depth contradictions**: Foreground objects appearing behind background elements

**Prompt Engineering Strategy**: Force the model to reason about physical causality, not just visual similarity.

### Indicator 2: Lighting & Specular Consistency
**Why this matters**: Diffusion models struggle with global illumination. They often generate lighting that "looks right" locally but fails physically.

**What to look for**:
- **Eye reflections**: Check if both eyes reflect the same light source in the same position/shape
- **Shadow direction**: Verify ALL shadows point away from the apparent light source
- **Specular highlights**: Check if shiny surfaces (glasses, jewelry, wet surfaces) reflect light consistently
- **Subsurface scattering**: Real skin shows light transmission (reddish glow on ear edges when backlit)
- **Multi-light paradoxes**: Conflicting shadow directions suggesting impossible lighting setups

**Prompt Engineering Strategy**: Trace light paths mentally - diffusion models can't do ray-tracing.

### Indicator 3: Textural Anomalies ("The Plastic Skin" Problem)
**Why this matters**: Diffusion model denoising process over-smooths high-frequency details, especially in faces.

**What to look for**:
- **Over-smoothing**: Skin that looks like plastic/wax, lacking pores and micro-textures
- **Noise reduction artifacts**: Areas that look "airbrushed" beyond what makeup could achieve
- **Uncanny valley smoothness**: Skin smoother than physically possible for the apparent age
- **Missing sub-surface scattering**: Skin that looks opaque rather than translucent
- **Texture discontinuities**: Sharp transitions between detailed and smooth areas without depth reason

**Prompt Engineering Strategy**: Compare texture detail to what's physically achievable with photography.

### Indicator 4: Anatomical & Functional Details
**Why this matters**: Beyond "counting fingers," modern models fail at understanding how things *work*.

**What to look for**:
- **Impossible poses**: Joint angles that would require broken bones
- **Asymmetric accessories**: Glasses/earrings that don't match left-right
- **Non-functional details**: Zippers that don't align, buttons without buttonholes, shoelaces that can't be tied
- **Anatomical count errors**: Yes, still check fingers/toes, but also teeth count, limb symmetry
- **Merged/missing features**: Body parts that blur together unnaturally

**Prompt Engineering Strategy**: Ask "could this physically exist and function?" not "does this look normal?"

### Indicator 5: Semantic/Contextual Gaps
**Why this matters**: Diffusion models don't understand narrative coherence or real-world constraints.

**What to look for**:
- **Temporal paradoxes**: Summer flowers blooming in snow, wrong clothing for weather
- **Cultural inconsistencies**: Architectural styles mixed impossibly, wrong signage for location
- **Background logic failures**: Crowds where people don't interact naturally, vehicles without drivers
- **Texture pattern degradation**: Brick walls that lose pattern without perspective reason, text that becomes gibberish
- **Implausible scenarios**: Situations that violate basic world knowledge

**Prompt Engineering Strategy**: Evaluate narrative coherence - AI doesn't understand stories.

## Analysis Protocol

1. **Systematic Scan**: Examine EVERY indicator category, even if early signs point to real/fake
2. **Severity Weighting**: Not all artifacts are equal - physics violations > minor texture issues
3. **Confidence Calibration**: Modern AI is VERY good - absence of obvious signs ≠ real image
4. **Aggregate Reasoning**: Multiple weak signals can indicate AI generation more than one strong signal

## Output Requirements

You MUST respond with a valid JSON object in this exact structure:
{
  "fake_probability": <float 0.0-1.0>,
  "reasoning_summary": "<concise summary of your analysis>",
  "flagged_artifacts": [
    {
      "indicator_type": "<one of the 5 indicator names>",
      "location": "<specific location in image>",
      "description": "<detailed description>",
      "severity": <float 0.0-1.0>
    }
  ],
  "confidence": <float 0.0-1.0>
}

**Scoring Guidelines**:
- fake_probability: 0.0 = certainly real, 1.0 = certainly AI-generated
- confidence: How certain you are of your assessment (0.0 = guessing, 1.0 = certain)
- severity: Impact of individual artifact (0.0 = minor, 1.0 = smoking gun)

## Critical Reminders

- **Modern AI is sophisticated**: Don't expect obvious errors like extra fingers in 2025 models
- **Context matters**: A few minor issues in a complex scene might be real photography; perfect smoothness might be AI
- **Be specific**: Instead of "looks fake," say "left eye reflection shows window at 2 o'clock, right eye shows at 10 o'clock - physically impossible"
- **Calibrate to reality**: Real photos have noise, compression artifacts, imperfect focus - these are GOOD signs
"""

    USER_PROMPT_TEMPLATE = """Analyze this image for signs of AI generation using the Top 5 Indicators framework.

**Your task**:
1. Systematically evaluate ALL 5 indicator categories
2. Identify specific artifacts with precise locations
3. Assess overall probability of AI generation
4. Provide detailed reasoning

**Return your analysis as a JSON object** following the exact schema provided in the system prompt.

**Remember**:
- Be thorough but concise
- Cite specific visual evidence
- Weight physics/lighting issues heavily
- Consider that real photos are imperfect; AI images often too perfect in wrong ways
"""

    def __init__(self):
        """
        Initialize the Visual Forensics Agent.

        Configuration is loaded from environment variables:
        - GOOGLE_API_KEY: API key for Google Gemini
        - LLM_MODEL: Model name (default: "gemini-2.5-pro")
        """
        # LLM configuration from environment
        self.model_name = os.getenv("LLM_MODEL", "gemini-2.5-pro")
        self.generation_config = {
            "temperature": 0.0,
            "max_output_tokens": 4096,
            "response_mime_type": "application/json",
        }

        self._init_google_client()

    def _init_google_client(self):
        """Initialize Google Gemini client"""
        try:
            import google.generativeai as genai
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=self.SYSTEM_PROMPT
            )
        except ImportError:
            raise ImportError("google-generativeai package not installed. Install with: pip install google-generativeai")

    def run(self, input_data: TaskInput) -> StepResult:
        """
        Execute visual forensics analysis on the given image.

        This method:
        1. Sends the image to a Vision-Language Model with specialized prompts
        2. Receives structured analysis based on 2025 detection indicators
        3. Parses and validates the response
        4. Returns StepResult with analysis in content field

        Args:
            input_data: TaskInput containing image_path and optional text

        Returns:
            StepResult with:
                - source: "VisualForensicsAgent"
                - content: Dictionary containing analysis results

        Raises:
            FileNotFoundError: If image_path doesn't exist
            ValueError: If LLM response cannot be parsed
            Exception: For other errors during analysis
        """
        print(f"  [VisualForensicsAgent] Analyzing image for AI generation: {input_data.image_path}...")

        try:
            # Send image and prompts to Vision-Language Model
            response_text = self._call_google(
                image_path=input_data.image_path,
                user_prompt=self.USER_PROMPT_TEMPLATE
            )

            # Parse JSON response
            analysis_result = self._parse_response(response_text)

            # Add metadata
            analysis_result['model_used'] = self.model_name
            analysis_result['image_path'] = input_data.image_path

            # Return as StepResult
            return StepResult(
                source="VisualForensicsAgent",
                content=analysis_result
            )

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Image not found: {input_data.image_path}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}") from e
        except Exception as e:
            raise Exception(f"Visual forensics analysis failed: {e}") from e

    def _call_google(self, image_path: str, user_prompt: str) -> str:
        """
        Call the Google Gemini Model to analyze an image.

        Args:
            image_path: Path to the image file
            user_prompt: User prompt/question about the image

        Returns:
            Model's text response
        """
        from PIL import Image
        
        # Open image using PIL
        img = Image.open(image_path)

        # Generate content
        response = self.model.generate_content(
            [user_prompt, img],
            generation_config=self.generation_config
        )
        
        return response.text

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse and clean the LLM response to extract JSON.

        Args:
            response_text: Raw text response from LLM

        Returns:
            Parsed dictionary matching VisualForensicsResult schema

        Raises:
            ValueError: If JSON cannot be extracted or is invalid
        """
        # Try to extract JSON from response (handle markdown code blocks)
        response_text = response_text.strip()

        # Remove markdown code fences if present
        if response_text.startswith('```'):
            lines = response_text.split('\n')
            # Remove first and last lines (``` markers)
            response_text = '\n'.join(lines[1:-1])
            # Remove 'json' language identifier if present
            if response_text.startswith('json'):
                response_text = response_text[4:].strip()

        # Parse JSON
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to find JSON object in text
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx+1]
                result = json.loads(json_str)
            else:
                raise ValueError("No valid JSON found in response")

        # Validate required fields
        required_fields = ['fake_probability', 'reasoning_summary', 'flagged_artifacts', 'confidence']
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")

        # Ensure flagged_artifacts is a list
        if not isinstance(result['flagged_artifacts'], list):
            result['flagged_artifacts'] = []

        return result
