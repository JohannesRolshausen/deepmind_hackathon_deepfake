"""
Visual Forensics Agent - Advanced AI-Generated Image Detection
Author: Mark
Description: Uses Vision-Language Models to detect AI-generated images based on
             2025 state-of-the-art detection indicators targeting Diffusion Model weaknesses.
"""

import json
from typing import Dict, Any, List
from steps.base import Step
from core.llm import get_default_client
from core.schemas import VisualForensicsResult, ArtifactDetail


class VisualForensicsAgent(Step):
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

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Visual Forensics Agent.

        Args:
            config: Optional configuration dictionary
                - llm_client: Custom LLM client (if None, uses default)
                - temperature: Sampling temperature (default: 0.0 for deterministic)
                - max_tokens: Maximum response tokens (default: 4096)
        """
        super().__init__(config)

        # Initialize LLM client
        self.llm_client = self.config.get('llm_client') or get_default_client()
        self.temperature = self.config.get('temperature', 0.0)
        self.max_tokens = self.config.get('max_tokens', 4096)

    def run(self, image_path: str) -> Dict[str, Any]:
        """
        Execute visual forensics analysis on the given image.

        This method:
        1. Sends the image to a Vision-Language Model with specialized prompts
        2. Receives structured analysis based on 2025 detection indicators
        3. Parses and validates the response
        4. Returns structured results

        Args:
            image_path: Path to the image file to analyze

        Returns:
            Dictionary containing:
                - fake_probability: float (0.0-1.0)
                - reasoning_summary: str
                - flagged_artifacts: List[Dict] with indicator_type, location, description, severity
                - confidence: float (0.0-1.0)
                - model_used: str (name of LLM model used)

        Raises:
            FileNotFoundError: If image_path doesn't exist
            ValueError: If LLM response cannot be parsed
            Exception: For other errors during analysis
        """
        try:
            # Send image and prompts to Vision-Language Model
            response_text = self.llm_client.analyze_image(
                image_path=image_path,
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=self.USER_PROMPT_TEMPLATE,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            # Parse JSON response
            analysis_result = self._parse_response(response_text)

            # Add metadata
            analysis_result['model_used'] = self.llm_client.model

            # Validate using Pydantic schema
            validated_result = VisualForensicsResult(**analysis_result)

            return validated_result.model_dump()

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Image not found: {image_path}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}") from e
        except Exception as e:
            raise Exception(f"Visual forensics analysis failed: {e}") from e

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

    def batch_analyze(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze multiple images in batch.

        Args:
            image_paths: List of paths to image files

        Returns:
            List of analysis results, one per image
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.run(image_path)
                result['image_path'] = image_path
                result['status'] = 'success'
            except Exception as e:
                result = {
                    'image_path': image_path,
                    'status': 'error',
                    'error_message': str(e)
                }
            results.append(result)

        return results


# Convenience function for standalone usage
def analyze_image_for_deepfake(image_path: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Standalone function to analyze an image for AI generation.

    Args:
        image_path: Path to the image file
        config: Optional configuration for the agent

    Returns:
        Analysis results dictionary

    Example:
        >>> result = analyze_image_for_deepfake("suspicious_photo.jpg")
        >>> print(f"Fake probability: {result['fake_probability']:.2%}")
        >>> print(f"Reasoning: {result['reasoning_summary']}")
    """
    agent = VisualForensicsAgent(config=config)
    return agent.run(image_path)


if __name__ == "__main__":
    # Example usage for testing
    import sys

    if len(sys.argv) < 2:
        print("Usage: python visual_forensics.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    print(f"Analyzing: {image_path}")
    print("-" * 60)

    try:
        result = analyze_image_for_deepfake(image_path)

        print(f"\n{'='*60}")
        print(f"ANALYSIS RESULTS")
        print(f"{'='*60}")
        print(f"\nFake Probability: {result['fake_probability']:.1%}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Model Used: {result['model_used']}")

        print(f"\n{'─'*60}")
        print(f"REASONING")
        print(f"{'─'*60}")
        print(f"{result['reasoning_summary']}")

        if result['flagged_artifacts']:
            print(f"\n{'─'*60}")
            print(f"FLAGGED ARTIFACTS ({len(result['flagged_artifacts'])} found)")
            print(f"{'─'*60}")

            for i, artifact in enumerate(result['flagged_artifacts'], 1):
                print(f"\n{i}. [{artifact['indicator_type']}]")
                print(f"   Location: {artifact['location']}")
                print(f"   Description: {artifact['description']}")
                print(f"   Severity: {artifact['severity']:.1%}")

        print(f"\n{'='*60}\n")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
