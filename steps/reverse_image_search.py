import os

import requests
from google import genai
from serpapi import GoogleSearch

from core.schemas import StepResult, TaskInput
from steps.base import BaseStep


def upload_image_to_host(image_path):
    """
    Upload a local image file to a temporary hosting service and return the URL.

    Args:
        image_path: Path to the local image file

    Returns:
        Publicly accessible URL of the uploaded image
    """
    # Using Imgur anonymous upload API (no API key needed)
    # Convert image to PNG if needed (Imgur doesn't support webp)
    import base64
    import io

    from PIL import Image

    # Open and convert image to PNG if needed
    img = Image.open(image_path)
    # Convert RGBA to RGB if necessary (Imgur doesn't support RGBA)
    if img.mode == 'RGBA':
        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
        rgb_img.paste(img, mask=img.split()[3])  # Use alpha channel as mask
        img = rgb_img

    # Save to bytes buffer as PNG
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    image_data = buffer.getvalue()
    image_base64 = base64.b64encode(image_data).decode('utf-8')

    headers = {'Authorization': 'Client-ID 546c25a59c58ad7'}  # Imgur's public client ID
    data = {'image': image_base64}
    response = requests.post('https://api.imgur.com/3/image', headers=headers, data=data)

    if response.status_code == 200:
        data = response.json()
        if data.get('success'):
            return data.get('data', {}).get('link')
        raise Exception(f"Upload failed: {data}")
    else:
        raise Exception(f"Failed to upload image: {response.status_code} - {response.text}")


def reverse_image_search(image_input, api_key="d25810ce04a9e104958cd66b7461f3de474c9a8d7f3f863df08af9a9e6682cc8"):
    """
    Perform a Google reverse image search using SerpAPI.

    Args:
        image_input: URL of the image or path to a local image file
        api_key: SerpAPI key (defaults to provided key)

    Returns:
        Dictionary containing search results, including image_results
    """
    params = {
        "engine": "google_reverse_image",
        "api_key": api_key
    }

    # Check if input is a local file path or URL
    if os.path.isfile(image_input):
        # Local file - upload to hosting service first
        print(f"Uploading {image_input} to temporary hosting service...")
        image_url = upload_image_to_host(image_input)
        print(f"Image uploaded to: {image_url}")
        params["image_url"] = image_url
    else:
        # Assume it's a URL
        params["image_url"] = image_input

    search = GoogleSearch(params)
    results = search.get_dict()

    return results


def query_gemini_with_search_results(search_results, prompt_template=None, gemini_api_key=None, model_name="gemini-2.0-flash"):
    """
    Query Gemini API with reverse image search results.

    Args:
        search_results: Dictionary from reverse_image_search() containing image_results
        prompt_template: Custom prompt template. If None, uses a default template.
                         The template should include {search_results} placeholder.
        gemini_api_key: Gemini API key. If None, reads from GOOGLE_API_KEY environment variable.
        model_name: Name of the Gemini model to use (default: "gemini-2.0-flash")

    Returns:
        Response text from Gemini API
    """
    # Get API key from parameter or environment variable
    if gemini_api_key is None:
        gemini_api_key = os.getenv("GOOGLE_API_KEY")
        if gemini_api_key is None:
            raise ValueError("Gemini API key not provided. Set GOOGLE_API_KEY environment variable or pass gemini_api_key parameter.")

    # Create Gemini client
    client = genai.Client(api_key=gemini_api_key)

    # Extract image results from search results
    image_results = search_results.get("image_results", [])

    if not image_results:
        raise ValueError("No image_results found in search_results")

    # Format search results into a readable string
    formatted_results = []
    for i, result in enumerate(image_results, 1):
        title = result.get('title', 'No title')
        link = result.get('link', 'No link')
        source = result.get('source', 'Unknown source')
        formatted_results.append(f"{i}. Title: {title}\n   Source: {source}\n   Link: {link}")

    search_results_text = "\n\n".join(formatted_results)

    # Use default prompt template if none provided
    if prompt_template is None:
        prompt_template = """Based on the following reverse image search results, I want to know solely based on the results of the reverse image search whether the uploaded image is a deepfake or not:

{search_results}

1. visit the links and analyze the content of the pages
2. With the context of the pages, tell me whether the uploaded image is a deepfake or not

Analysis:"""

    # Format the prompt with search results
    prompt = prompt_template.format(search_results=search_results_text)

    # Generate response from Gemini
    response = client.models.generate_content(
        model=model_name,
        contents=[prompt]
    )

    return response.text


class ReverseImageSearch(BaseStep):
    def run(self, input_data: TaskInput) -> StepResult:
        # PUT YOUR LOGIC IN HERE
        print(f"  [Tool3] Performing reverse image search on {input_data.image_path}...")
        
        try:
            # Perform reverse image search
            search_results = reverse_image_search(input_data.image_path)
            
            if "image_results" in search_results:
                image_results = search_results["image_results"]
                print(f"  Found {len(image_results)} image results")
                
                # Query Gemini with the search results
                try:
                    gemini_response = query_gemini_with_search_results(search_results)
                    
                    return StepResult(
                        source="ReverseImageSearch",
                        content={
                            "num_results": len(image_results),
                            "analysis": gemini_response,
                            "top_results": [
                                {
                                    "title": img.get('title', 'No title'),
                                    "link": img.get('link', 'No link'),
                                    "source": img.get('source', 'Unknown source')
                                }
                                for img in image_results[:5]
                            ]
                        }
                    )
                except ValueError as e:
                    return StepResult(
                        source="ReverseImageSearch",
                        content={
                            "error": str(e),
                            "num_results": len(image_results)
                        }
                    )
                except Exception as e:
                    return StepResult(
                        source="ReverseImageSearch",
                        content={
                            "error": f"Error calling Gemini API: {str(e)}",
                            "num_results": len(image_results)
                        }
                    )
            else:
                return StepResult(
                    source="ReverseImageSearch",
                    content={
                        "error": "No image results found",
                        "available_keys": list(search_results.keys())
                    }
                )
        except Exception as e:
            return StepResult(
                source="ReverseImageSearch",
                content={
                    "error": f"Error performing reverse image search: {str(e)}"
                }
            )


if __name__ == "__main__":
    image_url = "real.webp"

    results = reverse_image_search(image_url)

    if "image_results" in results:
        image_results = results["image_results"]
        print(f"Found {len(image_results)} image results")
        print("\nResults:")
        for i, img in enumerate(image_results[:5], 1):  # Show first 5
            print(f"{i}. {img.get('title', 'No title')} - {img.get('link', 'No link')}")
        
        # Query Gemini with the search results
        print("\n" + "="*50)
        print("Querying Gemini API...")
        print("="*50 + "\n")
        try:
            gemini_response = query_gemini_with_search_results(results)
            print("Gemini Analysis:")
            print("-" * 50)
            print(gemini_response)
        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
    else:
        print("No image results found")
        print(f"Available keys: {results.keys()}")
