import os

import requests
from serpapi import GoogleSearch


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


if __name__ == "__main__":
    image_url = "real.webp"

    results = reverse_image_search(image_url)

    # Example usage with local file
    # results = reverse_image_search("real.webp")

    if "image_results" in results:
        image_results = results["image_results"]
        print(f"Found {len(image_results)} image results")
        print("\nResults:")
        for i, img in enumerate(image_results[:5], 1):  # Show first 5
            print(f"{i}. {img.get('title', 'No title')} - {img.get('link', 'No link')}")
    else:
        print("No image results found")
        print(f"Available keys: {results.keys()}")
