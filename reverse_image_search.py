from serpapi import GoogleSearch


def reverse_image_search(image_url, api_key="d25810ce04a9e104958cd66b7461f3de474c9a8d7f3f863df08af9a9e6682cc8"):
    """
    Perform a Google reverse image search using SerpAPI.

    Args:
        image_url: URL of the image to search
        api_key: SerpAPI key (defaults to provided key)

    Returns:
        Dictionary containing search results, including image_results
    """
    params = {
        "engine": "google_reverse_image",
        "image_url": image_url,
        "api_key": api_key
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    return results


if __name__ == "__main__":
    # Example usage
    image_url = "https://i.imgur.com/5bGzZi7.jpg"
    results = reverse_image_search(image_url)

    if "image_results" in results:
        image_results = results["image_results"]
        print(f"Found {len(image_results)} image results")
        print("\nResults:")
        for i, img in enumerate(image_results[:5], 1):  # Show first 5
            print(f"{i}. {img.get('title', 'No title')} - {img.get('link', 'No link')}")
    else:
        print("No image results found")
        print(f"Available keys: {results.keys()}")
