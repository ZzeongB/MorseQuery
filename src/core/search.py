"""Google Custom Search functionality for MorseQuery."""

from typing import Dict, List, Union

import requests

from src.core.config import GOOGLE_SEARCH_API_KEY, GOOGLE_SEARCH_ENGINE_ID


def google_custom_search(
    query: str, search_type: str = "text"
) -> Union[List[Dict], Dict[str, List[Dict]]]:
    """Perform Google Custom Search.

    Args:
        query: Search query string
        search_type: Type of search - 'text', 'image', or 'both'

    Returns:
        List of search results or dict with 'text' and 'image' keys for 'both' type
    """
    url = "https://www.googleapis.com/customsearch/v1"

    # Handle 'both' search type by making two separate API calls
    if search_type == "both":
        text_results = google_custom_search(query, "text")
        image_results = google_custom_search(query, "image")
        return {"text": text_results, "image": image_results}

    params = {
        "key": GOOGLE_SEARCH_API_KEY,
        "cx": GOOGLE_SEARCH_ENGINE_ID,
        "q": query,
        "num": 5,
    }

    if search_type == "image":
        params["searchType"] = "image"

    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()
    results = []

    if "items" in data:
        for item in data["items"]:
            if search_type == "image":
                results.append(
                    {
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "thumbnail": item.get("image", {}).get("thumbnailLink", ""),
                        "context": item.get("snippet", ""),
                    }
                )
            else:
                # Extract image from pagemap if available
                pagemap = item.get("pagemap", {})
                image_url = None

                # Try cse_image first, then cse_thumbnail, then metatags og:image
                if "cse_image" in pagemap and len(pagemap["cse_image"]) > 0:
                    image_url = pagemap["cse_image"][0].get("src", "")
                elif "cse_thumbnail" in pagemap and len(pagemap["cse_thumbnail"]) > 0:
                    image_url = pagemap["cse_thumbnail"][0].get("src", "")
                elif "metatags" in pagemap and len(pagemap["metatags"]) > 0:
                    image_url = pagemap["metatags"][0].get("og:image", "")

                results.append(
                    {
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                        "image": image_url,
                    }
                )

    return results


def count_results(
    search_results: Union[List[Dict], Dict[str, List[Dict]]], search_type: str
) -> int:
    """Count the number of search results.

    Args:
        search_results: Results from google_custom_search()
        search_type: Type of search - 'text', 'image', or 'both'

    Returns:
        Total number of results
    """
    if search_type == "both" and isinstance(search_results, dict):
        return len(search_results.get("text", [])) + len(
            search_results.get("image", [])
        )
    elif isinstance(search_results, list):
        return len(search_results)
    return 0
