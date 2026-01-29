"""Google Custom Search functionality for MorseQuery Simple."""

import os
from typing import Dict, List

import requests

GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY")
GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_CUSTOM_SEARCH_ENGINE_ID")


def google_custom_search(query: str, search_type: str = "text") -> List[Dict]:
    """Perform Google Custom Search."""
    url = "https://www.googleapis.com/customsearch/v1"

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
                pagemap = item.get("pagemap", {})
                image_url = None

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
