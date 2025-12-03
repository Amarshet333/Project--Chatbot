# utils/web_search.py
# SerpAPI wrapper (returns concatenated snippets). If SERPAPI_KEY missing, returns None.
import requests
from config.config import SERPAPI_KEY

def serpapi_search(query: str, num_results: int = 3, timeout: int = 8):
    """
    Returns a short text with top snippets + links, or None if SERPAPI_KEY not set / error.
    """
    if not SERPAPI_KEY:
        return None
    try:
        params = {
            "q": query,
            "engine": "google",
            "api_key": SERPAPI_KEY,
            "num": num_results,
        }
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=timeout)
        if r.status_code != 200:
            return None
        data = r.json()
        snippets = []
        for item in data.get("organic_results", [])[:num_results]:
            snippet = item.get("snippet") or item.get("title", "")
            link = item.get("link", "")
            snippets.append(f"{snippet}\n{link}")
        return "\n\n".join(snippets) if snippets else None
    except Exception as e:
        print(f"Web search error: {e}")
        return None
