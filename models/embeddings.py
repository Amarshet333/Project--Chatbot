# models/embeddings.py

from sentence_transformers import SentenceTransformer
from config.config import EMBEDDING_BACKEND, OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL
import numpy as np
import requests

# Load SBERT model once (fast)
_sbert_model = None

def get_sbert_model():
    global _sbert_model
    if _sbert_model is None:
        _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sbert_model


def embed_texts(texts):
    """
    Returns embeddings for a list of texts.
    - Uses SBERT by default (no API key required)
    - Can switch to OpenAI if EMBEDDING_BACKEND='openai'
    """
    backend = EMBEDDING_BACKEND.lower()

    # SBERT (default, no API key)
    if backend == "sbert":
        model = get_sbert_model()
        embeddings = model.encode(texts, show_progress_bar=False)
        return np.array(embeddings)

    # OpenAI embeddings (optional)
    elif backend == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI embedding backend requested but OPENAI_API_KEY is missing.")

        url = "https://api.openai.com/v1/embeddings"

        # Request
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "input": texts,
                "model": OPENAI_EMBEDDING_MODEL
            }
        )

        data = response.json()
        vectors = [item["embedding"] for item in data["data"]]
        return np.array(vectors)

    else:
        raise ValueError(f"Unknown EMBEDDING_BACKEND: {backend}")
