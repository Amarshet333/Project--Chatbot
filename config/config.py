# config/config.py
# Put your keys here OR set them as environment variables.
# Do NOT commit real keys to GitHub.

import os

# Groq (LLM) config
GROQ_API_KEY = "Add_GROQ_API_KEY"
GROQ_MODEL = "llama-3.1-8b-instant"

# SerpAPI (web search) config - optional but required for live web search
SERPAPI_KEY = "ADD_SERP_API_KEY"

# Embedding choice - 'sbert' uses sentence-transformers (no API key)
EMBEDDING_BACKEND = "sbert"  # "sbert" or "openai"

# Optional: OpenAI (if you later want to switch embedding backend)
OPENAI_API_KEY = "ADD_OPENAI_API_KEY"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
