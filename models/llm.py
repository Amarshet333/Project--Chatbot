# models/llm.py

from langchain_groq import ChatGroq
from config.config import GROQ_API_KEY, GROQ_MODEL

def get_chatgroq_model():
    """
    Returns a ChatGroq model instance using API key and model name
    stored in config/config.py
    """
    try:
        if not GROQ_API_KEY:
            raise ValueError("❌ GROQ_API_KEY not found in config/config.py")

        model = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL
        )
        return model

    except Exception as e:
        raise RuntimeError(f"Error initializing Groq model: {str(e)}")