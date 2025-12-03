# app.py (upgraded)
import streamlit as st
import os
import sys
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from models.llm import get_chatgroq_model
from utils.rag_utils import build_index, search_index, format_rag_context
from utils.web_search import serpapi_search
from config.config import EMBEDDING_BACKEND

# Keep a small cache folder for uploaded files
TMP_DIR = Path("tmp_docs")
TMP_DIR.mkdir(exist_ok=True)

def get_chat_response(chat_model, messages, system_prompt):
    """Get response from the chat model. If chat_model is None, return a helpful message."""
    try:
        if chat_model is None:
            return ("LLM provider not configured. Please set the GROQ_API_KEY in config/config.py "
                    "or environment variables. The app can still build an index for RAG locally.")
        # Prepare messages for the model
        formatted_messages = [SystemMessage(content=system_prompt)]
        for msg in messages:
            if msg["role"] == "user":
                formatted_messages.append(HumanMessage(content=msg["content"]))
            else:
                formatted_messages.append(AIMessage(content=msg["content"]))

        response = chat_model.invoke(formatted_messages)
        # Depending on the Groq wrapper, response may be a str or an object
        if hasattr(response, "content"):
            return response.content
        return str(response)
    except Exception as e:
        return f"Error getting response: {str(e)}"

def instructions_page():
    """Instructions and setup page"""
    st.title("The Chatbot Blueprint")
    st.markdown("Welcome! Follow these instructions to set up and use the chatbot.")
    st.markdown("""
    ## 🔧 Installation
    1. `pip install -r requirements.txt`
    2. Add your API keys in `config/config.py` or export environment variables:
       - `GROQ_API_KEY` (required for Groq LLM)
       - `SERPAPI_KEY` (optional, for live web search)
    3. Run `streamlit run app.py`
    """)

    st.markdown("### Embeddings\nWe use **SentenceTransformers** (all-MiniLM-L6-v2) by default (no API key needed).")

def chat_page():
    """Main chat interface page"""
    st.title("🤖 AI ChatBot — RAG-enabled")
    # mode + options
    st.sidebar.header("Options")
    upload_files = st.sidebar.file_uploader("Upload docs (PDF/TXT)", accept_multiple_files=True)
    chunk_size = st.sidebar.number_input("Chunk size (words)", min_value=100, max_value=2000, value=300, step=50)
    overlap = st.sidebar.number_input("Overlap (words)", min_value=0, max_value=500, value=50, step=10)
    enable_web_search = st.sidebar.checkbox("Enable live web search fallback", value=False)
    mode = st.sidebar.radio("Response mode", ["Concise", "Detailed"])
    temp = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

    # load model
    chat_model = get_chatgroq_model()

    # session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "index" not in st.session_state:
        st.session_state.index = None
    if "filepaths" not in st.session_state:
        st.session_state.filepaths = []

    # Save uploaded files to tmp
    if upload_files:
        saved = []
        for f in upload_files:
            p = TMP_DIR / f.name
            with open(p, "wb") as out:
                out.write(f.getbuffer())
            saved.append(str(p))
        st.session_state.filepaths = saved
        st.success(f"Saved {len(saved)} files to {TMP_DIR}")

    # Build index button
    if st.sidebar.button("Build RAG Index"):
        fps = st.session_state.get("filepaths", [])
        if not fps:
            st.sidebar.error("Upload at least one file first.")
        else:
            with st.spinner("Building index..."):
                idx = build_index(fps, chunk_size_words=int(chunk_size), overlap_words=int(overlap))
                st.session_state.index = idx
                st.sidebar.success(f"Index built with {len(idx.get('docs', []))} chunks.")

    # Clear index
    if st.sidebar.button("Clear Index"):
        st.session_state.index = None
        st.sidebar.success("Index cleared.")

    # Render chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    prompt = st.chat_input("Type your message here...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepare system prompt based on mode
        if mode == "Concise":
            system_prompt = "You are a helpful assistant. Provide short, precise answers (2-3 sentences)."
        else:
            system_prompt = "You are a helpful assistant. Provide detailed answers explaining reasoning and steps."

        # Try RAG retrieval
        context_str = ""
        index = st.session_state.get("index")
        results = []
        if index is not None and len(index.get("embeddings", [])) > 0:
            results = search_index(index, prompt, top_k=3)
            if results:
                context_str = format_rag_context(results)

        # If no RAG context and web search enabled, call web search
        web_snippets = None
        if (not context_str) and enable_web_search:
            web_snippets = serpapi_search(prompt, num_results=3)

        # Compose final_user_prompt
        pieces = []
        if context_str:
            pieces.append("Context from uploaded documents:\n" + context_str)
        if web_snippets:
            pieces.append("Live web search results:\n" + web_snippets)
        pieces.append("User question:\n" + prompt)
        final_user_prompt = "\n\n".join(pieces)

        # Call LLM
        with st.chat_message("assistant"):
            with st.spinner("Getting response..."):
                answer = get_chat_response(chat_model, st.session_state.messages + [{"role": "user", "content": final_user_prompt}], system_prompt)
                st.markdown(answer)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})

        # Show RAG sources and web snippets
        if results:
            st.markdown("### Sources (from uploaded docs)")
            for score, doc in results:
                st.write(f"**{Path(doc['source']).name}** — score: {score:.3f}")
                st.write(doc['text'][:800] + ("..." if len(doc['text']) > 800 else ""))
        if web_snippets:
            st.markdown("### Live Search Snippets")
            st.write(web_snippets)

    # Sidebar index status
    st.sidebar.markdown("---")
    st.sidebar.write("Index status:")
    idx = st.session_state.get("index")
    if idx:
        st.sidebar.write(f"Chunks: {len(idx.get('docs', []))}")
    else:
        st.sidebar.write("No index built.")

def main():
    st.set_page_config(
        page_title="LangChain Multi-Provider ChatBot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Go to:", ["Chat", "Instructions"], index=0)
        if page == "Chat":
            st.divider()
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

    if page == "Instructions":
        instructions_page()
    if page == "Chat":
        chat_page()

if __name__ == "__main__":
    main()
