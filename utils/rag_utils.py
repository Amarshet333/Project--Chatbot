# utils/rag_utils.py
# Document loading, chunking, index build, and search.
from typing import List, Dict, Any, Tuple
from pathlib import Path
import math
import heapq

from models.embeddings import embed_texts
import numpy as np

# Simple whitespace chunker (words)
def chunk_text(text: str, chunk_size_words: int = 300, overlap_words: int = 50) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i: i + chunk_size_words]
        chunks.append(" ".join(chunk))
        i += chunk_size_words - overlap_words
    return chunks

def load_text_from_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    # TXT files
    if p.suffix.lower() == ".txt":
        return p.read_text(encoding="utf-8", errors="ignore")

    # PDF files (normal + scanned)
    elif p.suffix.lower() == ".pdf":
        text = ""

        # --- First try PyPDF (fast extractor) ---
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(p))
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        except:
            pass

        # --- If PyPDF failed to extract anything, try pdfplumber ---
        if len(text.strip()) < 10:
            try:
                import pdfplumber
                with pdfplumber.open(str(p)) as pdf:
                    for page in pdf.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text += extracted + "\n"
            except:
                pass

        # --- If STILL empty, fallback to OCR ---
        if len(text.strip()) < 10:
            try:
                import pytesseract
                from pdf2image import convert_from_path
                images = convert_from_path(str(p))
                for img in images:
                    text += pytesseract.image_to_string(img) + "\n"
            except:
                pass

        return text

    # OTHER file types
    else:
        return p.read_text(encoding="utf-8", errors="ignore")


def build_index(filepaths: List[str], chunk_size_words: int = 300, overlap_words: int = 50) -> Dict[str, Any]:
    """
    Builds an in-memory index:
    {
      "docs": [{"source": <path>, "text": <chunk>}...],
      "embeddings": [..],
    }
    """
    docs = []
    for fp in filepaths:
        try:
            txt = load_text_from_file(fp)
            chunks = chunk_text(txt, chunk_size_words, overlap_words)
            for c in chunks:
                docs.append({"source": fp, "text": c})
        except Exception as e:
            # skip files that cannot be read
            print(f"Warning: could not load {fp}: {e}")

    texts = [d["text"] for d in docs]
    if not texts:
        return {"docs": [], "embeddings": []}

    embeddings = embed_texts(texts)  # list[list[float]]
    return {"docs": docs, "embeddings": embeddings}

# cosine similarity
def _cosine_sim(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def search_index(index: Dict[str, Any], query: str, top_k: int = 3) -> List[Tuple[float, Dict[str, Any]]]:
    # Fix: check embeddings safely
    emb = index.get("embeddings", None)
    if index is None or emb is None or len(emb) == 0:
        return []

    q_emb = embed_texts([query])[0]

    scores = []
    for emb_vec, doc in zip(index["embeddings"], index["docs"]):
        s = _cosine_sim(q_emb, emb_vec)
        scores.append((s, doc))

    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:top_k]


def format_rag_context(results: List[Tuple[float, Dict[str, Any]]]) -> str:
    parts = []
    for score, doc in results:
        parts.append(f"Source: {doc['source']}\nScore: {score:.4f}\nExcerpt: {doc['text']}\n---")
    return "\n".join(parts)
