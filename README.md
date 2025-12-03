# Project--Chatbot
# 💬 NeoStats AI Chatbot — RAG + Groq LLM + Web Search

A powerful AI chatbot built using **Retrieval Augmented Generation (RAG)**, **Groq’s ultra-fast LLM**, and **Streamlit UI**.  
This chatbot allows users to upload documents, build a RAG index, and ask questions with accurate context-based responses.

---

## 🚀 Features

✅ Upload documents (PDF / TXT)  
✅ Automatic text extraction  
✅ RAG-based context retrieval  
✅ Embedding generation using SBERT  
✅ LLM-powered responses using Groq API  
✅ Optional SerpAPI web search fallback  
✅ Streamlit chat UI  
✅ Fast and beginner-friendly  

---

# 🧠 System Architecture (Explanation Section)

This project includes multiple components working together to provide accurate answers:

---

## **1️⃣ Document Upload**
Users upload PDFs or text files through Streamlit.  
The app extracts raw text from each page using:

- `pypdf` (for PDFs)
- `.read_text()` (for .txt files)

---

## **2️⃣ Text Chunking**
Long documents are broken into smaller pieces (“chunks”) using a sliding window approach:

- **Chunk size:** 300 words  
- **Overlap:** 50 words  

This ensures:
- Better embedding quality  
- More context recall  
- No loss of meaning between chunks  

---

## **3️⃣ Embedding Generation**
Chunks are converted into **vector embeddings** using **SentenceTransformers (SBERT)**.

Embeddings represent text in numeric form so the model can “understand” similarity.

Example:
"This is an example sentence." → [0.123, -0.221, 0.876, ...]

---

## **4️⃣ Vector Index Creation**
All embeddings are stored in memory:

{
"docs": [...],
"embeddings": [...]
}


This acts like a custom mini database for retrieval.

---

## **5️⃣ Query → Embedding → Similarity Search**
When a user asks a question:
- The query is embedded
- Cosine similarity is calculated against all document vectors
- Top 3 most relevant chunks are selected

Formula:
cosine_similarity = dot(a, b) / (||a|| * ||b||)


---

## **6️⃣ RAG Context Construction**
The retrieved document chunks are combined like this:

Source: document.pdf
Score: 0.91
Excerpt: ....


This context is prepended to the user query and sent to the LLM.

---

## **7️⃣ Groq LLM Response**
The chatbot uses **Groq’s Llama 3.1 models**, known for:

- Ultra-fast inference  
- Low cost  
- High accuracy  

The final message is generated using:

chat_model.invoke([SystemMessage, UserMessage, RAG Context])


---

# 🛠️ Installation

## 1️ Clone the repository

```bash
git clone https://github.com/your-username/neostat_chatbot.git
cd neostat_chatbot

2 Create a virtual environment

python -m venv venv
venv\Scripts\activate   # Windows
# or
source venv/bin/activate  # Mac/Linux

3  Install dependencies

pip install -r requirements.txt


API Key Setup
Groq API (Required)

Get API key from:
https://console.groq.com/keys

SerpAPI (Optional)

For web search fallback:
https://serpapi.com/manage-api-key


Configure API Keys
Add directly into config/config.py

GROQ_API_KEY = "your_key_here"
GROQ_MODEL = "llama-3.1-8b-instant"
SERPAPI_KEY = ""


Run the Application

streamlit run app.py

http://localhost:8501








