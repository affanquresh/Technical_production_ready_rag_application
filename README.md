# 🚀 Hybrid RAG Assistant

## 📌 Overview
This project is a **production-ready Retrieval-Augmented Generation (RAG) system** that improves LLM accuracy using hybrid retrieval, reranking, and conversational memory.

It combines keyword-based and semantic search to deliver more relevant and grounded responses.

---

## 🎯 Problem
Standard LLMs:
- Hallucinate information  
- Lack domain-specific knowledge  
- Cannot access private data  

Basic RAG helps, but still suffers from:
- Weak keyword matching OR semantic gaps  
- Noisy retrieval results  

---

## 💡 Solution
This system implements a **Hybrid RAG pipeline**:

- **BM25 Retrieval** → keyword matching  
- **Embedding Retrieval (Chroma)** → semantic search  
- **Score Fusion** → combines both methods  
- **Reranking** → improves top results  
- **LLM Generation** → context-based answers  
- **Memory** → supports conversational queries  

---

## 🧠 Architecture
User Query
↓
Hybrid Retrieval (BM25 + Embeddings)
↓
Score Fusion
↓
Top-K Results
↓
Reranking
↓
LLM (Context + Memory)
↓
Final Answer


---

## ⚙️ Tech Stack

- **Backend:** FastAPI  
- **Vector DB:** ChromaDB  
- **Embeddings:** BAAI/bge-small-en-v1.5  
- **Retrieval:** BM25 + Dense Embeddings  
- **Reranking:** Cross-Encoder  
- **LLM:** Groq (gpt-oss-120b)  
- **Frontend:** HTML, CSS, JavaScript  

---

## 🔥 Key Features

- Hybrid retrieval (keyword + semantic)
- Reranking for improved relevance
- Conversational memory (multi-turn support)
- FastAPI backend + interactive UI
- Evaluation pipeline for performance analysis

---

## 📊 Results

| Metric          | Score |
|----------------|------|
| Without Rerank | ~0.71 |
| With Rerank    | ~0.70 |
| Groundedness   | ~0.93 |

---

## 🧪 Example
User: How to reset DHCP on Windows?
User: What about mac?


→ System handles follow-up using memory

---

## 📁 Project Structure
backend/
├── app.py
├── rag.py
├── chroma_db/
├── texts.pkl
├── metadatas.pkl
├── bm25.pkl

frontend/
├── index.html
├── styles.css
├── script.js


---

## 🚀 How to Run

### Backend
```bash
cd backend
uvicorn app:app --reload

Frontend

Open:

frontend/index.html
🔐 Environment Variables

Create .env file:

GROQ_API_KEY=your_api_key
🏁 Conclusion

This project demonstrates a strong RAG system design with hybrid retrieval, reranking, and conversational capabilities.

👤 Author

Affan Qureshi