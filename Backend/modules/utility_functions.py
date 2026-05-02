from fastapi import FastAPI
from pydantic import BaseModel
import os

# ---- Your existing imports ----
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import numpy as np

import pickle

# Load texts
with open("texts.pkl", "rb") as f:
    texts = pickle.load(f)

# Load metadatas
with open("metadatas.pkl", "rb") as f:
    metadatas = pickle.load(f)

# Load BM25
with open("bm25.pkl", "rb") as f:
    bm25 = pickle.load(f)

def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


def hybrid_retrieve(query, top_k=5, alpha=0.5):
    formatted_query = "Represent this sentence for searching relevant passages: " + query
    
    bm25_scores = bm25.get_scores(query.split())
    
    docs = vector_store.similarity_search_with_score(formatted_query, k=len(texts))
    
    chroma_scores = np.zeros(len(texts))
    
    for doc, score in docs:
        idx = doc.metadata["id"]
        chroma_scores[idx] = score
    
    chroma_scores = 1 / (1 + chroma_scores)
    
    bm25_norm = normalize(bm25_scores)
    chroma_norm = normalize(chroma_scores)
    
    final_scores = alpha * chroma_norm + (1 - alpha) * bm25_norm
    
    top_indices = np.argsort(final_scores)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            "content": texts[idx],
            "metadata": metadatas[idx]
        })
    
    return results


def build_context(chunks):
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"[Chunk {i+1}]\n{chunk['content']}\n\n"
    return context


def generate_answer(query, chunks):
    context = build_context(chunks)
    
    prompt = f"""
You are an AI assistant.

Answer ONLY from the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""
    
    response = llm.invoke(prompt)
    return response.content
