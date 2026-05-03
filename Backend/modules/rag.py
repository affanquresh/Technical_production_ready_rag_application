# import numpy as np
# from sentence_transformers import CrossEncoder

# chat_history = []

# # ---- format history ----
# def format_history(chat_history):
#     history_text = ""
#     for msg in chat_history:
#         history_text += f"{msg['role']}: {msg['content']}\n"
#     return history_text

# # ---- Load reranker once ----
# reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# # ---- Normalize ----
# def normalize(arr):
#     return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


# # ---- Hybrid Retrieval ----
# def hybrid_retrieve(query, bm25, vector_store, texts, metadatas, top_k=10, alpha=0.5):
    
#     formatted_query = "Represent this sentence for searching relevant passages: " + query
    
#     bm25_scores = bm25.get_scores(query.split())
    
#     docs = vector_store.similarity_search_with_score(formatted_query, k=len(texts))
    
#     chroma_scores = np.zeros(len(texts))
    
#     for doc, score in docs:
#         idx = doc.metadata["id"]
#         chroma_scores[idx] = score
    
#     # Convert distance → similarity
#     chroma_scores = 1 / (1 + chroma_scores)
    
#     bm25_norm = normalize(bm25_scores)
#     chroma_norm = normalize(chroma_scores)
    
#     final_scores = alpha * chroma_norm + (1 - alpha) * bm25_norm
    
#     top_indices = np.argsort(final_scores)[-top_k:][::-1]
    
#     results = []
#     for idx in top_indices:
#         results.append({
#             "content": texts[idx],
#             "metadata": metadatas[idx]
#         })
    
#     return results


# # ---- Reranking ----
# def rerank_chunks(query, chunks, top_k=3):
    
#     pairs = [(query, chunk["content"]) for chunk in chunks]
    
#     scores = reranker_model.predict(pairs)
    
#     for i, chunk in enumerate(chunks):
#         chunk["rerank_score"] = scores[i]
    
#     chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    
#     return chunks[:top_k]


# # ---- Build Context ----
# def build_context(chunks):
#     context = ""
#     for i, chunk in enumerate(chunks):
#         context += f"[Chunk {i+1}]\n{chunk['content']}\n\n"
#     return context


# # ---- Generate Answer ----
# def generate_answer(query, chunks, llm):
#     global chat_history

#     context = build_context(chunks)
#     history_text = format_history(chat_history[-7:])  # last 2 turns

#     prompt = f"""
# You are an AI assistant.

# Use the conversation history and context to answer the question.

# Conversation History:
# {history_text}

# Context:
# {context}

# Question:
# {query}

# Rules:
# - Answer ONLY from the context
# - Use history for understanding the question
# - If answer is not in context, say "I don't know"

# Answer:
# """
#     print(history_text)
#     response = llm.invoke(prompt)
#     return response.content

# # ---- Full Pipeline ----
# def run_rag(query, bm25, vector_store, texts, metadatas, llm):
#     global chat_history

#     chunks = hybrid_retrieve(query, bm25, vector_store, texts, metadatas)
#     reranked_chunks = rerank_chunks(query, chunks)

#     answer = generate_answer(query, reranked_chunks, llm)

#     # ✅ Update memory
#     chat_history.append({"role": "user", "content": query})
#     chat_history.append({"role": "assistant", "content": answer})

#     # ✅ Limit memory (very important)
#     chat_history = chat_history[-6:]

#     return answer



import numpy as np

chat_history = []

# ---- format history ----
def format_history(chat_history):
    history_text = ""
    for msg in chat_history:
        history_text += f"{msg['role']}: {msg['content']}\n"
    return history_text

# ---- Normalize ----
def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

# ---- Hybrid Retrieval ----
def hybrid_retrieve(query, bm25, vector_store, texts, metadatas, top_k=5, alpha=0.5):
    
    bm25_scores = bm25.get_scores(query.split())
    
    docs = vector_store.similarity_search_with_score(query, k=len(texts))
    
    chroma_scores = np.zeros(len(texts))
    
    for doc, score in docs:
        idx = doc.metadata["id"]
        chroma_scores[idx] = score
    
    # Convert distance → similarity
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

# ---- Build Context ----
def build_context(chunks):
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"[Chunk {i+1}]\n{chunk['content']}\n\n"
    return context

# ---- Generate Answer ----
def generate_answer(query, chunks, llm):
    global chat_history

    context = build_context(chunks)
    history_text = format_history(chat_history[-6:])

    prompt = f"""
You are an AI assistant.

Use the conversation history and context to answer the question.

Conversation History:
{history_text}

Context:
{context}

Question:
{query}

Rules:
- Answer ONLY from the context
- If not found, say "I don't know"

Answer:
"""

    response = llm.invoke(prompt)
    return response.content

# ---- Full Pipeline ----
def run_rag(query, bm25, vector_store, texts, metadatas, llm):
    global chat_history

    chunks = hybrid_retrieve(query, bm25, vector_store, texts, metadatas)

    answer = generate_answer(query, chunks, llm)

    # ---- Update memory ----
    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": answer})

    chat_history = chat_history[-6:]

    return answer