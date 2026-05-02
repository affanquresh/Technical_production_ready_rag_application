from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from fastapi.middleware.cors import CORSMiddleware



from modules.loader import load_data
from modules.rag import run_rag

load_dotenv()
# ---- App ----
app = FastAPI(title="RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all (fine for dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Request Schema ----
class QueryRequest(BaseModel):
    query: str

# ---- Embedding ----
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# ---- Vector DB ----
vector_store = Chroma(
    persist_directory="new_chroma_db",
    embedding_function=embedding_model
)

# ---- Load Data ----
texts, metadatas, bm25 = load_data()

# ---- LLM ----
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY")
)

# ---- Endpoint ----
@app.post("/ask")
def ask_question(request: QueryRequest):
    
    answer = run_rag(
        request.query,
        bm25,
        vector_store,
        texts,
        metadatas,
        llm
    )
    
    return {
        "query": request.query,
        "answer": answer
    }


import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)