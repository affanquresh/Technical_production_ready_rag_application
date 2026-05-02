import os
import re
import pickle
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

# ---- CONFIG ----
DATA_PATH = "/Users/affanqureshi/Desktop/affan/Production_ready_RAG_application/synthetic_knowledge_items.csv"
CHROMA_PATH = "chroma_db_final"

# ---- Chunking Function ----
def chunk_text(text, doc_id, topic):
    chunks = []

    text = text.strip()

    section_patterns = [
        r"\*\*Prerequisites\*\*",
        r"\*\*Step \d+.*?\*\*",
        r"\*\*Troubleshooting.*?\*\*",
        r"\*\*Tips.*?\*\*"
    ]

    combined_pattern = "|".join(section_patterns)
    splits = re.split(f"({combined_pattern})", text)

    current_section = "general"
    step_number = None

    for part in splits:
        part = part.strip()
        if not part:
            continue

        if "Prerequisites" in part:
            current_section = "prerequisites"
            step_number = None
            continue

        elif "Step" in part:
            current_section = "step"
            match = re.search(r"Step (\d+)", part)
            step_number = int(match.group(1)) if match else None
            continue

        elif "Troubleshooting" in part:
            current_section = "troubleshooting"
            step_number = None
            continue

        elif "Tips" in part:
            current_section = "tips"
            step_number = None
            continue

        chunk = {
            "content": part,
            "metadata": {
                "doc_id": str(doc_id),
                "topic": topic,
                "section": current_section,
                "step_number": step_number
            }
        }

        chunks.append(chunk)

    # Fallback
    if len(chunks) == 0:
        paragraphs = text.split("\n\n")
        for para in paragraphs:
            para = para.strip()
            if para:
                chunks.append({
                    "content": para,
                    "metadata": {
                        "doc_id": str(doc_id),
                        "topic": topic,
                        "section": "general",
                        "step_number": None
                    }
                })

    return chunks


# ---- Step 1: Load CSV ----
if not os.path.exists(DATA_PATH):
    raise ValueError("❌ CSV file not found at data/data.csv")

df = pd.read_csv(DATA_PATH)

print("Columns found:", df.columns.tolist())

# ---- Step 2: Chunking ----
texts = []
metadatas = []

for idx, row in df.iterrows():
    topic = str(row.get("ki_topic", "")).strip()
    content = str(row.get("ki_text", "")).strip()

    if not content:
        continue

    chunks = chunk_text(content, doc_id=idx, topic=topic)

    for chunk in chunks:
        # Assign unique ID for hybrid retrieval mapping
        chunk["metadata"]["id"] = len(texts)

        texts.append(chunk["content"])
        metadatas.append(chunk["metadata"])

print(f"✅ Total chunks created: {len(texts)}")

# ---- Step 3: Embedding Model ----
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# ---- Step 4: Create Chroma DB ----
if not os.path.exists(CHROMA_PATH):
    os.makedirs(CHROMA_PATH)

vector_store = Chroma.from_texts(
    texts=texts,
    embedding=embedding_model,
    metadatas=metadatas,
    persist_directory=CHROMA_PATH
)

vector_store.persist()
print("✅ Chroma DB created")

# ---- Step 5: BM25 ----
tokenized = [text.split() for text in texts]
bm25 = BM25Okapi(tokenized)

# ---- Step 6: Save Files ----
with open("texts.pkl", "wb") as f:
    pickle.dump(texts, f)

with open("metadatas.pkl", "wb") as f:
    pickle.dump(metadatas, f)

with open("bm25.pkl", "wb") as f:
    pickle.dump(bm25, f)

print("✅ All files saved successfully")