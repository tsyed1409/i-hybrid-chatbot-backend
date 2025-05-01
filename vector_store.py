# vector_store.py

import faiss
import openai
import numpy as np
import os
import pickle
from typing import List

# Paths to save/load index and metadata
INDEX_PATH = "vector_index/faiss.index"
METADATA_PATH = "vector_index/chunks.pkl"
EMBEDDING_MODEL = "text-embedding-ada-002"

# Load index and associated chunks
if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        chunk_texts = pickle.load(f)
else:
    index = None
    chunk_texts = []

# Get embedding from OpenAI
def get_embedding(text: str) -> List[float]:
    response = openai.Embedding.create(input=[text], model=EMBEDDING_MODEL)
    return response["data"][0]["embedding"]

# Search top-k relevant chunks
def get_relevant_chunks(query: str, k: int = 3) -> List[str]:
    if not index or not chunk_texts:
        return []
    query_vector = np.array([get_embedding(query)], dtype="float32")
    distances, indices = index.search(query_vector, k)
    return [chunk_texts[i] for i in indices[0] if i < len(chunk_texts)]

# Optional function to build the index
def index_chunks(chunks: List[str]):
    global index, chunk_texts
    vectors = [get_embedding(chunk) for chunk in chunks]
    vectors_np = np.array(vectors, dtype="float32")
    index = faiss.IndexFlatL2(vectors_np.shape[1])
    index.add(vectors_np)
    chunk_texts = chunks
    os.makedirs("vector_index", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(chunk_texts, f)
