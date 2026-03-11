# rag/retriever.py

import faiss
import numpy as np
import pickle
from .embedder import get_embedding

def retrieve_chunks(question, top_k=5):
    index = faiss.read_index("faiss.index")

    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    question_embedding = get_embedding(question)
    question_embedding = np.array([question_embedding]).astype("float32")

    distances, indices = index.search(question_embedding, top_k)

    results = [chunks[i] for i in indices[0]]
    print("Retrieved chunks:", len(results))
    
    return results