# rag/indexer.py

import os
import faiss
import numpy as np
from .chunker import chunk_text
from .embedder import get_embedding

def index_repository(repo_path, index_path="faiss.index"):
    texts = []
    
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith((".py", ".ts", ".js", ".md")):
                full_path = os.path.join(root, file)
                
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    
                chunks = chunk_text(content)
                texts.extend(chunks)

    embeddings = [get_embedding(text) for text in texts]
    
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))

    faiss.write_index(index, index_path)

    # Save chunks for later retrieval
    with open("chunks.pkl", "wb") as f:
        import pickle
        pickle.dump(texts, f)