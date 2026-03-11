import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from .chunker import chunk_text

model = SentenceTransformer("all-MiniLM-L6-v2")

IGNORE_DIRS = {
    ".git",
    "node_modules",
    "venv",
    "__pycache__",
    "dist",
    "build"
}

MAX_CHUNKS_PER_FILE = 30


def index_repository(repo_path, index_path="faiss.index"):

    texts = []

    print("Reading repository files...")

    for root, dirs, files in os.walk(repo_path):

        # ignore heavy folders
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file in files:

            if file.endswith((".py", ".ts", ".js", ".md")):

                full_path = os.path.join(root, file)

                try:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                    chunks = chunk_text(content)

                    # limit chunks per file
                    chunks = chunks[:MAX_CHUNKS_PER_FILE]

                    texts.extend(chunks)

                except Exception:
                    print("Error reading file:", full_path)

    print("Total chunks:", len(texts))

    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True
    )

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    print("Saving FAISS index...")

    faiss.write_index(index, index_path)

    with open("chunks.pkl", "wb") as f:
        pickle.dump(texts, f)

    print("Indexing completed")











# import os
# import faiss
# import numpy as np
# import pickle
# from sentence_transformers import SentenceTransformer
# from .chunker import chunk_text

# model = SentenceTransformer("all-MiniLM-L6-v2")

# def index_repository(repo_path, index_path="faiss.index"):

#     texts = []

#     print("Reading repository files...")

#     for root, _, files in os.walk(repo_path):

#         for file in files:

#             if file.endswith((".py", ".ts", ".js", ".md")):

#                 full_path = os.path.join(root, file)

#                 try:
#                     with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
#                         content = f.read()

#                     chunks = chunk_text(content)

#                     texts.extend(chunks)

#                 except Exception as e:
#                     print("Error reading file:", full_path)

#     print("Total chunks:", len(texts))

#     print("Generating embeddings...")

#     embeddings = model.encode(texts)

#     embeddings = np.array(embeddings).astype("float32")

#     dimension = embeddings.shape[1]

#     index = faiss.IndexFlatL2(dimension)

#     index.add(embeddings)

#     print("Saving FAISS index...")

#     faiss.write_index(index, index_path)

#     with open("chunks.pkl", "wb") as f:
#         pickle.dump(texts, f)

#     print("Indexing completed")



# # # rag/indexer.py

# # import os
# # import faiss
# # import numpy as np
# # from .chunker import chunk_text
# # from .embedder import get_embedding

# # def index_repository(repo_path, index_path="faiss.index"):
# #     texts = []
    
# #     for root, _, files in os.walk(repo_path):
# #         for file in files:
# #             if file.endswith((".py", ".ts", ".js", ".md")):
# #                 full_path = os.path.join(root, file)
                
# #                 with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
# #                     content = f.read()
                    
# #                 chunks = chunk_text(content)
# #                 texts.extend(chunks)

# #     embeddings = [get_embedding(text) for text in texts]
    
# #     dimension = len(embeddings[0])
# #     index = faiss.IndexFlatL2(dimension)
# #     index.add(np.array(embeddings).astype("float32"))

# #     faiss.write_index(index, index_path)

# #     # Save chunks for later retrieval
# #     with open("chunks.pkl", "wb") as f:
# #         import pickle
# #         pickle.dump(texts, f)



