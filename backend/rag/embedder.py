# from openai import OpenAI
# import os
# # from sentence_transformers import SentenceTransformer

# # model = SentenceTransformer("all-min")

# # client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# # def get_embedding(text):
# #     response = client.embeddings.create(
# #         model="text-embedding-3-small",
# #         input=text
# #     )

# #     return response.data[0].embedding


from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str):
    embedding = model.encode(text)
    return embedding.tolist()