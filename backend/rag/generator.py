import requests

def generate_answer(question, chunks):
    chunks = chunks[:3]

    context = "\n".join(chunks)

    prompt = f"""

you are a code assistant

Answert the question using only the provided context.


Context:
{context}

Question:
{question}

"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "phi3:mini",
            "prompt": prompt,
            "stream": False,
            "options":{
                "num_predict": 120
            }
        }
    )

    return response.json()["response"]




# import requests

# def generate_answer(question, chunks):
    
#     context = "\n".join(chunks)

#     prompt = f"""

# use the following context to answer the question.

# Context:
# {context}

# Question:
# {question}
# """

#     response = requests.post(
#         "http://localhost:11434/api/generate",
#         json = {
#             "model": "phi3",
#             "prompt": prompt,
#             "stream": False

#         }
#     )

#     return response.json()["response"]