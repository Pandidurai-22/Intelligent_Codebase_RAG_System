from fastapi import FastAPI
from pydantic import BaseModel
from rag.clone import clone_repo
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
import os

from dotenv import load_dotenv

load_dotenv()

from rag.retriever import retrieve_chunks
from rag.generator import generate_answer
from pydantic import BaseModel



OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RepoRequest(BaseModel):
    repo_url: str


class ChatRequest(BaseModel):
    repo_name: str
    question: str

@app.get("/")
def root():
    return {"status":"RAG running in backend"}

@app.post("/index-repo")
def index_repo(request: RepoRequest):
    repo_path = clone_repo(request.repo_url)
    index_repository(repo_path)
    return{"message": "repo Cloned and indexed", "path": repo_path}

@app.get("/repo-structure")
def get_repo_structure(repo_name: str):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    repo_path = os.path.join(BASE_DIR, "repos", repo_name)

    print("Current working dir:", os.getcwd())
    print("Repo path:", repo_path)

    if not os.path.exists(repo_path):
        raise HTTPException(status_code=404, detail="Repo not found")

    return build_tree(repo_path)


def build_tree(path, depth=0, max_depth=2):
    if depth > max_depth:
        return []

    tree = []

    try:
        items = os.listdir(path)
    except Exception as e:
        print("Error reading directory:", e)
        return []

    for item in items:
        if item in [".git", "__pycache__", "node_modules", "venv"]:
            continue

        item_path = os.path.join(path, item)

        if os.path.isdir(item_path):
            tree.append({
                "name": item,
                "type": "folder",
                "children": build_tree(item_path, depth + 1, max_depth)
            })
        else:
            tree.append({
                "name": item,
                "type": "file"
            })

    return tree


@app.get("/file-content")
def get_file_content(repo_name: str, file_path: str):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(BASE_DIR, "repos", repo_name, file_path)

    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        return {"content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/chat")
def chat(request: ChatRequest):
    chunks = retrieve_chunks(request.question)
    answer = generate_answer(request.question, chunks)
    return {"answer": answer}