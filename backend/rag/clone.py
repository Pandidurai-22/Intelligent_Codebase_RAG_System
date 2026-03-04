import os
import shutil
from git import Repo
import subprocess


BASE_DIR = "data"

# def clone_repo(repo_url:str):
#     repo_name =  repo_url.split("/")[-1].replace(".git","")
#     repo_path = os.path.join(BASE_DIR, repo_name)

#     if os.path.exist(repo_path):
#         shutil.rmtree(repo_path)

#     Repo.clone_from(repo_path, repo_url)

#     return repo_path


def clone_repo(repo_url: str):
    repo_name = repo_url.split("/")[-1]
    
    if repo_name.endswith(".git"):
        repo_name = repo_name.replace(".git", "")
    
    repo_path = os.path.join("repos", repo_name)

    if os.path.exists(repo_path):
        return repo_path  # already cloned

    subprocess.run(["git", "clone", repo_url, repo_path], check=True)

    return repo_path
