"""How to run the code
conda activate leet_aider
python ./Projects/LeetcodeCheatcode/006-query_agent.py
"""

import json
import os
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import requests
from datetime import datetime


# === Load Model and Index ===
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("./Projects/LeetcodeCheatcode/leetcode_index.faiss")

with open("./Projects/LeetcodeCheatcode/problem_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# === Parameters ===
OLLAMA_MODEL = "qwen2.5-coder:7b"  # or your choice from Ollama
TOP_K = 3


def query_similar_problems(problem_text):
    """
    Query the index to find similar problems based on the given problem text.

    Args:
        problem_text (str): The problem statement for which to find similar problems.

    Returns:
        list: A list of dictionaries containing metadata for similar problems.
    """
    query_embedding = embedding_model.encode([problem_text], convert_to_numpy=True)
    D, I = index.search(query_embedding, TOP_K)  # I = indices, D = distances
    return [metadata[i] for i in I[0]]


def build_prompt(new_problem_text, retrieved_contexts):
    """
    Build a prompt by combining the new problem text with similar problems.

    Args:
        new_problem_text (str): The new problem statement to solve.
        retrieved_contexts (list): A list of dictionaries containing metadata for similar problems.

    Returns:
        str: The constructed prompt for the LLM.
    """
    prompt = f"You are a Python coding assistant. Solve the following problem.\n\nProblem:\n{new_problem_text}\n\n"

    for i, ctx in enumerate(retrieved_contexts):
        print(i, ctx["url"])
        # prompt += f"\n---\nSimilar Problem {i+1}: {ctx['title']}\ntopics:\n{ctx['topicTags']}\nDescription:\n{ctx['Description']}\nSolution:\n{ctx['solution_code']}\n"
        prompt += f"\n---\nSimilar Problem {i+1}: {ctx['title']}\ntopics:\n{ctx['topicTags']}\nSolution:\n{ctx['solution_code']}\n"

    prompt += (
        "\n---\nNow provide a clean, working Python solution for the original problem."
    )

    return prompt


def query_ollama(prompt, model=OLLAMA_MODEL):
    """
    Query the Ollama language model to get a response based on the given prompt.

    Args:
        prompt (str): The prompt to send to the LLM.
        model (str, optional): The name of the LLM model to use. Defaults to "qwen2.5-coder:7b".

    Returns:
        str: The response from the LLM.
    """
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {"model": model, "prompt": prompt, "stream": False}
    if "deepseek" in model:
        # https://ollama.com/blog/thinking
        payload["think"] = False

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    json_data = response.json()
    try:
        del json_data["context"]
    except:
        pass

    current_datetime = datetime.now()
    timestamp_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    json_path = os.path.join("./Projects/LeetcodeCheatcode/ollama_response", f"{timestamp_str}.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json_data["prompt"] = prompt
        json.dump(json_data, jf, indent=2)

    return json_data["response"]


def solve_problem(problem_text, model=OLLAMA_MODEL):
    """
    Solve a problem by querying similar problems and generating a solution using the LLM.

    Args:
        problem_text (str): The new problem statement to solve.
        model (str, optional): The name of the LLM model to use. Defaults to "qwen2.5-coder:7b".

    Returns:
        None
    """
    print("\n🔍 Searching similar problems...")
    similar = query_similar_problems(problem_text)

    print("📄 Building prompt...")
    prompt = build_prompt(problem_text, similar)

    print("🤖 Querying LLM via Ollama...")
    result = query_ollama(prompt, model)
    print("\n🧠 LLM Output:\n")
    print(result)


# === Example Usage ===
if __name__ == "__main__":
    print("codellama:7b-instruct never listens to instructions!")
    print("deepseek-r1:8b never stops thinking! But, deepseek is best.")
    print(
        f"OLLAMA Model (default {OLLAMA_MODEL}, press ENTER) or codellama:7b-instruct | deepseek-r1:8b: ",
        end="",
    )
    model = OLLAMA_MODEL
    line = input()
    if line.strip() != "":
        model = line.strip()

    print("Paste your new LeetCode problem statement:")
    counter, problem_input = 0, ""
    while True:
        line = input()
        
        if line.strip() == "":
            if counter >= 1:
                break
            else:
                counter += 1
                continue
        else:
            counter = 0
            
        problem_input += line + "\n"

    solve_problem(problem_input, model)
