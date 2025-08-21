"""
python -m pip install faiss-cpu sentence-transformers
"""

import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle


JSON_FOLDER = "./Projects/LeetcodeCheatcode/JSON"


def load_all_problems(json_folder: str) -> list:
    """
    Load all problems from JSON files in the specified folder.

    Args:
        json_folder (str): The path to the folder containing JSON files with problem descriptions.

    Returns:
        list: A list of dictionaries, each representing a problem.
    """
    problems = []
    for fname in os.listdir(json_folder):
        if fname.endswith(".json"):
            json_path = os.path.join(json_folder, fname)
            # print(f"The path '{json_path}' exists.")
            with open(json_path, "r", encoding="utf-8") as f:
                problem = json.load(f)
                if "Description" not in problem:
                    print(f"Error reading '{json_path}'.")
                    continue

                problems.append(problem)
    return problems


def main():
    """
    Main function to build the FAISS index for LeetCode problems.
    """
    problems = load_all_problems(JSON_FOLDER)
    corpus_texts = [p["Description"] for p in problems]

    # Choose a lightweight embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")  # ~80MB, good tradeoff

    # Embed the problem descriptions
    corpus_embeddings = model.encode(
        corpus_texts, convert_to_numpy=True, show_progress_bar=True
    )

    embedding_dim = corpus_embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)  # L2 = Euclidean distance

    # Add all embeddings to the index
    index.add(corpus_embeddings)
    print(f"Indexed {index.ntotal} problems.")

    # Save a mapping from FAISS index to problem metadata
    with open("./Projects/LeetcodeCheatcode/problem_metadata.pkl", "wb") as f:
        pickle.dump(problems, f)

    faiss.write_index(index, "./Projects/LeetcodeCheatcode/leetcode_index.faiss")


if __name__ == "__main__":
    main()
