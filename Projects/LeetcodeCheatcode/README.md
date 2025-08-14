# Project: Leetcode Cheatcode

In this project, I plan to create an application "Leetcode Cheatcode". I want to use Python with my light machine. I do not want a heavy solution that will run locally. I envision it as an agent that can find a solution to a LeetCode problem. Until now, I thought I would vectorize all my current solutions and upload them to a vector database. When I write a solution for a new problem, I create a new token from the problem statement and perform a similarity search in the vector database. It will be the context. Then I will send the request to the LLM model running on Ollama. 

## Environment

    conda create --name leet_aider python=3.12

    # To activate this environment, use
    conda activate leet_aider

    python -m pip install pandas
    python -m pip install requests
    python -m pip install beautifulsoup4
    python -m pip install faiss-cpu sentence-transformers
    
    # To deactivate an active environment, use
    conda deactivate

## Ollama setup

    ollama pull qwen2.5-coder:7b
	ollama show qwen2.5-coder:7b

---

## 🛠️ Requirements Recap

- ✅ Ollama running locally (`ollama run codellama` or similar)
- ✅ FAISS index + metadata
- ✅ SentenceTransformer installed and consistent across all steps
- ✅ Python packages: `requests`, `faiss-cpu`, `sentence-transformers`

## How to Run

1. **Install Dependencies**:
    ```bash
    conda create --name leet_aider python=3.12
    conda activate leet_aider
    python -m pip install pandas requests beautifulsoup4 faiss-cpu sentence-transformers
    ```

2. **Fetch Problem Data**:
    ```bash
    python ./Projects/LeetcodeCheatcode/002-fetch.py
    ```

3. **Process Problem Data**:
    ```bash
    python ./Projects/LeetcodeCheatcode/003-process.py
    ```

4. **Fetch details about individual problems**:
    ```bash
    python ./Projects/LeetcodeCheatcode/004-details_fetch.py
    ```

5. **Build FAISS Index and Metadata**:
    ```bash
    python ./Projects/LeetcodeCheatcode/005-faiss_index_builder.py
    ```

6. **Query Agent**:
    ```bash
    python ./Projects/LeetcodeCheatcode/006-query_agent.py
    ```
