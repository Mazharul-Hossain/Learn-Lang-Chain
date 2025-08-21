# AlgoWhisper: A Local RAG Assistant for LeetCode Problems

- Type: Personal Project / AI Agent
- Focus Areas: RAG, FAISS, LLM orchestration, Python

AlgoWhisper is a lightweight Retrieval-Augmented Generation (RAG) application
designed to assist with solving algorithmic coding problems. The system indexes
past LeetCode solutions, retrieves relevant examples using semantic similarity,
and leverages a locally hosted Large Language Model (LLM) through Ollama to
generate new Python solutions in real time.

## Motivation

LeetCode and algorithmic practice often involve recurring problem patterns. I
wanted to build a personal coding assistant that:

- Runs entirely on my local machine (no heavy cloud dependencies)
- Leverages my own solution history as a knowledge base
- Adapts past solutions to accelerate solving new problems

## Architecture

1. Solution Corpus

    - Extracted my .py solutions into structured JSON with problem statement, description, and code.

2. Vector Database

    - Retrieve problem statements from leetcode using GraphQL API request
    - Embedded problem statements using sentence-transformers (all-MiniLM-L6-v2)
    - Indexed embeddings with FAISS for efficient similarity search.

3. Retriever

    - For a new problem, embed the statement → search FAISS → retrieve top-k similar problems.

4. Generator (LLM)

    - Contextualize retrieved problems + solutions into a structured prompt.
    - Send the prompt to a locally running Ollama model (e.g., codellama:7b-instruct) for code generation.

5. Result

    - Return a clean, working Python solution with inspiration from past solutions.
    - No need to write the boiler-plate code.
    - Update and refine as per updated requirements.

## Key Features

- Lightweight & Local: No dependency on external APIs or cloud infrastructure
- Personalized Knowledge Base: Learns from my own problem-solving history
- Retrieval-Augmented Generation: Combines semantic search with generative reasoning
- Modular Design: Easily extensible with new embedding models or LLM

## Outcomes

- Built a functional RAG coding assistant capable of retrieving context and generating Python solutions.
- Learned hands-on about vector search, embeddings, and prompt engineering.
- Demonstrated how advanced AI techniques can be applied in a lightweight, practical setting.

## Tech Stack

- Language: Python
- Vector Database: FAISS
- Embeddings: SentenceTransformers (all-MiniLM-L6-v2)
- LLM Runtime: Ollama (qwen2.5-coder)
- Data Format: JSON for problem metadata and solutions

## Ollama setup

    ollama pull qwen2.5-coder:7b
	ollama show qwen2.5-coder:7b

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
