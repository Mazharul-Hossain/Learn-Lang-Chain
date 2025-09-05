# Smart Research Doc Finder - AI-Powered Semantic Document Search

This project aims to help researchers and students extract key information from academic papers and
literature. It leverages natural language processing (NLP) techniques to identify relevant keywords and build a corpus of related articles.

## Features

- **Keyword Extraction**: Extracts top keywords from a given text using methods such as embedding, frequency, hybrid, LLM, and LLM-hybrid.
- **Semantic Search**: Queries the Semantic Scholar API to find related papers based on the extracted keywords.
- **LLM Integration**: Uses an Ollama language model to refine keyword extraction and provide context-aware summaries.

## Ollama setup

    ollama pull llama3.1:8b
    ollama show llama3.1:8b --modelfile > Modelfile

        FROM llama3.1:8b
        PARAMETER num_ctx 65536
        
    ollama create llama3.1:8b -f Modelfile
    ollama show llama3.1:8b

## Installation

1. **Install Dependencies**:
    ```bash
    conda create --name leet_aider python=3.12
    
    conda activate leet_aider
    python -m pip install pandas requests beautifulsoup4 faiss-cpu sentence-transformers nltk python-dotenv --upgrade
    ```
2. Set up environment variables in `.env` (e.g., `API_KEY` for Semantic Scholar)

## Usage

1. Run the main script to extract keywords and create a corpus:
    ```bash
    screen -L
    python main.py
    ```
2. Paste an abstract or literature when prompted.
3. The script will output the top keywords and create a JSON file with relevant papers.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.

--------------------------------------------

Create an intelligent document search tool where users can ask questions in plain English, and the system returns not just a list of documents, but actual answers pulled and reasoned from those documents — leveraging both FAISS DB and Langchain for power and flexibility. [Idea from [Top 8 LLM + RAG Projects for your AI Portfolio 2025](https://medium.com/ai-in-plain-english/top-8-llm-rag-projects-for-your-ai-portfolio-2025-c721a5e37b43)]

![SmartDoc Finder](image.png)

## Tools & Technologies

- FAISS — to store and retrieve embeddings of documents
- Langchain — to handle chaining of LLM prompts, memory, and logic
- OpenAI / LLaMA / Claude — as LLM backend (via Langchain)
- Streamlit or React — for a quick and elegant front-end

## Step-by-Step Design Process

1. Data Ingestion & Preprocessing
    - Upload PDFs, docs, or scraped text.
    - Chunk documents (e.g., 500–1000 tokens) for more accurate embedding.
    - Generate embeddings for each chunk using Langchain’s wrapper around an embedding model (OpenAI, Hugging Face, etc.).
    - Store all vector embeddings with references in FAISS DB.

2. Semantic Search
    - User inputs a natural language query (e.g., “What are the benefits of AI in logistics?”)
    - Langchain converts the query into an embedding vector.
    - FAISS searches for top N most semantically similar document chunks.

3. Intelligent Answering
    - Langchain passes retrieved chunks as context to the LLM.
    - The LLM then: Summarizes, Extracts answers, Or holds a conversation about the documents

4. UI & Interaction
    - Display top results with:
    - Highlighted source chunks
    - Direct answer
    - Option to “ask follow-up” or "read more".

## Real-World Applications

- Internal document search for large corporations
- Smart customer support (pulling from manuals, FAQs)
- Academic paper search engines
- Personal knowledge management systems (Second Brain)

## Bonus Upgrade Ideas

- Add document tagging and filtering (e.g., date, topic).
- Train with company-specific language or jargon.
- Implement a feedback loop to fine-tune search quality.
