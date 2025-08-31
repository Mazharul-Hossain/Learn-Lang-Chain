import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
import requests
import json
import os
from sentence_transformers import SentenceTransformer
import requests
from datetime import datetime
from NltkDownload import NltkDownload

class ExtractTopKeywords:
    
    def __init__(self):
        _ = NltkDownload()

        # Load embedding model (you can swap for a local LLM embedding generator)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")  # small & fast

    def extract_candidate_phrases(self, text, max_ngram=3):
        """
        Split text into candidate keyphrases (1 to max_ngram words),
        filtering out stopwords.
        """
        words = nltk.word_tokenize(text.lower())
        sw = set(stopwords.words("english"))
        words = [w for w in words if w.isalnum() and w not in sw]

        # Generate n-grams (single words + phrases)
        candidates = []
        for n in range(1, max_ngram+1):
            for i in range(len(words)-n+1):
                cand = " ".join(words[i:i+n])
                candidates.append(cand)
        
        # return unique phrases
        return set(candidates)
    
    def get_embedding_rank_keywords(self, text, candidates, top_k=5):        
        # Encode passage and candidates
        text_emb = self.model.encode(text, convert_to_tensor=True)
        cand_embs = self.model.encode(candidates, convert_to_tensor=True)

        # Compute similarity
        scores = util.cos_sim(text_emb, cand_embs)[0].cpu().tolist()

        # Rank by similarity
        scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [kw for kw, _ in scored[:top_k]]

    def build_prompt(self, text, top_k=5, retrieved_contexts=[]):
        """
        Build a prompt by combining the new problem text with similar problems.

        Args:
            new_problem_text (str): The new problem statement to solve.
            retrieved_contexts (list): A list of dictionaries containing metadata for similar problems.

        Returns:
            str: The constructed prompt for the LLM.
        """
        prompt = "You are a technical literature summarization assistant. You are great at finding most relevent keywords from technical articles, when asked you provide the top keywords in a comma separated list. A keyword should be a clean and concise phrase consisting of one or two words."
        prompt += f"Extract keywords from this passage: {text}\n"

        if len(retrieved_contexts) > 0:
            prompt += "\nFor reference, these are the relevent keyword I extracted using all-MiniLM-L6-v2: "

        for i, ctx in enumerate(retrieved_contexts):
            prompt += f"\nKeyword {i+1}: {ctx}"

        prompt += (
            f"\n---\nNow, Provide the top {top_k} keywords in descending order."
        )

        return prompt

    def query_ollama(self, prompt, model="llama3.1:8b"):
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

    def get_llm_rank_keywords(self, text, top_k=5, retrieved_contexts=[]):            
        prompt = self.build_prompt(text, top_k, retrieved_contexts)
        self.query_ollama(prompt)

    def rank_keywords(self, text, top_k=5, method="embedding"):
        """
        Rank candidate keywords by semantic similarity to the passage.
        method = 'embedding' (Sentence-Transformers) or 'llm' (if you integrate your LLM).
        """
        # Extract candidate single/multi-word terms
        candidates = self.extract_candidate_phrases(text)

        if method == "embedding":
            return self.get_embedding_rank_keywords(text, candidates, top_k)

        elif method == "llm":
            return self.get_llm_rank_keywords(text, top_k)

# ------------------ Example ------------------
if __name__ == "__main__":
    extractor = ExtractTopKeywords()
    passage = """We propose a domain-adversarial neural network for hyperspectral anomaly detection.
    Our method integrates image-level and instance-level alignment with an ensemble strategy,
    reducing domain shift between source and target datasets."""

    print("Embedding-based keywords:")
    print(extractor.rank_keywords(passage, top_k=5, method="embedding"))

    print("\nLLM-based keywords:")
    print(extractor.rank_keywords(passage, top_k=5, method="llm"))
