import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
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
        print("Initializing the NLTK library.")
        nltk_download = NltkDownload()
        nltk.data.path.append(nltk_download.download_dir)

        # Load embedding model (you can swap for a local LLM embedding generator)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")  # small & fast

        self.json_dir = "./ollama_response"
        os.makedirs(self.json_dir, exist_ok=True)

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
        for n in range(1, max_ngram + 1):
            for i in range(len(words) - n + 1):
                cand = " ".join(words[i : i + n])
                candidates.append(cand)

        # return unique phrases
        return list(set(candidates))

    def get_embedding_rank_keywords(self, text, candidates, top_k=5):
        # Encode passage and candidates
        text_emb = self.model.encode(text, convert_to_tensor=True)
        cand_embs = self.model.encode(candidates, convert_to_tensor=True)

        # Compute similarity
        scores = util.cos_sim(text_emb, cand_embs)[0].cpu().tolist()

        # Rank by similarity
        scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def get_rank_by_frequency(self, text, candidates, top_k=5):
        """
        Rank candidates by frequency count.
        """
        words = nltk.word_tokenize(text.lower())
        counts = Counter(words)
        scored = sorted(
            [(c, sum(counts[w] for w in c.split())) for c in candidates],
            key=lambda x: x[1],
            reverse=True,
        )
        # print(f"Ranking candidates by frequency count: {scored[:2*top_k]}")
        return scored[:top_k]

    def get_rank_by_hybrid(self, text, candidates, top_k=5):
        # Get both scores
        emb_ranking = dict(
            self.get_embedding_rank_keywords(text, candidates, top_k * 2)
        )
        freq_ranking = dict(self.get_rank_by_frequency(text, candidates, top_k * 2))

        # Normalize scores
        if emb_ranking:
            max_emb = max(emb_ranking.values())
            emb_ranking = {k: v / max_emb for k, v in emb_ranking.items()}
        if freq_ranking:
            max_freq = max(freq_ranking.values())
            freq_ranking = {k: v / max_freq for k, v in freq_ranking.items()}

        # Combine (embedding + frequency)
        combined = {}
        for cand in set(list(emb_ranking.keys()) + list(freq_ranking.keys())):
            combined[cand] = emb_ranking.get(cand, 0) + freq_ranking.get(cand, 0)

        # Sort by combined score
        scored = sorted(combined.items(), key=lambda x: x[1], reverse=True)
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
        prompt = "You are a technical literature summarization assistant. You are great at finding most relevent keywords from technical articles, when asked you provide the top keywords in a comma separated list. A keyword should be a clean and concise phrase consisting of one to three words."
        prompt += f"\nExtract keywords from this passage: {text}\n"

        if len(retrieved_contexts) > 0:
            prompt += "\nFor reference, these are the relevent keyword I extracted using all-MiniLM-L6-v2: "

        for i, ctx in enumerate(retrieved_contexts):
            prompt += f" Keyword {i+1}: {ctx}, "

        prompt += f"\n---\nNow, provide the top {top_k} most relevant keywords in descending order of their relevance. Your response must end with the keywords as a comma-separated list."

        return prompt

    def query_ollama(self, prompt, model="llama3.1:8b"):
        """
        Query the Ollama language model to get a response based on the given prompt.

        Args:
            prompt (str): The prompt to send to the LLM.
            model (str, optional): The name of the LLM model to use. Defaults to "llama3.1:8b".

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
        json_path = os.path.join(self.json_dir, f"{timestamp_str}.json")
        with open(json_path, "w", encoding="utf-8") as jf:
            json_data["prompt"] = prompt
            json.dump(json_data, jf, indent=2)

        return json_data["response"]

    def get_llm_rank_keywords(self, text, top_k=5, retrieved_contexts=[]):
        prompt = self.build_prompt(text, top_k, retrieved_contexts)
        response = self.query_ollama(prompt)
        print(f"LLM Response: {response}")

        keywords = response.split("\n")
        for keyword in keywords:
            keyword = keyword.split(",")
            if len(keyword) == top_k:
                new_keywords = []
                for k in keyword:
                    k = k.split(":")[-1]
                    new_keywords.append(k.strip().lower())
                return new_keywords
        
        return []

    def rank_keywords(self, text, top_k=5, method="embedding"):
        """
        Rank candidate keywords by semantic similarity to the passage.
        method = 'embedding' (Sentence-Transformers) or 'llm' (if you integrate your LLM).
        """
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)

        # Extract candidate single/multi-word terms
        candidates = self.extract_candidate_phrases(text)

        if method == "embedding":
            scored = self.get_embedding_rank_keywords(text, candidates, top_k)
            return [kw for kw, _ in scored]

        elif method == "frequency":
            scored = self.get_rank_by_frequency(text, candidates, top_k)
            return [kw for kw, _ in scored]

        elif method == "hybrid":
            return self.get_rank_by_hybrid(text, candidates, top_k)

        elif method == "llm":
            return self.get_llm_rank_keywords(text, top_k)

        elif method == "llm_hybrid":
            scored = self.get_embedding_rank_keywords(text, candidates, top_k)
            keywords =  [kw for kw, _ in scored]
            return self.get_llm_rank_keywords(text, top_k, keywords)


# ------------------ Example ------------------
if __name__ == "__main__":
    extractor = ExtractTopKeywords()
    passage = """We propose a domain-adversarial neural network for hyperspectral anomaly detection.
    Our method integrates image-level and instance-level alignment with an ensemble strategy,
    reducing domain shift between source and target datasets."""

    keywords = extractor.rank_keywords(passage, top_k=5, method="embedding")
    print(f"\nEmbedding-based keywords: {keywords}.")

    keywords = extractor.rank_keywords(passage, top_k=5, method="frequency")
    print(f"\nFrequency-based keywords: {keywords}")

    keywords = extractor.rank_keywords(passage, top_k=5, method="hybrid")
    print(f"\nHybrid keywords: {keywords}")

    keywords = extractor.rank_keywords(passage, top_k=5, method="llm")
    print(f"\nLLM-based keywords: {keywords}.")

    keywords = extractor.rank_keywords(passage, top_k=5, method="llm_hybrid")
    print(f"\nLLM-based keywords: {keywords}.")
