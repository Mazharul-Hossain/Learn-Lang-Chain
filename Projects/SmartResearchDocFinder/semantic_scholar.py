import json
import os
import queue
import random
import re
import threading
import time
import traceback
import requests
import itertools
import urllib.parse
from datetime import datetime
from dotenv import load_dotenv
from helper import extract_json_info, query_ollama

REQUEST_DELAY_SECONDS = 2.0
# A sentinel object to signal the consumer to stop
STOP_SIGNAL = None


class SemanticScholar:
    def __init__(self, text=None):
        """
        Initializes the SemanticScholar library.

        This method sets up the necessary directories and initializes variables for querying the Semantic Scholar API.
        """
        print("Initializing the SemanticScholar library.")

        self.json_dir = "./scholar_response"
        os.makedirs(self.json_dir, exist_ok=True)

        # Load variables from .env
        load_dotenv()
        self.api_key = os.getenv("API_KEY")

        self.url = "https://api.semanticscholar.org/graph/v1/paper/search"

        # Get the current date and time
        current_datetime = datetime.now()

        # Extract the year attribute
        current_year = current_datetime.year

        self.year = f"{current_year-2}-{current_year}"
        self.fields = "title,abstract,authors,year,url"
        # print(self.year, self.fields)

        self.queue = queue.Queue(maxsize=10)
        self.text = text
        self.results = {}

    def get_query(self, query: list):
        """
        Constructs a query string for the Semantic Scholar API.

        Args:
            query (list): A list of keywords or phrases to search for.

        Returns:
            str: The constructed query string.
        """
        query = "+".join(query)
        encoded_query = urllib.parse.quote(query)
        return f"{self.url}?query={encoded_query}&year={self.year}&fields={self.fields}&limit=10"

    def query_semantic_scholar(self, query: list, json_path: str = ""):
        """
        Queries the Semantic Scholar API and returns the results.

        Args:
            query (list): A list of keywords or phrases to search for.
            json_path (str, optional): The path where the JSON response should be saved. Defaults to an empty string.

        Returns:
            list: A list of dictionaries containing the search results.
        """
        url = self.get_query(query)
        headers = {"Content-Type": "application/json", "x-api-key": self.api_key}

        response = requests.get(url, headers=headers)
        response.raise_for_status()
        json_data = response.json()

        if json_path:
            with open(json_path, "w", encoding="utf-8") as jf:
                json_data["url"] = url
                json.dump(json_data, jf, indent=2)

        return list(json_data["data"])

    def build_prompt(self, retrieved_context: str):
        """
        Builds a prompt for the LLM by combining the new problem text with similar problems.

        Args:
            retrieved_context (str): The new problem statement to solve.

        Returns:
            str: The constructed prompt for the LLM.
        """
        prompt = "You are a technical synopsis assistant. You excel at identifying the technical relevance between two passages. You will receive an original passage along with a second passage from different sources. When asked, you will compare two passages to determine their relevance and must respond in JSON format that starts with ```json. First `relevant` key in True or False. If they are relevant, set the `relevant` key to True and write a synopsis of the second passage in clean and concise paragraph consisting of one to three sentences with at most fifty words in the second `summary` key."

        prompt += f"\nHere is the original passage: {self.text}\n"
        prompt += f"\n---\nHere is the second passage: {retrieved_context.lower()}\n"
        prompt += f"\n---\nNow, provide me the JSON response."

        return prompt

    def _producer_task(self, query: list):
        """
        Creates a corpus of search results by querying the Semantic Scholar API for each pair of queries.

        Args:
            query (list): A list of keywords or phrases to search for.

        Returns:
            list: A list of dictionaries containing the search results.
        """
        current_datetime = datetime.now()
        timestamp_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

        delay = REQUEST_DELAY_SECONDS
        combinations_4c2 = list(itertools.combinations(query[1:], 2))

        for i, q in combinations_4c2:
            if i > 0:
                # sleep with added jitter
                time.sleep(delay + random.uniform(0, delay * 0.1))

                # Exponential increase is 2
                delay *= 1.5

            json_path = os.path.join(self.json_dir, f"{timestamp_str}_{i:02d}.json")
            try:
                q = query[:1] + list(q)

                data = self.query_semantic_scholar(q, json_path)

                for item in data:
                    self.queue.put(item)
            except:
                print(traceback.format_exc())

    def _consumer_task(self, consumer_id=0):
        """Task run by a consumer thread."""
        while True:
            item = self.queue.get()
            if item is STOP_SIGNAL:
                # Put the signal back for other consumers before exiting
                self.queue.put(STOP_SIGNAL)
                print(f"Consumer-{consumer_id}: Exiting.")
                break
            self.queue.task_done()

            try:
                abstract = item["abstract"]
                if not abstract:
                    print(f'\n# Missing {item["paperId"]}: {item["title"]}')
                    continue

                if item["paperId"] in self.results:
                    continue

                prompt = self.build_prompt(abstract)
                response = query_ollama(prompt)
                response = extract_json_info(response)
                if response and response["relevant"]:
                    item["summary"] = response["summary"]
                    self.results[item["paperId"]] = item
            except:
                print(traceback.format_exc())

    def create_corpus(self, text, query: list):
        """
        Creates a corpus of search results by querying the Semantic Scholar API for each pair of queries.

        Args:
            text (str): The new problem statement to solve.
            query (list): A list of keywords or phrases to search for.

        Returns:
            None
        """
        text = text.replace("\n", " ")
        self.text = re.sub(r"\s+", " ", text)
        self.text = self.text.lower()

        producer = threading.Thread(target=self._producer_task, args=(query,))
        producer.start()

        consumer = threading.Thread(target=self._consumer_task)
        consumer.start()

        producer.join()
        print("All producers have finished. Signaling consumers to stop.")
        self.queue.put(STOP_SIGNAL)

        consumer.join()
        print("Producer-consumer system has shut down gracefully.")

    def save_corpus(self):
        """
        Saves the corpus of search results to a JSON file.

        Returns:
            None
        """
        current_datetime = datetime.now()
        timestamp_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

        json_path = os.path.join(self.json_dir, f"LLM_relevence_{timestamp_str}.json")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(self.results, jf, indent=2)

    def test_consumer(self, text):
        """
        Tests the consumer task by feeding it a predefined set of data.

        Args:
            text (str): The new problem statement to solve.

        Returns:
            None
        """
        text = text.replace("\n", " ")
        self.text = re.sub(r"\s+", " ", text)
        self.text = self.text.lower()

        consumer = threading.Thread(target=self._consumer_task)
        consumer.start()

        json_path = "./scholar_response/2025-09-01_13-35-59_02.json"
        with open(json_path, encoding="utf-8") as jf:
            json_data = json.load(jf)

        data = json_data["data"]
        for item in data:
            self.queue.put(item)
        self.queue.put(STOP_SIGNAL)

        consumer.join()
        print("Producer-consumer system has shut down gracefully.")

        self.save_corpus()


def main():
    """
    Main function to execute the script.
    It prompts the user to paste an abstract or literature, ranks keywords using a hybrid method,
    and then creates corpora based on those keywords.
    """
    print("Paste the abstract/ literature here:")
    counter, passage = 0, ""
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

        passage += line + " "

    scholar = SemanticScholar()
    scholar.test_consumer(passage)


if __name__ == "__main__":
    main()
