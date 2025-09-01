import json
import os
import time
import traceback
import requests
from datetime import datetime
import urllib.parse

REQUEST_DELAY_SECONDS = 2.0


class SemanticScholar:
    def __init__(self):
        print("Initializing the SemanticScholar library.")

        self.json_dir = "./scholar_response"
        os.makedirs(self.json_dir, exist_ok=True)

        self.url = "https://api.semanticscholar.org/graph/v1/paper/search"

        # Get the current date and time
        current_datetime = datetime.now()

        # Extract the year attribute
        current_year = current_datetime.year

        self.year = f"{current_year-2}-{current_year}"
        self.fields = "title,abstract,authors,year,url"
        # print(self.year, self.fields)

    def get_query(self, query: list):
        query = "+".join(query)
        encoded_query = urllib.parse.quote(query)
        return f"{self.url}?query={encoded_query}&year={self.year}&fields={self.fields}&limit=10"

    def query_semantic_scholar(self, query: list, json_path:str=""):
        """
        Query the Ollama language model to get a response based on the given prompt.

        Args:
            prompt (str): The prompt to send to the LLM.
            model (str, optional): The name of the LLM model to use. Defaults to "llama3.1:8b".

        Returns:
            str: The response from the LLM.
        """
        url = self.get_query(query)
        headers = {"Content-Type": "application/json"}

        response = requests.get(url, headers=headers)
        response.raise_for_status()
        json_data = response.json()

        if json_path:
            with open(json_path, "w", encoding="utf-8") as jf:
                json_data["url"] = url
                json.dump(json_data, jf, indent=2)

        return list(json_data["data"])

    def create_corpas(self, query: list):
        current_datetime = datetime.now()
        timestamp_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

        data = []
        for i in range(len(query)):
            if i > 0:
                time.sleep(REQUEST_DELAY_SECONDS)

            json_path = os.path.join(self.json_dir, f"{timestamp_str}_{i:02d}.json")
            try:
                new_data = self.query_semantic_scholar(query[i : i + 2], json_path)
                data.extend(new_data)
            except:
                print(traceback.format_exc())

        return data
