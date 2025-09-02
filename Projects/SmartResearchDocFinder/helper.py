import requests
import json
import os
from datetime import datetime


def query_ollama(prompt, model="llama3.1:8b", json_dir=None):
    """
    Queries the Ollama language model to get a response based on the given prompt.

    Args:
        prompt (str): The prompt to send to the LLM.
        model (str, optional): The name of the LLM model to use. Defaults to "llama3.1:8b".
        json_dir (str, optional): The directory to store the response from LLM.

    Returns:
        str: The response from the LLM.
    """
    if json_dir is None:
        json_dir = "./ollama_response"
    os.makedirs(json_dir, exist_ok=True)

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
    json_path = os.path.join(json_dir, f"{timestamp_str}.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json_data["prompt"] = prompt
        json.dump(json_data, jf, indent=2)

    return json_data["response"]


def extract_json_info(text):
    """
    Extracts JSON information from a given text.

    Args:
        text (str): The text containing the JSON data.

    Returns:
        dict or None: The extracted JSON data as a dictionary if successful; otherwise, None.
    """
    # Extract the JSON part from the text
    start_index = text.find("```json") + len("```json")
    end_index = text.find("```", start_index)
    json_str = text[start_index:end_index].strip()

    # Parse the JSON string
    try:
        json_data = json.loads(json_str)
        return json_data
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON: {e}")
        print(f"{start_index} to {end_index} json_str: {json_str}. \n\n{text}")

        return None
