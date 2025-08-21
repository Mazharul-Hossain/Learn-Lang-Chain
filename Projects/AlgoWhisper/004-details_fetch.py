"""Origin: https://github.com/kaushik-bhat/Leetcode-problem-scraper/blob/main/process.py"""

import copy
import json
import os
import pandas
import requests
from bs4 import BeautifulSoup
import time
import re


FOLDER = "./leetcode"
JSON_FOLDER = "./Projects/LeetcodeCheatcode/JSON"
CSV_FILENAME = "./Projects/LeetcodeCheatcode/leetcode_problems.csv"

LEETCODE_GRAPHQL_URL = "https://leetcode.com/graphql"
GRAPHQL_REQUEST_HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
}
GRAPHQL_QUERY = """
query getQuestionDetail($titleSlug: String!) {
  question(titleSlug: $titleSlug) {
    questionFrontendId
    title
    content
    difficulty
    topicTags {
        slug
    }
  }
}
"""

REQUEST_DELAY_SECONDS = 2.0
REQUEST_TIMEOUT_SECONDS = 20
SAVE_PROGRESS_INTERVAL = 100


def load_problem_metadata():
    """
    Load problem metadata from a CSV file.

    Returns:
        pandas.DataFrame: DataFrame containing problem metadata.
    """
    df = pandas.read_csv(CSV_FILENAME, index_col=0)
    return df


def clean_html_description(html_content):
    """
    Clean HTML description to extract plain text and format it.

    Args:
        html_content (str): HTML content of the problem description.

    Returns:
        dict: Dictionary containing cleaned description parts.
    """
    if not html_content:
        return {}

    soup = BeautifulSoup(html_content, "html.parser")
    plain_text = soup.get_text()
    normalized_text = " ".join(plain_text.split())

    cleaned_description = {}
    for key in ["Follow up", "Constraints", "Example"]:
        constraints_match = re.search(rf"{key}", normalized_text, re.IGNORECASE)

        truncation_index = -1
        if constraints_match:
            truncation_index = constraints_match.start()

        if truncation_index != -1:
            cleaned_description[key] = normalized_text[truncation_index:].strip()
            normalized_text = normalized_text[:truncation_index]

        else:
            cleaned_description[key] = ""

    Constraints = cleaned_description["Constraints"]
    Constraints = Constraints.replace("105", "10**5")
    Constraints = Constraints.replace("104", "10**4")
    cleaned_description["Constraints"] = Constraints

    cleaned_description["Description"] = normalized_text.strip()

    return cleaned_description


def fetch_details(slug):
    """
    Fetch details for a problem from LeetCode using GraphQL.

    Args:
        slug (str): The slug of the problem.

    Returns:
        dict: Dictionary containing fetched details or an empty dictionary if an error occurs.
    """
    print(f"\nRequesting Leetcode for {slug}")
    payload = {"query": GRAPHQL_QUERY, "variables": {"titleSlug": slug}}

    try:
        response = requests.post(
            LEETCODE_GRAPHQL_URL,
            json=payload,
            headers=GRAPHQL_REQUEST_HEADERS,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        graphql_data = response.json()

        question_details = graphql_data.get("data", {}).get("question")
        # print(f"question_details: {question_details}")

        if question_details and question_details.get("content"):
            cleaned_description = clean_html_description(question_details["content"])
            cleaned_description["difficulty_str"] = question_details["difficulty"]

            topicTags = question_details.get("topicTags")
            if topicTags:
                topics = []
                for topic in topicTags:
                    if topic.get("slug"):
                        topics.append(topic.get("slug"))
                cleaned_description["topicTags"] = ", ".join(topics)

            # print(f"cleaned_description: {cleaned_description}")
            return cleaned_description

    except requests.exceptions.RequestException as error:
        print(f"Network error for slug '{slug}': {error}")
    except Exception as error:
        print(f"An unexpected error occurred for slug '{slug}': {error}")

    return {}


def extract_problem_metadata(file_path):
    """
    Extract problem metadata from a Python file.

    Args:
        file_path (str): Path to the Python file.

    Returns:
        dict: Dictionary containing extracted metadata.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    metadata = {}
    comment_lines, code_lines = [], []
    comment_flag = False

    # Basic heuristic: parse top comment block for metadata
    for line in lines:
        if '"""' in line:
            comment_lines.append(line.strip('# """ \n'))
            comment_flag = not comment_flag
        elif comment_flag:
            comment_lines.append(line.strip('# """ \n'))
        elif line.strip().startswith("#"):
            comment_lines.append(line.strip("# \n"))
        else:
            code_lines.append(line)

    metadata["comments"] = "\n".join(comment_lines)
    metadata["solution_code"] = "".join(code_lines)

    return metadata


def process_and_fetch_details(problem_df):
    """
    Process and fetch details for problems from a DataFrame.

    Args:
        problem_df (pandas.DataFrame): DataFrame containing problem metadata.
    """
    for root, _, files in os.walk(FOLDER):
        for file in files:
            if not file.endswith(".py"):
                continue

            parts = file.split("_")
            first_part = parts[0]
            if not first_part.isdigit():
                continue

            first_part = int(first_part)
            print(f"\n->Start #{first_part}: {root}/{file}")

            # Get row as dictionary from "problem_df" using "first_part" as index.
            try:
                row_dict = problem_df.loc[first_part].to_dict()
            except Exception as ex:
                continue

            row_dict["id"] = f"{first_part}"
            # print(row_dict)  # Example usage of the fetched row

            source_file = os.path.join(root, file)
            row_dict["source_file"] = source_file

            json_path = os.path.join(JSON_FOLDER, f"{row_dict['title_slug']}.json")
            if os.path.exists(json_path):
                print(f"The path '{json_path}' exists.")
                with open(json_path, "r", encoding="utf-8") as f:
                    problem = json.load(f)
                    problem_copy = copy.deepcopy(problem)

                    problem.update(row_dict)
                    row_dict = problem

            metadata = extract_problem_metadata(source_file)
            if metadata:
                row_dict.update(metadata)

            # print(row_dict)

            keys_to_check = [
                "Follow up",
                "Constraints",
                "Example",
                "Description",
                "difficulty_str",
                "topicTags",
            ]
            # Check if all keys in keys_to_check exist
            if all(key in row_dict for key in keys_to_check):
                print("All keys in keys_to_check are present.")
            else:
                cleaned_description = fetch_details(row_dict["title_slug"])
                if cleaned_description:
                    row_dict.update(cleaned_description)

            # print(row_dict)

            if problem_copy != row_dict:
                with open(json_path, "w", encoding="utf-8") as jf:
                    json.dump(row_dict, jf, indent=2)
                print(f"Saved: {json_path}")

            else:
                print("JSON data unchanged. No write operation performed.")

            time.sleep(REQUEST_DELAY_SECONDS)


def main():
    """
    Main function to load problem metadata and process details.
    """
    problem_df = load_problem_metadata()
    process_and_fetch_details(problem_df)


if __name__ == "__main__":
    main()
