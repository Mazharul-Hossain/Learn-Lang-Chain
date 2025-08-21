"""Origin: https://github.com/kaushik-bhat/Leetcode-problem-scraper/blob/main/fetch.py"""

import requests
import json

LEETCODE_ALGORITHMS_URL = "https://leetcode.com/api/problems"
OUTPUT_FILENAME = "./Projects/LeetcodeCheatcode/leetcode_algorithms_raw.json"
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


def fetch_algorithm_problem_data(url="algorithms"):
    """
    Fetches algorithm problem data from LeetCode API and saves it to a JSON file.

    Args:
        url (str, optional): The endpoint of the LeetCode API to fetch data from.
            Can be "algorithms", "database", "shell", or "all". Defaults to "algorithms".

    Returns:
        None
    """
    url = f"{LEETCODE_ALGORITHMS_URL}/{url}"
    print(f"Fetching data from {url}...")

    try:
        response = requests.get(url, headers=REQUEST_HEADERS)
        response.raise_for_status()

        api_data = response.json()
        problem_list = api_data.get("stat_status_pairs", [])

        if not problem_list:
            print("No problem data was found in the API response.")
            return

        print(f"Found data for {len(problem_list)} algorithm problems.")

        with open(OUTPUT_FILENAME, "w", encoding="utf-8") as output_file:
            json.dump(problem_list, output_file, ensure_ascii=False, indent=4)

        print(
            f"Successfully saved all raw algorithm problem data to '{OUTPUT_FILENAME}'"
        )

    except requests.exceptions.RequestException as error:
        print(f"An error occurred during the network request: {error}")
    except json.JSONDecodeError:
        print("Failed to parse the JSON response from the server.")
    except IOError as error:
        print(f"Could not write to the file '{OUTPUT_FILENAME}': {error}")


if __name__ == "__main__":
    fetch_algorithm_problem_data("all")
