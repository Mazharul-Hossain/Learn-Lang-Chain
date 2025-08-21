import json
import pandas

SAVE_PROGRESS_INTERVAL = 100
INPUT_FILENAME = "./Projects/LeetcodeCheatcode/leetcode_algorithms_raw.json"
OUTPUT_FILENAME = "./Projects/LeetcodeCheatcode/leetcode_problems.csv"

def load_problem_metadata():
    """
    Loads problem metadata from a JSON file and processes it to create a CSV file.
    
    This function reads the input JSON file, extracts relevant information about each problem,
    and saves the processed data into a CSV file. It also handles progress saving at regular intervals.
    """
    all_problems = []
    try:
        with open(INPUT_FILENAME, "r", encoding="utf-8") as file:
            all_problems = json.load(file)

    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_FILENAME}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{INPUT_FILENAME}'.")
        return None

    if not all_problems:
        return

    print(f"Starting to fetch details for {len(all_problems)} problems...")

    problem_list = []
    for index, problem_meta in enumerate(all_problems):
        slug = problem_meta.get("stat", {}).get("question__title_slug")
        if not slug:
            print(f"Skipping a problem due to missing slug: {problem_meta}")
            continue

        problem_dict = {}
        problem_dict["index"] = problem_meta.get("stat", {}).get("frontend_question_id")
        problem_dict["title_slug"] = slug
        problem_dict["title"] = problem_meta.get("stat", {}).get("question__title")
        problem_dict["difficulty"] = problem_meta.get("difficulty", {}).get("level")
        problem_dict["url"] = f"https://leetcode.com/problems/{slug}"

        problem_list.append(problem_dict)

        if (index + 1) == len(all_problems) or (
            (index + 1) % SAVE_PROGRESS_INTERVAL == 0
            and len(all_problems) > SAVE_PROGRESS_INTERVAL
        ):
            # Create a pandas DataFrame from the list of dictionaries, using "index" as the index column.
            df = pandas.DataFrame(
                problem_list, index=[item["index"] for item in problem_list]
            )

            # Drop the "index" column since it's now used as the index.
            df.drop(columns=["index"], inplace=True)

            # Write the DataFrame to a CSV file with UTF-8 encoding
            df.to_csv(OUTPUT_FILENAME, encoding="utf-8")
            print(f"Progress saved. {len(all_problems)} problems processed.")

if __name__ == "__main__":
    load_problem_metadata()
