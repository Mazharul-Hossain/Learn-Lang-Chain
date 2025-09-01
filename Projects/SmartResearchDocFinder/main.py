from ExtractTopKeywords import ExtractTopKeywords
from semantic_scholar import SemanticScholar


def main():
    """
    Main function to execute the script.
    It prompts the user to paste an abstract or literature, ranks keywords using a hybrid method,
    and then creates corpora based on those keywords.
    """
    extractor = ExtractTopKeywords()

    print("Paste the abstract/ literaure here:")
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

    keywords = extractor.rank_keywords(passage, top_k=5, method="llm_hybrid")
    print(f"\nLLM-based keywords: {keywords}.")

    scholar = SemanticScholar()
    scholar.create_corpas(keywords)


if __name__ == "__main__":
    main()
