from ExtractTopKeywords import ExtractTopKeywords


def main():
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

    print("\nLLM-based keywords:")
    print(extractor.rank_keywords(passage, top_k=5, method="llm"))

if __name__ == "__main__":
    main()