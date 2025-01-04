import json

def get_company_ticker(json_file_path: str = "/home/zahemen/projects/dl-lib/DocAnalyzerAI/data/company_tickers.json") -> str:
    """
    Searches for a company ticker based on user-provided input.
    
    Args:
        json_file_path (str): Path to the company_tickers.json file.

    Returns:
        str: The ticker symbol of the selected company.
    """
    # Step 1: Load the JSON file
    try:
        with open(json_file_path, 'r') as file:
            company_data = json.load(file)
    except FileNotFoundError:
        print(f"Error: The file '{json_file_path}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file '{json_file_path}' contains invalid JSON.")
        return None
    
    # Step 2: Prompt user for input
    user_input = input("Enter the company name or part of the name: ").strip().lower()

    # Step 3: Search for matches in the titles
    matches = [
        {"title": company["title"], "ticker": company["ticker"]}
        for company in company_data.values()
        if user_input in company["title"].lower()
    ]

    # Step 4: Handle cases based on the number of matches
    if not matches:
        print("No matches found for the given company name.")
        exit()
        return None
    elif len(matches) == 1:
        print(f"Found one match: {matches[0]['title']}")
        return matches[0]["ticker"]
    else:
        print("Multiple matches found:")
        for idx, match in enumerate(matches, start=1):
            print(f"{idx}. {match['title']}")
        
        # Step 5: Prompt user to select one from the list
        while True:
            try:
                choice = int(input(f"Select the number corresponding to the company (1-{len(matches)}): "))
                if 1 <= choice <= len(matches):
                    selected_match = matches[choice - 1]
                    return selected_match["ticker"]
                else:
                    print("Invalid choice. Please enter a valid number.")
            except ValueError:
                print("Invalid input. Please enter a number.")

# Example usage
if __name__ == "__main__":
    # json_file_path = "/home/zahemen/projects/dl-lib/DocAnalyzerAI/data/company_tickers.json"  # Update with the correct path to your JSON file
    ticker = get_company_ticker()
    if ticker:
        print(f"The ticker for the selected company is: {ticker}")
