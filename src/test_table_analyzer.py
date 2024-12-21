from table_analyzer import TableAnalyzer
from pathlib import Path

def main():
    # Initialize analyzer
    analyzer = TableAnalyzer()

    # Load one of your financial CSVs
    csv_path = Path("../data/apple_financials.csv")
    df = analyzer.load_csv(csv_path)

    # Test some example queries
    test_queries = [
        "What was the Net sales value for 2024?",
        "What was the Operating income for 2023?",
        "What was the Research and development expense in 2024?",
        "What is the Net income for 2023?"
    ]

    print("\nQuerying Apple financial data:")
    print("-" * 50)
    for query in test_queries:
        answer = analyzer.query_table(df, query)
        print(f"Q: {query}")
        print(f"A: {answer}\n")

if __name__ == "__main__":
    main()
