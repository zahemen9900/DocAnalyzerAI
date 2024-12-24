import pandas as pd
from pathlib import Path
from chatbot_2 import FinancialChatbot

import sys
sys.path.append('../src')


def test_financial_chat():
    # Initialize chatbot
    chatbot = FinancialChatbot()
    
    # Load financial data
    data_path = Path("../data/AAPL/AAPL_financials.csv")
    financial_data = pd.read_csv(data_path)
    
    # Test queries
    test_queries = [
        "What was the revenue in 2024?",
        "Show me the growth rate for net income",
        "What's the trend in operating income?",
        "Compare research and development expenses between 2023 and 2024",
        "Can you explain what net income means?"
    ]
    
    print("\nTesting Financial Chatbot:")
    print("-" * 50)
    
    for query in test_queries:
        print(f"\nUser: {query}")
        response = chatbot.chat(query, financial_data)
        print(f"Bot: {response}")
        
        # Show suggested follow-up queries
        suggestions = chatbot.suggest_related_queries(chatbot.last_metric_discussed)
        if suggestions:
            print("\nSuggested queries:")
            for suggestion in suggestions:
                print(f"- {suggestion}")

if __name__ == "__main__":
    test_financial_chat()
