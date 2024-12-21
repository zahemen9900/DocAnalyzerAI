This data folder contains the data used for the financial chatbot, fetched from the Edgar Online API using the `edgartools` package.

The data is stored in a JSON file, `data.json`, which contains the following fields:
- `company_name`: The name of the company
- `ticker`: The ticker symbol of the company
- `cik`: The Central Index Key of the company
- `form_type`: The type of form filed by the company
- `date_filed`: The date the form was filed
- `url`: The URL of the form


The data is extracted using the `ExtractFinancialData` class in the `financial_data_extraction.py` file.

The data is then stored in a JSON file, `data.json`, which is used by the `FinancialChatbot` class in the `financial_chatbot.py` file.

The data is also used by the `FinancialDataComparison` class in the `financial_data_comparison.py` file to compare the financial data of two companies.

The data is also used by the `FinancialDataVisualization` class in the `financial_data_visualization.py` file to visualize the financial data.

The data is also used by the `FinancialDataChatbot` class in the `financial_data_chatbot.py` file to chat with the financial data.

