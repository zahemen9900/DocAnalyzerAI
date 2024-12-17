import requests
import pandas as pd
from IPython.display import display

headers = {'User-Agent': 'davidzahemenyeboah@gmail.com'}

companyTickers = requests.get(
    'https://www.sec.gov/files/company_tickers.json', 
    headers=headers, timeout=30
)

# print(companyTickers.json())

first_company = companyTickers.json()['0']

# Parse the CIK from the company tickers without leading zeros
directCIK = companyTickers.json()['0']['cik_str']
print(directCIK)

companyData = pd.DataFrame.from_dict(companyTickers.json(), orient = 'index')
print(companyData.head())

# Add leading zeros to the CIK
companyData['cik_str'] = companyData['cik_str'].apply(lambda x: f'{x:010d}')
print(companyData.head())

cik = companyData.loc[:,'cik_str'].iloc[0] # Get the first company's CIK
print(cik)


# Get some company-specific filing metadata

filingMetadata = requests.get(
    f'https://data.sec.gov/submissions/CIK{cik}.json',
    headers=headers, timeout=30
)

# print(filingMetadata.json())

# Get all the allforms for the company into a dataframe
allforms = pd.DataFrame.from_dict(
    filingMetadata.json()['filings']['recent']
)
display(allforms.head())
display(allforms.loc[allforms['primaryDocDescription'] == '10-K'].head(7))
