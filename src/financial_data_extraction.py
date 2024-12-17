import pandas as pd
from pathlib import Path
import logging
from edgar import (
    set_identity, Company
)
from fetch_ticker import get_company_ticker
from IPython.display import display
from typing import Optional, Union, Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExtractFinancialData:
    """A class to extract and process financial data from SEC EDGAR database.

    This class handles the extraction of financial statements (income and cashflow) 
    from company 10-K filings using the EDGAR database.

    Attributes:
        ticker (str): The stock ticker symbol of the company
        credentials (str): EDGAR API credentials in format "Name email@domain.com"
        company (Optional[Company]): Edgar Company object once initialized
        cashflow_df (Optional[pd.DataFrame]): Processed cashflow statements
        income_df (Optional[pd.DataFrame]): Processed income statements  
        company_financial_df (Optional[pd.DataFrame]): Combined financial statements
        PATH (Path): Directory path for saving output files
    """

    # Class-level constants
    CASHFLOW_COLUMNS_TO_DROP = [
        'Cash, cash equivalents, and restricted cash and cash equivalents, ending balances',
        'Share-based compensation expense',
        'Other',
        'Vendor non-trade receivables',
        'Payments for taxes related to net share settlement of equity awards',
        'Proceeds from/(Repayments of) commercial paper, net',
    ]

    INCOME_COLUMNS_TO_DROP = [
        'Products',
        'Services',
        'Basic (in shares)',
        'Diluted (in shares)'
    ]

    FLOAT_COLUMNS = ['Basic (in dollars per share)', 'Diluted (in dollars per share)']

    def __init__(self, ticker: str, credentials: str = "David Yeboah davidzahemenyeboah@gmail.com"):
        """Initialize the ExtractFinancialData object.

        Args:
            ticker (str): Company stock ticker symbol
            credentials (str, optional): EDGAR API credentials. Defaults to provided value.
        """
        if not isinstance(ticker, str) or not ticker.strip():
            logger.error("Invalid ticker provided")
            raise ValueError("Ticker must be a non-empty string")
            
        self.ticker = ticker.upper()
        self.credentials = credentials
        self.company = None
        self.cashflow_df = None
        self.income_df = None
        self.company_financial_df = None
        
        # Initialize data directory
        self.PATH = Path.cwd() / 'data'
        self.PATH.mkdir(exist_ok=True)
        logger.info(f"Initialized ExtractFinancialData for {self.ticker}")

    def _initialize_edgar_connection(self) -> None:
        """Set up the EDGAR API connection with credentials."""
        try:
            set_identity(self.credentials)
            logger.info("EDGAR API connection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize EDGAR API connection: {e}")
            raise

    def retrieve_company_info(self) -> Company:
        """Initialize and return the EDGAR Company object for the specified ticker.

        Returns:
            Company: Initialized EDGAR Company object

        Raises:
            Exception: If company initialization fails
        """
        try:
            self._initialize_edgar_connection()
            self.company = Company(self.ticker)
            logger.info(f"Successfully retrieved company info for {self.ticker}")
            return self.company
        except Exception as e:
            logger.error(f"Failed to initialize company for ticker {self.ticker}: {e}")
            raise

    def _process_dataframe(self, df: pd.DataFrame, float_columns: List[str] = None) -> pd.DataFrame:
        """Process dataframe by converting data types appropriately.

        Args:
            df (pd.DataFrame): Input dataframe to process
            float_columns (List[str], optional): Columns to convert to float. Defaults to None.

        Returns:
            pd.DataFrame: Processed dataframe
        """
        try:
            if float_columns:
                for col in df.columns:
                    if col not in float_columns:
                        df[col] = df[col].astype(int)
                df[float_columns] = df[float_columns].astype('Float32')
            else:
                for col in df.columns:
                    df[col] = df[col].astype(int)
            return df
        except Exception as e:
            logger.error(f"Error processing dataframe: {e}")
            raise

    def retrieve_10k_cashflows(self) -> Optional[pd.DataFrame]:
        """Retrieve and process the 3 most recent 10-K cashflow statements.

        Returns:
            Optional[pd.DataFrame]: Processed cashflow statements or None if processing fails
        """
        try:
            if self.company is None:
                self.company = self.retrieve_company_info()

            logger.info(f"Retrieving cashflow statements for {self.ticker}")
            df = self.company.financials.cashflow.to_dataframe().T
            df = self._process_dataframe(df)
            df = df.drop(self.CASHFLOW_COLUMNS_TO_DROP, axis=1)    

            self.cashflow_df = df
            logger.info(f"Successfully processed cashflow statements for {self.ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to retrieve cashflow statements: {e}")
            return None
    
    def retrieve_10k_income(self) -> Optional[pd.DataFrame]:
        """Retrieve and process the 3 most recent 10-K income statements.

        Returns:
            Optional[pd.DataFrame]: Processed income statements or None if processing fails
        """
        try:
            if self.company is None:
                self.company = self.retrieve_company_info()

            logger.info(f"Retrieving income statements for {self.ticker}")
            df = self.company.financials.income.to_dataframe().T
            df = self._process_dataframe(df, self.FLOAT_COLUMNS)

            # Calculate growth rates
            df['Net_Sale_Growth_rate'] = (df['Net sales'] / df['Net sales'].shift(1) - 1) * 100
            df['Net_Income_Growth_rate'] = (df['Net income'] / df['Net income'].shift(1) - 1) * 100
            df.fillna(0, inplace=True)

            df = df.drop(self.INCOME_COLUMNS_TO_DROP, axis=1)
            self.income_df = df
            logger.info(f"Successfully processed income statements for {self.ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to retrieve income statements: {e}")
            return None

    def concat_statements(self) -> Optional[pd.DataFrame]:
        """Combine income and cashflow statements into a single dataframe.

        Returns:
            Optional[pd.DataFrame]: Combined financial statements or None if concatenation fails
        """
        try:
            if self.income_df is None:
                self.income_df = self.retrieve_10k_income()

            if self.cashflow_df is None:
                self.cashflow_df = self.retrieve_10k_cashflows()

            if self.income_df is not None and self.cashflow_df is not None:
                self.company_financial_df = pd.concat([self.income_df, self.cashflow_df], axis=1)
                logger.info("Successfully merged financial statements")
                return self.company_financial_df
            else:
                raise ValueError("Income or Cashflow dataframes are None")
                
        except Exception as e:
            logger.error(f"Failed to concatenate financial statements: {e}")
            return None

    def save_all_to_csv(self) -> bool:
        """Save all dataframes to CSV files in the data directory.

        Returns:
            bool: True if all saves successful, False otherwise
        """
        try:
            if not all([self.income_df is not None, 
                       self.cashflow_df is not None,
                       self.company_financial_df is not None]):
                raise ValueError("One or more dataframes are not initialized")

            dfs: Dict[str, pd.DataFrame] = {
                f'income_{self.ticker}.csv': self.income_df,
                f'cashflow_{self.ticker}.csv': self.cashflow_df,
                f'{self.ticker}_financials.csv': self.company_financial_df
            }

            for filename, df in dfs.items():
                filepath = self.PATH / filename
                df.to_csv(filepath)
                logger.info(f'Saved {filename}')

            logger.info(f'Successfully saved all CSV files to {self.PATH}')
            return True
            
        except Exception as e:
            logger.error(f"Failed to save CSV files: {e}")
            return False


def main():
    try:
        ticker = get_company_ticker()
        if ticker:
            pipeline = ExtractFinancialData(ticker)
            display(pipeline.retrieve_company_info())
            
            final_financial_df = pipeline.concat_statements()
            if final_financial_df is not None:
                pipeline.save_all_to_csv()
                print(final_financial_df)
            else:
                logger.error("Failed to generate financial statements")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")


if __name__ == "__main__":
    main()