import pandas as pd
from pathlib import Path
import logging
import sys
from IPython.display import display
from typing import Optional, Union, Dict, List
import warnings
from edgar import (
    set_identity, Company
)
sys.path.append('../src')
warnings.filterwarnings("ignore", category=FutureWarning)

from fetch_ticker import get_company_ticker


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

    # Class-level constants for columns that should be kept as decimals
    DECIMAL_COLUMNS = ['Basic (in dollars per share)', 'Diluted (in dollars per share)']

    # Common patterns for columns to drop
    CASHFLOW_DROP_PATTERNS = [
        'shares', 'Shares', 'balances', 'Other', 'vendor', 'Vendor',
        'taxes related to', 'non-trade', 'Non-trade'
    ]

    INCOME_DROP_PATTERNS = [
        'shares', 'Shares', 'Products', 'Services', 'Other'
    ]

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
        
        # Initialize data directory and company-specific directory
        self.PATH = Path.cwd() / 'data'
        self.PATH.mkdir(exist_ok=True)
        self.company_dir = self.PATH / self.ticker
        self.company_dir.mkdir(exist_ok=True)
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

    def _should_drop_column(self, column: str, patterns: List[str]) -> bool:
        """Determine if a column should be dropped based on patterns.
        
        Args:
            column (str): Column name to check
            patterns (List[str]): List of patterns to match against
            
        Returns:
            bool: True if column should be dropped, False otherwise
        """
        return any(pattern.lower() in column.lower() for pattern in patterns)

    def _convert_to_numeric(self, value: str) -> Union[int, float]:
        """Convert string value to appropriate numeric type.
        
        Args:
            value (str): String value to convert
            
        Returns:
            Union[int, float]: Converted numeric value
        """
        try:
            # Remove any commas and parentheses
            cleaned = value.replace(',', '').replace('(', '-').replace(')', '')
            return int(float(cleaned))
        except (ValueError, TypeError):
            return 0

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process dataframe by converting data types appropriately.

        Args:
            df (pd.DataFrame): Input dataframe to process

        Returns:
            pd.DataFrame: Processed dataframe
        """
        try:
            # Create a copy to avoid modifying the original
            processed_df = df.copy()
            
            # Convert all numeric columns to appropriate types
            for col in processed_df.columns:
                if col in self.DECIMAL_COLUMNS:
                    # Keep decimal columns as float
                    processed_df[col] = processed_df[col].astype(float)
                else:
                    # Convert other numeric columns to integers
                    processed_df[col] = processed_df[col].apply(self._convert_to_numeric)
                    
            return processed_df
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
            
            # Drop columns based on patterns
            columns_to_drop = [col for col in df.columns 
                             if self._should_drop_column(col, self.CASHFLOW_DROP_PATTERNS)]
            df = df.drop(columns_to_drop, axis=1)
            
            # Process numeric values
            df = self._process_dataframe(df)

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
            
            # Drop columns based on patterns
            columns_to_drop = [col for col in df.columns 
                             if self._should_drop_column(col, self.INCOME_DROP_PATTERNS)]
            df = df.drop(columns_to_drop, axis=1)
            
            # Process numeric values
            df = self._process_dataframe(df)

            # Calculate growth rates
            if 'Net sales' in df.columns:
                df['Net_Sale_Growth_rate'] = (df['Net sales'].astype(float) / 
                                            df['Net sales'].astype(float).shift(1) - 1) * 100 if 'Net sales' in df.columns else None
            if 'Net income' in df.columns:
                df['Net_Income_Growth_rate'] = (df['Net income'].astype(float) / 
                                              df['Net income'].astype(float).shift(1) - 1) * 100 if 'Net income' in df.columns else None
            
            df.fillna(0, inplace=True)

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
            if self.company.get_filings(form='10-K').latest().xbrl() is not None:
                logger.info(f"Financial structure of company's 10-K: \n{self.company.get_filings(form='10-K').latest().xbrl()}")
                
            return None

    def extract_10k_items(self) -> Optional[Dict[str, str]]:
        """Extract Items 2 and 7 from the latest 10-K filing.

        Returns:
            Optional[Dict[str, str]]: Dictionary containing Items 2 and 7 text or None if extraction fails
        """
        try:
            if self.company is None:
                self.company = self.retrieve_company_info()

            logger.info(f"Retrieving 10-K items for {self.ticker}")
            latest_10k = self.company.get_filings(form="10-K").latest().obj()
            
            items = {
                'Item 2': latest_10k['Item 2'],
                'Item 7': latest_10k['Item 7']
            }

            # Save items to company directory
            for item_name, content in items.items():
                if content:  # Only save if content exists
                    filename = f"{self.ticker}_{item_name.replace(' ', '_')}.txt"
                    filepath = self.company_dir / filename
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    logger.info(f"Saved {item_name} to {filepath}")

            return items

        except Exception as e:
            logger.error(f"Failed to extract 10-K items: {e}")
            return None

    def save_all_to_csv(self) -> bool:
        """Save all dataframes to CSV files in the company-specific directory.

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
                filepath = self.company_dir / filename
                df.to_csv(filepath)
                logger.info(f'Saved {filename}')

            logger.info(f'Successfully saved all CSV files to {self.company_dir}')
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
            
            # Extract and save 10-K items
            items = pipeline.extract_10k_items()
            if items:
                logger.info("Successfully extracted 10-K items")
            
            # Process and save financial statements
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