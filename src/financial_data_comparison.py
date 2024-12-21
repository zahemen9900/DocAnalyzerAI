import pandas as pd
from pathlib import Path
import logging
import traceback
from typing import Optional, Tuple, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from edgar import Company, set_identity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style for visualizations
plt.style.use('seaborn-v0_8-notebook')
sns.set_palette("husl")


class CompareFinancialData:
    """A class to compare financial data between two companies.

    This class handles the comparison of financial statements (income and cashflow) 
    between two companies by merging their data and providing analysis tools.

    Attributes:
        ticker1 (str): First company's ticker symbol
        ticker2 (str): Second company's ticker symbol
        data_dir (Path): Base directory for data
        compare_dir (Path): Directory for saving comparison results
    """

    # Key metrics to visualize
    KEY_METRICS = [
        'Net sales',
        'Net income',
        'Operating income',
        'Gross profit',
        'Operating expenses',
        'Net cash provided by operating activities'
    ]

    def __init__(self, ticker1: str, ticker2: str, credentials: str = "David Yeboah davidzahemenyeboah@gmail.com"):
        """Initialize the CompareFinancialData object.

        Args:
            ticker1 (str): First company's ticker symbol
            ticker2 (str): Second company's ticker symbol
            credentials (str): EDGAR API credentials
        """
        if not all(isinstance(ticker, str) and ticker.strip() for ticker in [ticker1, ticker2]):
            logger.error("Invalid ticker(s) provided")
            raise ValueError("Tickers must be non-empty strings")

        self.ticker1 = ticker1.upper()
        self.ticker2 = ticker2.upper()
        self.credentials = credentials
        
        # Initialize directories
        self.data_dir = Path.cwd() / 'data'
        self.compare_dir = Path.cwd() / f'compare_data_{self.ticker1}_{self.ticker2}'
        self.compare_dir.mkdir(exist_ok=True)
        
        # Create plots directory
        self.plots_dir = self.compare_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized CompareFinancialData for {self.ticker1} and {self.ticker2}")

    def _fetch_company_data(self, ticker: str, data_type: str) -> Optional[pd.DataFrame]:
        """Fetch financial data directly from EDGAR if file not found.

        Args:
            ticker (str): Company ticker symbol
            data_type (str): Type of data to fetch ('income' or 'cashflow')

        Returns:
            Optional[pd.DataFrame]: Fetched dataframe or None if fetching fails
        """
        try:
            set_identity(self.credentials)
            company = Company(ticker)
            
            if data_type == 'income':
                df = company.financials.income.to_dataframe().T
            else:  # cashflow
                df = company.financials.cashflow.to_dataframe().T
                
            logger.info(f"Successfully fetched {data_type} data for {ticker} from EDGAR")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch {data_type} data for {ticker} from EDGAR: {e}")
            return None

    def _load_company_data(self, ticker: str, data_type: str) -> Optional[pd.DataFrame]:
        """Load financial data for a company, fetching from EDGAR if file not found.

        Args:
            ticker (str): Company ticker symbol
            data_type (str): Type of data to load ('income' or 'cashflow')

        Returns:
            Optional[pd.DataFrame]: Loaded dataframe or None if loading fails
        """
        try:
            company_dir = self.data_dir / ticker
            filename = f"{data_type}_{ticker}.csv"
            filepath = company_dir / filename
            
            if filepath.exists():
                df = pd.read_csv(filepath, index_col=0)
                logger.info(f"Successfully loaded {data_type} data for {ticker} from file")
                return df
            else:
                logger.info(f"File not found: {filepath}, attempting to fetch from EDGAR")
                return self._fetch_company_data(ticker, data_type)
            
        except Exception as e:
            logger.error(f"Failed to load {data_type} data for {ticker}: {e}")
            return None

    def _prepare_comparison_df(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                             ticker1: str, ticker2: str) -> pd.DataFrame:
        """Prepare dataframes for comparison by renaming columns and aligning lengths.

        Args:
            df1 (pd.DataFrame): First company's dataframe
            df2 (pd.DataFrame): Second company's dataframe
            ticker1 (str): First company's ticker
            ticker2 (str): Second company's ticker

        Returns:
            pd.DataFrame: Merged and processed dataframe
        """
        # Rename columns to include company tickers
        df1 = df1.copy()
        df2 = df2.copy()
        
        df1.columns = [f"{col}_{ticker1}" for col in df1.columns]
        df2.columns = [f"{col}_{ticker2}" for col in df2.columns]
        
        # Align lengths to the shorter dataframe
        min_len = min(len(df1), len(df2))
        df1 = df1.iloc[:min_len]
        df2 = df2.iloc[:min_len]
        
        # Merge dataframes
        merged_df = pd.concat([df1, df2], axis=1)
        return merged_df

    def _calculate_growth_rates(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Calculate year-over-year growth rates for key metrics.

        Args:
            df (pd.DataFrame): Input dataframe
            ticker (str): Company ticker

        Returns:
            pd.DataFrame: Dataframe with growth rates
        """
        metrics = ['Net sales', 'Net income', 'Operating income']
        growth_df = pd.DataFrame()
        
        for metric in metrics:
            col_name = f"{metric}_{ticker}"
            if col_name in df.columns:
                growth_rate = df[col_name].pct_change() * 100
                growth_df[f"{metric}_Growth_{ticker}"] = growth_rate
        
        return growth_df

    def _plot_comparison(self, df: pd.DataFrame, metric: str, 
                        save_path: Optional[Path] = None) -> None:
        """Create an enhanced comparison plot for a specific metric.

        Args:
            df (pd.DataFrame): Data to plot
            metric (str): Metric to compare
            save_path (Optional[Path]): Path to save the plot
        """
        col1 = f"{metric}_{self.ticker1}"
        col2 = f"{metric}_{self.ticker2}"
        
        if col1 in df.columns and col2 in df.columns:
            # Create figure with higher DPI for better quality
            plt.figure(figsize=(12, 7), dpi=300)
            
            # Create the plot with enhanced styling
            years = df.index.astype(str)
            
            # Plot lines with markers
            plt.plot(years, df[col1], marker='o', linewidth=2.5, 
                    markersize=8, label=self.ticker1)
            plt.plot(years, df[col2], marker='s', linewidth=2.5, 
                    markersize=8, label=self.ticker2)
            
            # Add points annotations
            for i, (v1, v2) in enumerate(zip(df[col1], df[col2])):
                plt.annotate(f'{v1:,.0f}', (years[i], v1), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=8)
                plt.annotate(f'{v2:,.0f}', (years[i], v2), textcoords="offset points", 
                           xytext=(0,-15), ha='center', fontsize=8)
            
            # Enhance the plot appearance
            plt.title(f"{metric} Comparison\n{self.ticker1} vs {self.ticker2}", 
                     fontsize=14, pad=20)
            plt.xlabel("Year", fontsize=12, labelpad=10)
            plt.ylabel("Value (USD)", fontsize=12, labelpad=10)
            
            # Customize grid
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Enhance legend
            plt.legend(title="Companies", title_fontsize=12, 
                      fontsize=10, loc='upper left', 
                      bbox_to_anchor=(1.02, 1))
            
            # Rotate x-axis labels
            plt.xticks(rotation=45)
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            
            # Add a subtle background color
            ax = plt.gca()
            ax.set_facecolor('#f8f9fa')
            
            # Format y-axis labels with comma separator
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()

    def compare_statements(self, statement_type: str = 'income') -> Optional[pd.DataFrame]:
        """Compare financial statements between the two companies.

        Args:
            statement_type (str): Type of statement to compare ('income' or 'cashflow')

        Returns:
            Optional[pd.DataFrame]: Merged comparison dataframe or None if comparison fails
        """
        try:
            # Validate statement type
            if statement_type not in ['income', 'cashflow']:
                raise ValueError("statement_type must be either 'income' or 'cashflow'")

            # Load data for both companies
            df1 = self._load_company_data(self.ticker1, statement_type)
            df2 = self._load_company_data(self.ticker2, statement_type)

            if df1 is None or df2 is None:
                return None

            # Prepare comparison dataframe
            comparison_df = self._prepare_comparison_df(df1, df2, self.ticker1, self.ticker2)
            
            # Calculate growth rates
            growth_rates = pd.concat([
                self._calculate_growth_rates(comparison_df, self.ticker1),
                self._calculate_growth_rates(comparison_df, self.ticker2)
            ], axis=1)
            
            # Add growth rates to comparison dataframe
            comparison_df = pd.concat([comparison_df, growth_rates], axis=1)

            # Create comparison plots for key metrics
            for metric in self.KEY_METRICS:
                plot_path = self.plots_dir / f"{metric.lower().replace(' ', '_')}_comparison.png"
                self._plot_comparison(comparison_df, metric, plot_path)

            # Save comparison results
            output_path = self.compare_dir / f"{statement_type}_comparison_{self.ticker1}_{self.ticker2}.csv"
            comparison_df.to_csv(output_path)
            logger.info(f"Saved comparison results to {output_path}")

            # Display the first few rows
            display(comparison_df.head())
            
            return comparison_df

        except Exception as e:
            logger.error(f"Failed to compare {statement_type} statements: {e}")
            return None

    def generate_summary_report(self) -> Optional[Dict]:
        """Generate a summary report comparing key metrics between the companies.

        Returns:
            Optional[Dict]: Dictionary containing summary statistics or None if generation fails
        """
        try:
            income_df = self.compare_statements('income')
            cashflow_df = self.compare_statements('cashflow')
            
            if income_df is None or cashflow_df is None:
                return None

            # Find matching columns between companies
            def get_matching_columns(df: pd.DataFrame) -> Dict[str, Tuple[str, str]]:
                """Find columns that match between companies but have different ticker suffixes."""
                columns = df.columns
                matches = {}
                
                # Get base names (without ticker suffixes)
                for col in columns:
                    if col.endswith(f"_{self.ticker1}"):
                        base_name = col[:-len(f"_{self.ticker1}")]
                        matching_col = f"{base_name}_{self.ticker2}"
                        if matching_col in columns:
                            matches[base_name] = (col, matching_col)
                
                return matches

            # Get matching columns for both statement types
            income_matches = get_matching_columns(income_df)
            cashflow_matches = get_matching_columns(cashflow_df)

            # Calculate statistics for matching columns
            summary = {
                'Income Statement Analysis': {},
                'Cash Flow Analysis': {}
            }

            # Process income statement metrics
            for base_name, (col1, col2) in income_matches.items():
                try:
                    summary['Income Statement Analysis'][base_name] = {
                        self.ticker1: float(income_df[col1].mean()),
                        self.ticker2: float(income_df[col2].mean()),
                        'Difference': float(income_df[col1].mean() - income_df[col2].mean()),
                        'Ratio': float(income_df[col1].mean() / income_df[col2].mean()) if income_df[col2].mean() != 0 else None
                    }
                except Exception as e:
                    logger.warning(f"Could not process {base_name}: {str(e)}")
                    continue

            # Process cashflow statement metrics
            for base_name, (col1, col2) in cashflow_matches.items():
                try:
                    summary['Cash Flow Analysis'][base_name] = {
                        self.ticker1: float(cashflow_df[col1].mean()),
                        self.ticker2: float(cashflow_df[col2].mean()),
                        'Difference': float(cashflow_df[col1].mean() - cashflow_df[col2].mean()),
                        'Ratio': float(cashflow_df[col1].mean() / cashflow_df[col2].mean()) if cashflow_df[col2].mean() != 0 else None
                    }
                except Exception as e:
                    logger.warning(f"Could not process {base_name}: {str(e)}")
                    continue

            # Save summary report
            output_path = self.compare_dir / f"summary_report_{self.ticker1}_{self.ticker2}.txt"
            with open(output_path, 'w') as f:
                f.write(f"Financial Comparison Summary Report\n")
                f.write(f"{self.ticker1} vs {self.ticker2}\n")
                f.write(f"{'='*50}\n\n")

                for category, metrics in summary.items():
                    if metrics:  # Only write category if it has metrics
                        f.write(f"\n{category}:\n")
                        f.write(f"{'-'*50}\n")
                        
                        for metric_name, values in metrics.items():
                            if values:  # Only write metric if it has values
                                f.write(f"\n{metric_name}:\n")
                                for key, value in values.items():
                                    if value is not None:  # Only write if value exists
                                        if isinstance(value, float):
                                            if key == 'Ratio':
                                                f.write(f"  {key}: {value:.2f}x\n")
                                            else:
                                                f.write(f"  {key}: {value:,.2f}\n")
                                        else:
                                            f.write(f"  {key}: {value}\n")
                                f.write("\n")

            logger.info(f"Generated summary report at {output_path}")
            return summary

        except Exception as e:
            logger.error(f"Failed to generate summary report: {e}")
            logger.error(traceback.format_exc())
            return None


def main():
    try:
        # Example usage
        comparer = CompareFinancialData('TSLA', 'AAPL')
        
        # Compare income statements
        income_comparison = comparer.compare_statements('income')
        if income_comparison is not None:
            logger.info("Successfully compared income statements")
        
        # Compare cashflow statements
        cashflow_comparison = comparer.compare_statements('cashflow')
        if cashflow_comparison is not None:
            logger.info("Successfully compared cashflow statements")
        
        # Generate summary report
        summary = comparer.generate_summary_report()
        if summary is not None:
            logger.info("Successfully generated summary report")
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")


if __name__ == "__main__":
    main() 